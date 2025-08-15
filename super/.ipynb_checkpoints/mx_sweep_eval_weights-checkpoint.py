# ----------------------------------------------------------------------
#  Imports & global configuration
# ----------------------------------------------------------------------
import math, warnings
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.ndimage import zoom
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DATA   = "/pscratch/sd/h/hbassi"
ROOT_MODELS = f"{ROOT_DATA}/models/"
ROOT_STATS  = "./data"

OUT_CSV_FMT     = "mx_sweep_errors_trainmx{train_mx}.csv"
OUT_FIG_FMT     = "mx_sweep_relative_errors_trainmx{train_mx}.pdf"
SPECTRAL_FIG    = "spectral_norm_per_layer_trainmx4_vs_8.pdf"
NTK_EIG_FIG     = "ntk_eig_spectrum_trainmx4_vs_8.pdf"
NTK_KERNEL_FIG  = "ntk_kernel_heatmap_trainmx4_vs_8.pdf"

mx_vals  = list(range(4, 9))   # test-set resolutions
scale_x  = 4                   # 32 → 128 up-factor
device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps      = 1e-8                # ℓ₂ denom regulariser


# ----------------------------------------------------------------------
#  Model definition (unchanged)
# ----------------------------------------------------------------------
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * scale)
    def forward(self, coords):
        proj = 2 * math.pi * torch.matmul(coords, self.B)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)

def get_coord_grid(b, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

class FourierLayer(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat) /
            (in_ch * out_ch)
        )
    @staticmethod
    def compl_mul2d(x, w): return torch.einsum("bixy,ioxy->boxy", x, w)
    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1, m2 = min(self.modes1, H), min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(B, self.weight.size(1), H, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GELU())
    def forward(self, x): return self.block(x)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, up=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch*up*up, 3, padding=1)
        self.pix  = nn.PixelShuffle(up)
        self.act  = nn.GELU()
    def forward(self, x): return self.act(self.pix(self.conv(x)))

class SuperResUNet(nn.Module):
    def __init__(self, in_channels=101, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0, final_scale=4):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, 1)
        self.enc1 = ConvBlock(lift_dim, lift_dim)
        self.enc2 = ConvBlock(lift_dim, lift_dim*2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim*2, lift_dim*2),
            FourierLayer(lift_dim*2, lift_dim*2, 32, 32),
            nn.GELU())
        self.up1  = PixelShuffleUpsample(lift_dim*2, lift_dim*2, 1)
        self.dec2 = ConvBlock(lift_dim*4, lift_dim)
        self.up2  = PixelShuffleUpsample(lift_dim,   lift_dim)
        self.dec1 = ConvBlock(lift_dim*2, lift_dim//2)
        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim//2, lift_dim//4, final_scale),
            ConvBlock(lift_dim//4, lift_dim//4))
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim//4, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1))
    def forward(self, x):
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)
        x = self.lift(x)
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(e2)
        d2 = self.up1(b); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up2(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = self.dec0(d1)
        return self.out_head(d0)

# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------
def moments_1d(f):
    Nv = f.shape[1]; v = np.linspace(-1, 1, Nv, endpoint=False)
    return f.sum(1), (f*v).sum(1), (f*v**2).sum(1)

# ----------------------------------------------------------------------
#  1) Relative-error sweep
# ----------------------------------------------------------------------
def run_sweep(train_mx: int):
    stats_path = (f"{ROOT_STATS}/2d_vlasov_funet_phase1_stats_32to128_"
                  f"mx={train_mx}_my={train_mx}_v1.pt")
    model_path = (f"{ROOT_MODELS}/2d_vlasov_mx={train_mx}my={train_mx}_"
                  "FUnet_best_PS_FT_32to128_1k_t=101.pth")
    stats = torch.load(stats_path, map_location=device)
    μ, σ = stats["data_mean"].squeeze(0), stats["data_std"].squeeze(0)
    net = SuperResUNet(final_scale=4).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device)); net.eval()

    agg = {k: [] for k in ("f_mean","f_std","f_up",
                           "rho_mean","rho_std","rho_up",
                           "J_mean","J_std","J_up",
                           "M_mean","M_std","M_up")}

    for mx in mx_vals:
        coarse = (f"{ROOT_DATA}/2d_vlasov_multi_traj_coarse_32_fixed_timestep_"
                  f"mx={mx}_my={mx}_phase1_test_data.npy")
        fine   = (f"{ROOT_DATA}/2d_vlasov_multi_traj_fine_128_fixed_timestep_"
                  f"mx={mx}_my={mx}_phase1_test_data.npy")
        u_c = torch.tensor(np.load(coarse)[:, :101], device=device)
        u_f = torch.tensor(np.load(fine)  [:, :101], device=device)
        with torch.no_grad():
            pred = net((u_c.float()-μ)/σ); pred = (pred*σ+μ).cpu()
        N_cases, N_t = u_c.shape[:2]
        eF,eR,eJ,eM,uF,uR,uJ,uM = [],[],[],[],[],[],[],[]
        for c in trange(N_cases, desc=f"sweep trainmx={train_mx}, mx={mx}"):
            ef,er,ej,em,uf,ur,uj,um=[],[],[],[],[],[],[],[]
            for t in range(N_t):
                f_cg=u_c[c,t].cpu().numpy(); f_up=zoom(f_cg,scale_x,order=3)
                f_pr=pred[c,t].numpy();      f_gt=u_f[c,t].cpu().numpy()
                ef.append(np.linalg.norm(f_pr-f_gt)/(np.linalg.norm(f_gt)+eps))
                uf.append(np.linalg.norm(f_up-f_gt)/(np.linalg.norm(f_gt)+eps))
                ρpr,Jpr,M2pr = moments_1d(f_pr)
                ρup,Jup,M2up = moments_1d(f_up)
                ρgt,Jgt,M2gt = moments_1d(f_gt)
                er.append(np.linalg.norm(ρpr-ρgt)/(np.linalg.norm(ρgt)+eps))
                ej.append(np.linalg.norm(Jpr-Jgt)/(np.linalg.norm(Jgt)+eps))
                em.append(np.linalg.norm(M2pr-M2gt)/(np.linalg.norm(M2gt)+eps))
                ur.append(np.linalg.norm(ρup-ρgt)/(np.linalg.norm(ρgt)+eps))
                uj.append(np.linalg.norm(Jup-Jgt)/(np.linalg.norm(Jgt)+eps))
                um.append(np.linalg.norm(M2up-M2gt)/(np.linalg.norm(M2gt)+eps))
            eF.append(np.mean(ef)); eR.append(np.mean(er)); eJ.append(np.mean(ej)); eM.append(np.mean(em))
            uF.append(np.mean(uf)); uR.append(np.mean(ur)); uJ.append(np.mean(uj)); uM.append(np.mean(um))
        agg["f_mean"].append(np.mean(eF));   agg["f_std"].append(np.std(eF));   agg["f_up"].append(np.mean(uF))
        agg["rho_mean"].append(np.mean(eR)); agg["rho_std"].append(np.std(eR)); agg["rho_up"].append(np.mean(uR))
        agg["J_mean"].append(np.mean(eJ));   agg["J_std"].append(np.std(eJ));   agg["J_up"].append(np.mean(uJ))
        agg["M_mean"].append(np.mean(eM));   agg["M_std"].append(np.std(eM));   agg["M_up"].append(np.mean(uM))

    pd.DataFrame({"mx":mx_vals,
                  "f_pred":agg["f_mean"],"f_pred_std":agg["f_std"],"f_up":agg["f_up"],
                  "rho_pred":agg["rho_mean"],"rho_pred_std":agg["rho_std"],"rho_up":agg["rho_up"],
                  "J_pred":agg["J_mean"],"J_pred_std":agg["J_std"],"J_up":agg["J_up"],
                  "M_pred":agg["M_mean"],"M_pred_std":agg["M_std"],"M_up":agg["M_up"]}
                 ).to_csv(OUT_CSV_FMT.format(train_mx=train_mx), index=False)

    plt.figure(figsize=(9,6)); colors=['C0','C1','C2','C3']; lw=2
    plt.errorbar(mx_vals,agg["f_mean"], yerr=agg["f_std"], marker='o', lw=lw,
                 color=colors[0], label=r'$f_e$')
    plt.errorbar(mx_vals,agg["rho_mean"],yerr=agg["rho_std"],marker='s', lw=lw,
                 color=colors[1], label=r'$\rho$')
    plt.errorbar(mx_vals,agg["J_mean"],  yerr=agg["J_std"], marker='^', lw=lw,
                 color=colors[2], label=r'$J$')
    plt.errorbar(mx_vals,agg["M_mean"],  yerr=agg["M_std"], marker='d', lw=lw,
                 color=colors[3], label=r'$M$')
    plt.plot(mx_vals,agg["f_up"], '--o', color=colors[0])
    plt.plot(mx_vals,agg["rho_up"],'--s', color=colors[1])
    plt.plot(mx_vals,agg["J_up"],  '--^', color=colors[2])
    plt.plot(mx_vals,agg["M_up"],  '--d', color=colors[3])
    plt.yscale('log'); plt.grid(ls='--', alpha=.5)
    plt.ylim([1e-3, 1e0])
    plt.xlabel(r'$\mathbf{m_x,m_v}$', fontsize=14, fontweight='bold')
    plt.ylabel('error', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(fr'0 shot generalization (training $m_x$={train_mx})',
              fontsize=16, fontweight='bold')
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_FIG_FMT.format(train_mx=train_mx), dpi=300); plt.close()


# ----------------------------------------------------------------------
#  2) Per-layer spectral norms
# ----------------------------------------------------------------------
def spectral_norm_of_module(mod: nn.Module):
    if not hasattr(mod, 'weight'): return None
    W = mod.weight.data
    if torch.is_complex(W): W = torch.view_as_real(W).norm(dim=-1)
    W = W.reshape(W.shape[0], -1)
    u = torch.randn(W.shape[0], 1, device=W.device)
    for _ in range(8):
        v = (W.T @ u); v /= v.norm() + 1e-12
        u = (W @ v);   u /= u.norm() + 1e-12
    return float(abs((u.T @ (W @ v)).item()))

def spectral_norm_figure():
    names, norm_low, norm_high = None, None, None
    for mx in (4, 8):
        net = SuperResUNet(final_scale=4).to(device)
        net.load_state_dict(torch.load(
            f"{ROOT_MODELS}/2d_vlasov_mx={mx}my={mx}_"
            "FUnet_best_PS_FT_32to128_1k_t=101.pth", map_location=device))
        vals, layer_names = [], []
        for n, m in net.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, FourierLayer)):
                s = spectral_norm_of_module(m)
                if s is not None: layer_names.append(n); vals.append(math.log10(max(s,1e-12)))
        if mx == 4: names, norm_low = layer_names, vals
        else:       norm_high = vals
    x = np.arange(len(names))
    plt.figure(figsize=(max(10, 0.7*len(names)), 5))
    plt.plot(x, norm_low,  '-o', label='Per layer: low-freq',  color='C0')
    plt.plot(x, norm_high, '-o', label='Per layer: high-freq', color='C1')
    plt.axhline(np.mean(norm_low),  ls='--', color='C0', label='Mean low-freq')
    plt.axhline(np.mean(norm_high), ls='--', color='C1', label='Mean high-freq')
    plt.ylabel('log(Weight spectral norm)', fontsize=14, fontweight='bold')
    plt.xticks(x, names, rotation=60, ha='right', fontsize=9, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(axis='y', ls='--', alpha=.4)
    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(SPECTRAL_FIG, dpi=300); plt.close()


# ------------------------------------------------------------------
# 3) & 4)  NTK eigenspectrum + kernel heat-maps
# ------------------------------------------------------------------
TARGET_PARAM = "out_head.2.weight"
N_SAMPLES    = 16                    # number of real snapshots

def grab_target_param(net):
    for n, p in net.named_parameters():
        if n.endswith(TARGET_PARAM):
            return p
    raise KeyError(f"{TARGET_PARAM} not found")
    
def sample_real_inputs(mx: int, n: int):
    """
    Returns:
        samples : list of n tensors  (1,101,32,32)  on CPU (unnormalised)
        μ, σ    : normalisation stats on DEVICE
    """
    data_path  = (f"{ROOT_DATA}/2d_vlasov_multi_traj_coarse_32_fixed_timestep_"
                  f"mx={mx}_my={mx}_phase1_test_data.npy")
    stats_path = (f"{ROOT_STATS}/2d_vlasov_funet_phase1_stats_32to128_"
                  f"mx={mx}_my={mx}_v1.pt")

    data  = np.load(data_path)[:, :101]    # (N_case, 101, 32, 32)
    μ, σ  = (torch.load(stats_path, map_location=device)[k].squeeze(0)
             for k in ("data_mean", "data_std"))

    case_idx = np.random.choice(data.shape[0], n, replace=False)
    tensors  = []
    for idx in case_idx:
        arr = torch.tensor(data[idx], dtype=torch.float32)   # (101,32,32)
        tensors.append(arr.unsqueeze(0))                     # (1,101,32,32)
    return tensors, μ.to(device), σ.to(device)

def build_ntk(net, samples, μ, σ):
    """
    Build N×N NTK for TARGET_PARAM on samples.
    samples : list of (1,101,32,32) CPU tensors (unnormalised)
    """
    param = grab_target_param(net)        # complex or real weight tensor
    grads = []

    for x_cpu in samples:
        x = (x_cpu.to(device) - μ) / σ
        net.zero_grad()
        net(x).mean().backward()          # populate param.grad
        g = param.grad.detach()
        if torch.is_complex(g):
            g = torch.view_as_real(g)     # (..., 2) → real tensor
        grads.append(g.flatten().cpu().numpy())  # no shape hazard

    J = np.stack(grads)                   # (N, P_real)
    return J @ J.T                        # NTK matrix (N × N)



def ntk_figures():
    eigs, kernels = {}, {}
    for mx in (4, 8):
        # 1️⃣  model
        net = SuperResUNet(final_scale=4).to(device)
        net.load_state_dict(torch.load(
            f"{ROOT_MODELS}/2d_vlasov_mx={mx}my={mx}_"
            "FUnet_best_PS_FT_32to128_1k_t=101.pth",
            map_location=device))
        net.eval()

        # 2️⃣  real snapshots + stats
        samples, μ, σ = sample_real_inputs(mx, N_SAMPLES)

        # 3️⃣  NTK
        K = build_ntk(net, samples, μ, σ)
        kernels[mx] = K
        eigs[mx] = np.sort(np.linalg.eigvalsh(K))[::-1]

    # ---------- eigenspectrum ----------
    idx = np.arange(1, len(eigs[4]) + 1)
    plt.figure(figsize=(6,4))
    plt.loglog(idx, eigs[4], '-o', label='low-res train',  color='C0')
    plt.loglog(idx, eigs[8], '-o', label='high-res train', color='C1')
    plt.xlabel('Eigenvalue index', fontsize=12, fontweight='bold'); 
    plt.ylabel('Eigenvalue', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold'); plt.yticks(fontsize=10, fontweight='bold')
    plt.title(f'NTK eigenspectrum – {TARGET_PARAM}', fontsize=14, fontweight='bold')
    plt.grid(ls='--', alpha=.5); plt.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(NTK_EIG_FIG, dpi=300); plt.close()

    # ---------- kernel heat-map ----------
    vmax = max(kernels[4].max(), kernels[8].max())
    fig, axs = plt.subplots(1,2, figsize=(6,3))
    for ax, K, lab, col in zip(
            axs, [kernels[4], kernels[8]],
            ['low-freq train', 'high-freq train'], ['C0', 'C1']):
        im = ax.imshow(K, origin='lower', cmap='magma')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab, fontsize=10, fontweight='bold', color=col)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=.8, label='Kernel value')
    plt.suptitle(f'NTK kernel heat-map – {TARGET_PARAM}', fontsize=13, fontweight='bold')
    #plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(NTK_KERNEL_FIG, dpi=300); plt.close()

# ------------------------------------------------------------------
#  Produce NTK eigenspectrum + heat–map for multiple layers
# ------------------------------------------------------------------
LAYERS = ["bottleneck.1.weight",           # Fourier mixing layer
          "dec0.0.conv.weight"]            # first conv after 4× up-shuffle

NTK_EIG_FIG_FMT    = "ntk_eig_{layer}_trainmx4_vs_8.pdf"
NTK_KERNEL_FIG_FMT = "ntk_kernel_{layer}_trainmx4_vs_8.pdf"

def ntk_for_layer(layer_suffix: str):
    global TARGET_PARAM
    TARGET_PARAM = layer_suffix              # re-use all existing helpers
    eigs, kernels = {}, {}

    for mx in (4, 8):                        # 4 → low-res train, 8 → high-res
        net = SuperResUNet(final_scale=4).to(device)
        net.load_state_dict(torch.load(
            f"{ROOT_MODELS}/2d_vlasov_mx={mx}my={mx}_"
            "FUnet_best_PS_FT_32to128_1k_t=101.pth",
            map_location=device))
        net.eval()

        samples, μ, σ = sample_real_inputs(mx, N_SAMPLES)
        kernels[mx] = build_ntk(net, samples, μ, σ)
        eigs[mx]    = np.sort(np.linalg.eigvalsh(kernels[mx]))[::-1]

    # --- eigenspectrum ------------------------------------------------
    idx = np.arange(1, len(eigs[4]) + 1)
    plt.figure(figsize=(6,4))
    plt.loglog(idx, eigs[4], '-o', label='low-freq train',  color='C0')
    plt.loglog(idx, eigs[8], '-o', label='high-freq train', color='C1')
    plt.xlabel('Eigenvalue index', fontsize=12, fontweight='bold');  
    plt.ylabel('Eigenvalue', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold'); plt.yticks(fontsize=10, fontweight='bold')
    plt.title(f'NTK eigenspectrum – {layer_suffix}', fontsize=14, fontweight='bold')
    plt.grid(ls='--', alpha=.5);  plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(NTK_EIG_FIG_FMT.format(layer=layer_suffix.replace('.','_')),
                dpi=300);  plt.close()

    # --- heat-map -----------------------------------------------------
    vmax = max(kernels[4].max(), kernels[8].max())
    fig, axs = plt.subplots(1,2, figsize=(6,3))
    for ax, K, lab, col in zip(
            axs, [kernels[4], kernels[8]],
            ['low-freq train', 'high-freq train'], ['C0', 'C1']):
        im = ax.imshow(K, origin='lower', cmap='magma')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab, fontsize=10, fontweight='bold', color=col)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=.8, label='magnitude')
    plt.suptitle(f'NTK – {layer_suffix}', fontsize=13, fontweight='bold')
    #plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(NTK_KERNEL_FIG_FMT.format(layer=layer_suffix.replace('.','_')),
                dpi=300);  plt.close()

# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    for train_mx in (4, 8):
        run_sweep(train_mx)
    spectral_norm_figure()
    LAYERS = ["bottleneck.1.weight",           # Fourier mixing layer
           "enc2.block.0.weight"]            # first conv after 4× up-shuffle
    for layer in LAYERS:
        ntk_for_layer(layer)
    ntk_figures()
