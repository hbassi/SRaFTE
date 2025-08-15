#!/usr/bin/env python
# ============================================================
#  mx-sweep evaluation for the 2-D Vlasov Super-Resolution model
#
#  • Compares the learned SuperResUNet (solid lines)
#    against naïve bicubic up-sampling (dashed lines).
#  • Reports **relative ℓ₂ errors** for:
#      – full phase-space field  (fₑ)
#      – 0th, 1st, 2nd velocity moments (ρ, J, M)
# ============================================================
import os, math, glob
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.ndimage import zoom
import pandas as pd

# ------------------------- paths / config --------------------
ROOT_DATA   = "/pscratch/sd/h/hbassi"
ROOT_MODELS = f"{ROOT_DATA}/models/"
ROOT_STATS  = "./data"
OUT_FIG     = "mx_sweep_relative_errors.pdf"
OUT_CSV     = "mx_sweep_errors.csv"

mx_vals  = list(range(4, 9))          # 4 … 8
epochs   = [1200, 1500, 1800]   #
scale_x  = 4
device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps      = 1e-8

# -------------------------------------------------------------
#                >>> model definition (unchanged) <<<
# -------------------------------------------------------------
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * scale)
    def forward(self, coords):
        proj = 2 * math.pi * torch.matmul(coords, self.B)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)

def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)

class FourierLayer(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            / (in_ch * out_ch)
        )
    @staticmethod
    def compl_mul2d(inp, w):
        return torch.einsum("bixy,ioxy->boxy", inp, w)
    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1   = min(self.modes1, H)
        m2   = min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(B, self.weight.size(1), H, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GELU()
        )
    def forward(self, x): return self.block(x)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * upscale * upscale, 3, padding=1)
        self.pix  = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self, x): return self.act(self.pix(self.conv(x)))

class SuperResUNet(nn.Module):
    def __init__(self, in_channels=101, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0, final_scale=4):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, 1)

        self.enc1 = ConvBlock(lift_dim,        lift_dim)
        self.enc2 = ConvBlock(lift_dim,        lift_dim*2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim*2, lift_dim*2),
            FourierLayer(lift_dim*2, lift_dim*2, 32, 32),
            nn.GELU()
        )

        self.up1  = PixelShuffleUpsample(lift_dim*2, lift_dim*2, 1)
        self.dec2 = ConvBlock(lift_dim*4, lift_dim)
        self.up2  = PixelShuffleUpsample(lift_dim,   lift_dim)
        self.dec1 = ConvBlock(lift_dim*2, lift_dim//2)
        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim//2, lift_dim//4, final_scale),
            ConvBlock(lift_dim//4, lift_dim//4)
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim//4, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )
    def forward(self, x):
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)
        x = self.lift(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(e2)
        d2 = self.up1(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = self.dec0(d1)
        return self.out_head(d0)

# -------------------------------------------------------------
def moments_1d(f):
    Nv = f.shape[1]
    v  = np.linspace(-1,1,Nv,endpoint=False)
    rho = f.sum(1)
    J   = (f*v).sum(1)
    M2  = (f*v**2).sum(1)
    return rho,J,M2

# ------------------------- result containers ------------------
flow_mean, flow_std, flow_up = [], [], []
rho_mean,  rho_std,  rho_up = [], [], []
J_mean,    J_std,    J_up   = [], [], []
M_mean,    M_std,    M_up   = [], [], []

# =============================================================
#                  MAIN EVALUATION LOOP
# =============================================================
for mx in mx_vals:
    my = mx
    print(f"\n=== Evaluating mx = my = {mx} ===")

    coarse = (f"{ROOT_DATA}/2d_vlasov_multi_traj_coarse_32_fixed_timestep_"
              f"mx={mx}_my={my}_phase1_test_data.npy")
    fine   = (f"{ROOT_DATA}/2d_vlasov_multi_traj_fine_128_fixed_timestep_"
              f"mx={mx}_my={my}_phase1_test_data.npy")

    u_c = torch.tensor(np.load(coarse)[:, :101], device=device)
    u_f = torch.tensor(np.load(fine)  [:, :101], device=device)

    stats = torch.load(f"{ROOT_STATS}/2d_vlasov_funet_phase1_stats_32to128_mx=8_my=8_v1.pt",
                       map_location=device)
    μ, σ = stats["data_mean"].squeeze(0), stats["data_std"].squeeze(0)

    # ---------------- ensemble prediction ---------------------
    preds = []
    with torch.no_grad():
        for ep in epochs:
            print(f"  → model epoch {ep}")
            net = SuperResUNet(final_scale=4).to(device)
            ckpt = (f"{ROOT_MODELS}/2d_vlasov_mx=8my=8_"
                    f"FUnet_ckpt_{ep:04d}_PS_FT_32to128_1k_t=101.pth")
            net.load_state_dict(torch.load(ckpt,map_location=device)["model_state_dict"])
            net.eval()
            p = net((u_c.float()-μ)/σ); preds.append((p*σ+μ).cpu())
    mean_pred = torch.stack(preds).mean(0)

    # ----------------- per-case errors ------------------------
    N_cases, N_t = u_c.shape[:2]
    e_flow, e_rho, e_J, e_M = [], [], [], []
    up_flow, up_rho, up_J, up_M = [], [], [], []

    for c in trange(N_cases, desc=f"mx={mx} cases"):
        ef, er, ej, em = [], [], [], []
        uf, ur, uj, um = [], [], [], []
        for t in range(N_t):
            f_cg = u_c[c,t].cpu().numpy()
            f_up = zoom(f_cg, scale_x, order=3)
            f_pr = mean_pred[c,t].numpy()
            f_gt = u_f[c,t].cpu().numpy()

            # full-field ℓ₂
            ef.append(np.linalg.norm(f_pr-f_gt)/(np.linalg.norm(f_gt)+eps))
            uf.append(np.linalg.norm(f_up-f_gt)/(np.linalg.norm(f_gt)+eps))

            # moments
            ρ_pr, J_pr, M2_pr = moments_1d(f_pr)
            ρ_up, J_up_, M2_up = moments_1d(f_up)
            ρ_gt, J_gt, M2_gt = moments_1d(f_gt)

            er.append(np.linalg.norm(ρ_pr-ρ_gt)/(np.linalg.norm(ρ_gt)+eps))
            ej.append(np.linalg.norm(J_pr-J_gt)/(np.linalg.norm(J_gt)+eps))
            em.append(np.linalg.norm(M2_pr-M2_gt)/(np.linalg.norm(M2_gt)+eps))

            ur.append(np.linalg.norm(ρ_up-ρ_gt)/(np.linalg.norm(ρ_gt)+eps))
            uj.append(np.linalg.norm(J_up_-J_gt)/(np.linalg.norm(J_gt)+eps))
            um.append(np.linalg.norm(M2_up-M2_gt)/(np.linalg.norm(M2_gt)+eps))

        # average over time for this case
        e_flow.append(np.mean(ef)); e_rho.append(np.mean(er)); e_J.append(np.mean(ej)); e_M.append(np.mean(em))
        up_flow.append(np.mean(uf)); up_rho.append(np.mean(ur)); up_J.append(np.mean(uj)); up_M.append(np.mean(um))

    # aggregate across cases
    flow_mean.append(np.mean(e_flow)); flow_std.append(np.std(e_flow)); flow_up.append(np.mean(up_flow))
    rho_mean .append(np.mean(e_rho));  rho_std .append(np.std(e_rho));  rho_up .append(np.mean(up_rho))
    J_mean   .append(np.mean(e_J));    J_std   .append(np.std(e_J));    J_up   .append(np.mean(up_J))
    M_mean   .append(np.mean(e_M));    M_std   .append(np.std(e_M));    M_up   .append(np.mean(up_M))

    print(f"fₑ : {flow_mean[-1]:.4e} ± {flow_std[-1]:.1e}   |  bicubic = {flow_up[-1]:.4e}")
    print(f"ρ  : {rho_mean [-1]:.4e} ± {rho_std [-1]:.1e}   |  bicubic = {rho_up [-1]:.4e}")
    print(f"J  : {J_mean   [-1]:.4e} ± {J_std   [-1]:.1e}   |  bicubic = {J_up   [-1]:.4e}")
    print(f"M  : {M_mean  [-1]:.4e} ± {M_std  [-1]:.1e}   |  bicubic = {M_up  [-1]:.4e}")

# ---------------------- save CSV ------------------------------
df = pd.DataFrame({
    "mx": mx_vals,
    "f_pred": flow_mean, "f_pred_std": flow_std, "f_up": flow_up,
    "rho_pred": rho_mean, "rho_pred_std": rho_std, "rho_up": rho_up,
    "J_pred": J_mean, "J_pred_std": J_std, "J_up": J_up,
    "M_pred": M_mean, "M_pred_std": M_std, "M_up": M_up
})
df.to_csv(OUT_CSV, index=False)
print(f"\n⇢ Logged results to {OUT_CSV}")

# -------------------------- plot ------------------------------
plt.figure(figsize=(9, 6))                                     # larger canvas
colors = ['C0', 'C1', 'C2', 'C3']
lw     = 2.0                                                   # line-width

# model (solid) ------------------------------------------------
plt.errorbar(mx_vals, flow_mean, yerr=flow_std, marker='o',
             color=colors[0], label=r'$f_e$', lw=lw)
plt.errorbar(mx_vals, rho_mean,  yerr=rho_std,  marker='s',
             color=colors[1], label=r'$\rho$', lw=lw)
plt.errorbar(mx_vals, J_mean,    yerr=J_std,    marker='^',
             color=colors[2], label=r'$J$', lw=lw)
plt.errorbar(mx_vals, M_mean,    yerr=M_std,    marker='d',
             color=colors[3], label=r'$M$', lw=lw)

# bicubic (dashed) --------------------------------------------
plt.plot(mx_vals, flow_up, linestyle='--', marker='o',
         color=colors[0], lw=lw, label='_nolegend_')
plt.plot(mx_vals, rho_up,  linestyle='--', marker='s',
         color=colors[1], lw=lw, label='_nolegend_')
plt.plot(mx_vals, J_up,    linestyle='--', marker='^',
         color=colors[2], lw=lw, label='_nolegend_')
plt.plot(mx_vals, M_up,    linestyle='--', marker='d',
         color=colors[3], lw=lw, label='_nolegend_')

# axis / title styling ----------------------------------------
plt.xlabel(r" $\mathbf{m_x, m_v}$",
           fontsize=14, fontweight='bold')
plt.ylabel(r"Relative $L^2$ error", fontsize=14, fontweight='bold')
plt.yscale("log")
plt.title(r"1D1V Vlasov: high-frequency 0 shot low-frequency test",
          fontsize=16, fontweight='bold', pad=12)

plt.xticks(fontsize=12);  plt.yticks(fontsize=12)
plt.grid(which="both", linestyle="--", alpha=0.5, linewidth=0.6)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
           fontsize=12, frameon=False)
plt.tight_layout()

plt.savefig(OUT_FIG, dpi=300)
plt.show()
print(f"⇢ Figure saved to {OUT_FIG}")
