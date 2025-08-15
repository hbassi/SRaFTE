# ──────────────────────────────────────────────────────────────
#  Phase-2 autoregressive roll-out with *sliding 100-frame window*
# ──────────────────────────────────────────────────────────────
import os, math, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import trange
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from numpy.fft import rfft2, fft2, ifft2
from sklearn.decomposition import PCA

import models                         # FUnet, UNetSR, EDSR, FNO2dSR

torch.set_float32_matmul_precision("high")

# ---------- runtime / data paths ----------
PHASE2_CKPT = "/pscratch/sd/h/hbassi/models/heat_best_{model}_phase2.pth"
AR_CKPT     = "/pscratch/sd/h/hbassi/FNO_fine_tuning_2d-heat_32to128_best_model.pth"
MODELS      = ["funet", "unet", "edsr", "fno"]               # SR nets
AR_NAME     = "fnoar"

UPSCALE     = 4
Nc          = 32
nu, dt      = 0.1, 0.01
alpha       = 0.01                                            # forcing amp
WINDOW      = 100                                             # sliding context
device      = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------- load fine + coarse test data ----------
fine_np   = np.load(
    "/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k_test2A.npy"
)[-1:]                                                       # (B,101,Hf,Wf)
coarse_np = np.load(
    "/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_coarse_all_smooth_gauss_1k_test2A.npy"
)[-1:]                                                       # (B,101,Hc,Wc)

fine_t   = torch.tensor(fine_np,   dtype=torch.float32, device=device)
coarse_t = torch.tensor(coarse_np, dtype=torch.float32, device=device)
mean_c = coarse_t[:, :WINDOW].mean(dim=(0,2,3), keepdim=True)   # shape (1,WINDOW,1,1)
std_c  = coarse_t[:, :WINDOW].std (dim=(0,2,3), keepdim=True).clamp_min(1e-8)
B, Tp1, Hf, Wf = fine_t.shape
dx = dy        = 1.0 / Hf
kx = torch.fft.fftfreq(Nc, d=1.0 / Nc) * (2.0 * math.pi)
ky = kx.clone()
KX, KY = torch.meshgrid(kx, ky, indexing='ij')
K2 = (KX**2 + KY**2).to(device)
K2[0, 0] = 1e-14
exp_fac = torch.exp(-nu * K2 * dt)                           # (Nc,Nc)

xs = (torch.arange(Nc, device=device, dtype=torch.float64) + 0.5) / Nc
ys = xs.clone()
Xc, Yc = torch.meshgrid(xs, ys, indexing='ij')
f_xy  = torch.sin(2 * math.pi * Xc) * torch.sin(2 * math.pi * Yc)
f_hat = torch.fft.fft2(f_xy)
forcing_term = alpha * (1.0 - exp_fac) * f_hat / (nu * K2)   # (Nc,Nc)

# ---------- normalisation statistics ----------
stats = torch.load(f"./data/2d_heat_phase2_funet_stats_nu={nu}.pt")
mean, std = stats["data_mean"].to(device), stats["data_mean"].to(device)

def coarse_time_step_heat(window: torch.Tensor) -> torch.Tensor:
    u_hat = torch.fft.fft2(window.to(torch.float64))             # (B,W,Nc,Nc)
    u_hat = exp_fac * u_hat + forcing_term                       # broadcasts
    return torch.real(torch.fft.ifft2(u_hat)).to(torch.float32)  # same shape

# ---------- projection utility ----------
def project(fine):              # fine (Hf,Wf) → coarse (Hc,Wc)
    return fine[..., ::UPSCALE, ::UPSCALE]

# ---------- build Phase-2 SR nets -----------------------------
def build(name, in_ch=WINDOW):
    if   name == "funet": return models.SuperResUNet(in_channels=in_ch, final_scale=UPSCALE)
    elif name == "unet":  return models.UNetSR(in_ch, upscale_factor=UPSCALE)
    elif name == "edsr":  return models.EDSR(in_ch, 128, 16, UPSCALE,
                                             mean=np.zeros(in_ch, np.float32),
                                             std =np.ones(in_ch, np.float32))
    elif name == "fno":   return models.FNO2dSR(in_ch, modes1=16, modes2=16,
                                               upscale_factor=UPSCALE)
    else: raise ValueError

nets = {}
for m in MODELS:
    net = build(m).to(device)
    if m == 'unet' or m == 'fno' or m == 'edsr':
        net.load_state_dict(torch.load(PHASE2_CKPT.format(model=m), map_location="cpu")['model'])
    else:
        net.load_state_dict(torch.load(PHASE2_CKPT.format(model=m), map_location="cpu"))
    net.eval(); nets[m] = net
    print(f"Loaded Phase-2 {m.upper()}.")

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width):
        """
        in_channels: number of time steps (features) per spatial location
        out_channels: number of output channels 
        modes1, modes2: number of Fourier modes to keep
        """
        super(FNO2d, self).__init__()
        self.width = width
        # Lift the input (here, in_channels = T) to a higher-dimensional feature space.
        self.fc0 = nn.Linear(in_channels, self.width)

        # Fourier layers and pointwise convolutions 
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)

        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)

        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        """
        x: input of shape [B, T, H, W]
        """
        # Permute to [B, H, W, T] so each spatial location has a feature vector of length T
        x = x.permute(0, 2, 3, 1)
        # Lift to higher-dimensional space
        x = self.fc0(x)
        # Permute to [B, width, H, W] for convolutional operations
        x = x.permute(0, 3, 1, 2)

        # Apply Fourier layers with local convolution
        x = self.conv0(x) + self.w0(x)
        x = nn.GELU()(x)
        x = self.conv1(x) + self.w1(x)
        x = nn.GELU()(x)
        x = self.conv2(x) + self.w2(x)
        x = nn.GELU()(x)
        x = self.conv3(x) + self.w3(x)

        # Permute back and project to output space
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

# Spectral convolution layer remains unchanged
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy, ioxy -> boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights
        )
        x = torch.fft.irfft2(out_ft, s=x.shape[-2:])
        return x

print("Loading autoregressive FNO ...")
ar_net = FNO2d(WINDOW, 1, 16, 16, 64).to(device)
ar_net.load_state_dict(torch.load(AR_CKPT, map_location=device))
ar_net.eval(); print("AR-FNO ready.")

# ---------- allocate roll-out arrays ----------
roll = {m: torch.zeros_like(fine_t) for m in MODELS}
roll[AR_NAME] = torch.zeros_like(fine_t)           # AR-baseline
Tp1 = 106
# ---------- autoregressive sliding-window loop -------------
for b in trange(B, desc="Trajectories"):
    # ---- seed 100-frame windows ----------------------------
    fine_window   = fine_t[b, :WINDOW].clone()                       # (W,Hf,Wf)
    coarse_window = torch.stack([project(fr) for fr in fine_window]) # (W,Nc,Nc)

    # ---- seed output tensors -------------------------------
    for m in MODELS: roll[m][b, :WINDOW] = fine_window
    roll[AR_NAME][b, :WINDOW] = fine_window

    # ---- AR-FNO context ------------------------------------
    window_u = fine_window.unsqueeze(0)           # (1,W,Hf,Wf)

    for step in range(WINDOW, Tp1):
        # ====================================================
        # 1 ▸ coarse-grid physics update on the current window
        # ====================================================
        coarse_input = coarse_window.unsqueeze(0)             # (1,W,Nc,Nc)
        updated_coarse = coarse_time_step_heat(coarse_input)  # (1,W,Nc,Nc)
        # keep all W frames; newest = last channel
        coarse_window = updated_coarse.squeeze(0)

        # ====================================================
        # 2 ▸ AR-FNO on fine grid
        # ====================================================
        with torch.no_grad():
            outputs_u = ar_net(window_u)           # (1,W,Hf,Wf)
            fine_ar   = outputs_u[:, -1]           # newest fine frame
        roll[AR_NAME][b, step] = fine_ar.squeeze()
        window_u = torch.cat([window_u[:, 1:], fine_ar.unsqueeze(0)], dim=1)

        # ====================================================
        # 3 ▸ Super-resolution networks
        # ====================================================
        norm_in = (coarse_window.unsqueeze(0) - mean_c) / std_c  # (1,W,Nc,Nc)
        for m, net in nets.items():
            with torch.no_grad():
                out = net(norm_in.float()) * std_c + mean_c      # (1,W,Hf,Wf)
            next_fine = out[:, -1]                               # newest SR frame
            roll[m][b, step] = next_fine.squeeze()

        # ====================================================
        # 4 ▸ feedback: project SR output → coarse ➜ slide windows
        # ====================================================
        new_coarse = project(next_fine.squeeze())                # (Nc,Nc)
        coarse_window = torch.cat([coarse_window[1:],            # drop oldest
                                   new_coarse.unsqueeze(0)], 0)  # append newest
        fine_window   = torch.cat([fine_window[1:],              # drop oldest
                                   next_fine.squeeze().unsqueeze(0)], 0)

            

# ---------- bicubic baseline ---------------------------------
with torch.no_grad():
    ups_t = F.interpolate(coarse_t, scale_factor=UPSCALE,
                          mode="bicubic", align_corners=False)

# ---------- numpy conversion ---------------------------------
fine_np  = fine_t.cpu().numpy()
pred_np  = {m: roll[m].cpu().numpy() for m in MODELS}
pred_np["fnoar"] = roll[AR_NAME].cpu().numpy()
ups_np   = ups_t.cpu().numpy()

# ────────────────── metric helpers (unchanged) ───────────────
def spectral_mse_2d(a, b):
    return np.mean((np.abs(rfft2(a)) - np.abs(rfft2(b))) ** 2)
def ms_ssim_2d(a, b):
    return ssim(a, b, multiscale=True, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False,
                data_range=b.max() - b.min())
def psnr(pred, ref):
    mse = np.mean((pred - ref) ** 2)
    return np.inf if mse == 0 else 10.0 * math.log10(((ref.max()-ref.min())**2) / mse)
# def residual_mse(curr, nxt):
#     """Heat-eq residual  (scalar)"""
#     u_hat = fft2(curr)
#     pred  = np.real(ifft2(exp_fac.cpu().numpy()*u_hat + forcing_term.cpu().numpy()))
#     r     = nxt - pred
#     return np.mean(r**2)

# ─────────────── per-snapshot visualisations (optional) ───────
fig_dir = "./figures/heat_phase2_rollout_test123"
os.makedirs(fig_dir, exist_ok=True)
plot_steps = range(0, Tp1, 5)

COLS = ["GT", "UPS", "FNOAR", "FUNET", "EDSR", "FNO", "UNET"]
label_map = {"GT":"Ground truth", "FUNET":"FUnet", "FNO":"FNO-SR",
             "FNOAR":"FNO-AR", "UNET":"U-Net", "EDSR":"EDSR", "UPS":"Upsampled"}

for t in plot_steps:
    vmin, vmax = fine_np[0, t].min(), fine_np[0, t].max()
    fig, axes  = plt.subplots(1, len(COLS), figsize=(4 * len(COLS), 4))
    for ax, col in zip(axes, COLS):
        if   col == "GT":    img = fine_np[0, t]
        elif col == "UPS":   img = ups_np [0, t]
        elif col == "FNOAR": img = pred_np["fnoar"][0, t]
        else:                img = pred_np[col.lower()][0, t]

        ax.imshow(img, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(label_map[col], fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        if col != "GT":
            err = np.linalg.norm(img - fine_np[0, t]) / (np.linalg.norm(fine_np[0, t])+1e-8)
            ax.text(0.02, 0.94, rf"$\bf L_2={err:.2e}$",
                    transform=ax.transAxes, fontsize=9,
                    color="white", ha="left", va="top",
                    bbox=dict(fc="black", alpha=0.6, pad=0.25))

    # fig.colorbar(axes[0].images[0], ax=axes, fraction=0.02, pad=0.03)\
    #    .set_label("$u$", fontsize=12)
    fig.savefig(os.path.join(fig_dir, f"phase2_t{t:04d}.pdf"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"Saved snapshot figures to {fig_dir}")

tags     = ["UPS", "FUNET", "FNO", "FNOAR", "UNET", "EDSR"]
METRICS  = ["L2", "SSIM", "MSS", "SPEC", "CORR"]   # whichever ones you need

metrics  = {tag: {k: 0.0 for k in METRICS} for tag in tags}
sq       = {tag: {k: 0.0 for k in METRICS} for tag in tags}
count    = {tag: 0 for tag in tags}                # how many frames actually used

B        = fine_np.shape[0]
T        = WINDOW + Tp1 - 1                        # because we access t and t+1

for b in range(B):
    for t in range(T):
        ref_curr, ref_next = fine_np[b, t], fine_np[b, t + 1]

        for tag, arr in [
            ("UPS",   ups_np),
            ("FUNET", pred_np["funet"]),
            ("FNO",   pred_np["fno"]),
            ("FNOAR", pred_np["fnoar"]),
            ("UNET",  pred_np["unet"]),
            ("EDSR",  pred_np["edsr"]),
        ]:
            pred_curr, pred_next = arr[b, t], arr[b, t + 1]

            # ── skip frames that are identically zero ─────────────────
            # (except for UPS, which we *always* evaluate)
            if tag != "UPS" and np.linalg.norm(pred_curr) < 1e-12:
                continue

            vals = dict(
                L2   = np.linalg.norm(pred_curr - ref_curr)
                       / (np.linalg.norm(ref_curr) + 1e-8),
                SSIM = ssim(pred_curr, ref_curr, data_range=ref_curr.ptp()),
                MSS  = ms_ssim_2d(pred_curr, ref_curr),
                SPEC = spectral_mse_2d(pred_curr, ref_curr),
                CORR = pearsonr(ref_curr.ravel(), pred_curr.ravel())[0],
                # PSNR = psnr(pred_curr, ref_curr),
                # PHY  = residual_mse(pred_curr, pred_next),
            )

            for k, v in vals.items():
                metrics[tag][k] += v
                sq[tag][k]      += v ** 2
            count[tag] += 1

# ─────────────────────── print mean ± std ─────────────────────
print("\n─── Heat-equation dataset: mean ± std over *valid* snapshots ──")
for tag in tags:
    if count[tag] == 0:          # just in case an entire model was skipped
        print(f"{tag:6s} |  no valid frames")
        continue

    out = []
    for k in METRICS:
        mu = metrics[tag][k] / count[tag]
        sd = math.sqrt(max(sq[tag][k] / count[tag] - mu ** 2, 0.0))

        if k == "PSNR":
            out.append(f"{mu:5.2f}±{sd:4.2f}dB")
        elif k in ("L2", "SPEC", "PHY"):
            out.append(f"{mu:.2e}±{sd:.1e}")
        else:
            out.append(f"{mu:.4f}±{sd:.4f}")
    print(f"{tag:6s} | " + " | ".join(out) + f"   (N={count[tag]})")