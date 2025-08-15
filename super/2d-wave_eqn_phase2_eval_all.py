────────────────────────────────────────────────────────
#  Phase-2 autoregressive roll-out with *sliding 100-frame window*
# ──────────────────────────────────────────────────────────────
import os, math, numpy as np, matplotlib.pyplot as plt, torch, torch.nn.functional as F, models
from tqdm import trange
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from numpy.fft import fftfreq, rfft2
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

# ---------- runtime / data paths ----------
PHASE2_CKPT = "/pscratch/sd/h/hbassi/models/wave_best_{model}_phase2.pth"
AR_CKPT     = "/pscratch/sd/h/hbassi/FNO_fine_tuning_2d-wave_32to128_high-freq_best_model.pth"
MODELS      = ["funet", "unet", "edsr", "fno"]     # SR nets
AR_NAME     = "fnoar"
UPSCALE     = 8
c, dt       = 0.5, 0.01
WINDOW      = 100                                  # sliding context
device      = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------- load fine + coarse test data ----------
fine_np   = np.load(
    "/pscratch/sd/h/hbassi/"
    "wave_dataset_multi_sf_modes=10_kmax=7/u_fine_test.npy"
)[0]                                               # (B,101,Hf,Wf)
coarse_np = np.load(
    "/pscratch/sd/h/hbassi/"
    "wave_dataset_multi_sf_modes=10_kmax=7/u_coarse_sf=8_test.npy"
)[0]                                               # (B,101,Hc,Wc)

fine_t   = torch.tensor(fine_np,   dtype=torch.float32, device=device).unsqueeze(0)
coarse_t = torch.tensor(coarse_np, dtype=torch.float32, device=device).unsqueeze(0)

B, Tp1, Hf, Wf = fine_t.shape
Hc             = Hf // UPSCALE
dx_c = dy_c    = 1.0 / Hc
c2dt2          = (c * dt) ** 2

# ---------- normalisation statistics ----------
stats = torch.load("./data/phase2_funet_stats_2d-wave_high_freq_sf=8.pt")
mean, std = stats["mean"].to(device), stats["std"].to(device)

#   1 ▸  coarse one-step wave update (single frame) -------------
def coarse_step_wave(prev, curr):
    lap = (
        (torch.roll(curr, +1, 0) + torch.roll(curr, -1, 0) - 2 * curr) / dx_c**2 +
        (torch.roll(curr, +1, 1) + torch.roll(curr, -1, 1) - 2 * curr) / dy_c**2
    )
    return 2 * curr - prev + c2dt2 * lap

# ---------- projection utility ----------
def project(fine):              # fine (Hf,Wf) → coarse (Hc,Wc)
    return fine[..., ::UPSCALE, ::UPSCALE]

# ---------- build & load Phase-2 SR nets ----------
def build(name, in_ch=WINDOW):
    if   name == "funet": return models.SuperResUNet(in_channels=in_ch, final_scale=UPSCALE)
    elif name == "unet":  return models.UNetSR(in_ch, upscale_factor=UPSCALE)
    elif name == "edsr":  return models.EDSR(in_ch, 128, 16, UPSCALE,
                                             mean=np.zeros(in_ch, np.float32),
                                             std =np.ones(in_ch, np.float32))
    elif name == "fno":   return models.FNO2dSR(in_ch, modes1=8, modes2=8,
                                               upscale_factor=UPSCALE)
    else: raise ValueError

nets = {}
for m in MODELS:
    net = build(m).to(device)
    net.load_state_dict(torch.load(PHASE2_CKPT.format(model=m), map_location="cpu"))
    net.eval(); nets[m] = net
    print(f"Loaded Phase-2 {m.upper()}.")
class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width):
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
# ---------- build & load AR-FNO ----------------------------------------
print("Loading autoregressive FNO ...")
ar_net = FNO2d(100, 1, 16, 16, 64).to(device)
ar_net.load_state_dict(torch.load(AR_CKPT, map_location=device))
ar_net.eval()
print(f"Loaded AR-FNO weights from {AR_CKPT}")

# ---------- allocate roll-out arrays ----------
roll = {m: torch.zeros_like(fine_t) for m in MODELS}
roll[AR_NAME] = torch.zeros_like(fine_t)           # for FNO-AR

# ─────────────────── autoregressive roll-out ──────────────────
for b in trange(B, desc="Trajectories"):
    fine_window   = fine_t[b, :WINDOW].clone()               # (100,Hf,Wf)
    coarse_window = fine_window[:, ::UPSCALE, ::UPSCALE]     # (100,Hc,Wc)

    # seed outputs with ground truth
    for m in MODELS:  roll[m][b, :WINDOW]   = fine_window
    roll[AR_NAME][b, :WINDOW] = fine_window

    # initialise AR-FNO window (1,100,Hf,Wf)
    window_u = fine_window.unsqueeze(0)

    for step in range(WINDOW, Tp1):
        # ---------- coarse predictor ----------------------------------
        prev_c, curr_c = coarse_window[-2], coarse_window[-1]
        next_c         = coarse_step_wave(prev_c, curr_c)
        coarse_window  = torch.cat([coarse_window[1:], next_c.unsqueeze(0)], 0)

        # ---------- AR-FNO on fine grid -------------------------------
        with torch.no_grad():
            outputs_u  = ar_net(window_u)             # (1,100,Hf,Wf)
            final_time = outputs_u.squeeze(0)[-1]     # (Hf,Wf)
        roll[AR_NAME][b, step] = final_time
        window_u = torch.cat([window_u[:, 1:], final_time.unsqueeze(0).unsqueeze(0)], dim=1)

        # ---------- super-resolution nets -----------------------------
        coarse_net_in = coarse_window.unsqueeze(0)
        norm_in       = (coarse_net_in - mean) / std
        for m, net in nets.items():
            with torch.no_grad():
                out        = net(norm_in.float()) * std + mean   # (1,100,Hf,Wf)
            next_fine = out[:, -1]
            roll[m][b, step] = next_fine.squeeze()

        # ---------- slide fine_window (for completeness) -------------
        fine_window = torch.cat([fine_window[1:], final_time.unsqueeze(0)], 0)

# ---------- bicubic baseline -----------------------------------------
with torch.no_grad():
    ups_t = F.interpolate(coarse_t, scale_factor=UPSCALE,
                          mode="bicubic", align_corners=False)

# ---------- numpy conversion -----------------------------------------
cutoff   = 501                        # keep full length
fine_np  = fine_t.cpu().numpy()[:, :cutoff]
pred_np  = {m: roll[m].cpu().numpy()[:, :cutoff] for m in MODELS}
pred_np["fnoar"] = roll[AR_NAME].cpu().numpy()[:, :cutoff]
ups_np   = ups_t.cpu().numpy()[:, :cutoff]

# ────────────────── metric helpers (unchanged) ─────────────────
def spectral_mse_2d(a, b):
    return np.mean((np.abs(rfft2(a)) - np.abs(rfft2(b))) ** 2)

def ms_ssim_2d(a, b):
    return ssim(a, b, multiscale=True, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False,
                data_range=b.max() - b.min())

def psnr(pred, ref):
    mse = np.mean((pred - ref) ** 2)
    return np.inf if mse == 0 else 10.0 * math.log10(((ref.max() - ref.min()) ** 2) / mse)

def residual_mse(prev, curr, nxt):
    lap = (
        (np.roll(curr, +1, 0) + np.roll(curr, -1, 0) - 2 * curr) / dx_c**2 +
        (np.roll(curr, +1, 1) + np.roll(curr, -1, 1) - 2 * curr) / dy_c**2
    )
    r = (nxt - 2 * curr + prev) / dt**2 - c**2 * lap
    return np.mean(r**2)

# ────────────────── per-snapshot figures ──────────────────────
fig_dir = "./figures/wave_phase2_rollout_test123"
os.makedirs(fig_dir, exist_ok=True)
plot_steps = range(0, cutoff, 100)

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
            err = np.linalg.norm(img - fine_np[0, t]) / np.linalg.norm(fine_np[0, t])
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

# ─────────────────── global metric averages ───────────────────
METRICS = ["L2","SSIM","MSS","SPEC","CORR","PSNR","PHY"]
tags    = ["FUNET","FNO","FNOAR","UNET","EDSR","UPS"]

metrics = {m: {k:0. for k in METRICS} for m in tags}
sq      = {m: {k:0. for k in METRICS} for m in tags}

total = fine_np.shape[0] * fine_np.shape[1] - 2 * fine_np.shape[0]  # skip edges

for b in range(fine_np.shape[0]):
    for t in range(1, fine_np.shape[1]-1):
        ref_prev, ref_curr, ref_next = fine_np[b,t-1], fine_np[b,t], fine_np[b,t+1]
        for tag, arr in [("UPS",ups_np), ("FUNET",pred_np["funet"]),
                         ("FNO",pred_np["fno"]), ("FNOAR",pred_np["fnoar"]),
                         ("UNET",pred_np["unet"]), ("EDSR",pred_np["edsr"])]:
            pred_prev, pred_curr, pred_next = arr[b,t-1], arr[b,t], arr[b,t+1]
            vals = dict(
                L2   = np.linalg.norm(pred_curr-ref_curr) / (np.linalg.norm(ref_curr)+1e-8),
                SSIM = ssim(pred_curr, ref_curr, data_range=ref_curr.ptp()),
                MSS  = ms_ssim_2d(pred_curr, ref_curr),
                SPEC = spectral_mse_2d(pred_curr, ref_curr),
                CORR = pearsonr(ref_curr.ravel(), pred_curr.ravel())[0],
                PSNR = psnr(pred_curr, ref_curr),
                PHY  = residual_mse(pred_prev, pred_curr, pred_next)
            )
            for k,v in vals.items():
                metrics[tag][k] += v
                sq[tag][k]      += v**2

# ─────────────────────── print mean ± std ─────────────────────
print("\n─── Dataset-wide mean ± std ─────────────────────────────")
for tag in tags:
    out = []
    for k in METRICS:
        mu  = metrics[tag][k] / total
        var = max(sq[tag][k] / total - mu**2, 0.0)
        sd  = math.sqrt(var)
        if k=="PSNR":
            out.append(f"{mu:5.2f}±{sd:4.2f}dB")
        elif k in ("L2","SPEC","PHY"):
            out.append(f"{mu:.2e}±{sd:.1e}")
        else:
            out.append(f"{mu:.4f}±{sd:.4f}")
    print(f"{tag:6s} | " + " | ".join(out))
