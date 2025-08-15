#!/usr/bin/env python
# -------------------------------------------------------------
# Evaluation sweep over max_modes = 3 … 20 for the 2-D wave   #
# super-resolution problem                                    #
# -------------------------------------------------------------
import os, math, shutil, numpy as np
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch, torch.nn as nn
# =============================================================
# ---------- (1) DATA-GENERATION UTILITIES --------------------
# =============================================================
def sample_simple_ic(X, Y, max_modes=20):
    """Sum of up-to-max_modes low-frequency sine modes."""
    f0 = np.zeros_like(X, dtype=np.float32)
    M  = np.random.randint(1, max_modes + 1)
    for _ in range(M):
        kx  = np.random.randint(1, 4)
        ky  = np.random.randint(1, 4)
        A   = np.random.uniform(0.5, 1.0)
        phi = np.random.uniform(0, 2 * np.pi)
        f0 += A * np.sin(2 * np.pi * (kx * X + ky * Y) + phi)
    return f0, np.zeros_like(f0)

def generate_wave_dataset(
    N_samples      = 20,
    Lx             = 1.0, Ly      = 1.0,
    Nx             = 128, Ny      = 128,
    c              = 0.5,
    dt             = 0.01,
    t_final        = 1.0,
    downsample_fac = 4,
    max_modes      = 20,
    out_dir        = "./wave_dataset",
):
    """Generate fine / coarse trajectories normalised to [0,1]."""
    Nt = int(round(t_final / dt)) + 1
    r  = downsample_fac
    Nx_c, Ny_c = Nx // r, Ny // r
    dx_f, dy_f, dx_c, dy_c = Lx / Nx, Ly / Ny, Lx / Nx_c, Ly / Ny_c
    c2dt2 = (c * dt) ** 2

    x  = np.linspace(0, Lx, Nx, endpoint=False, dtype=np.float32)
    y  = np.linspace(0, Ly, Ny, endpoint=False, dtype=np.float32)
    Xf, Yf = np.meshgrid(x, y, indexing="ij")
    Xc, Yc = Xf[::r, ::r], Yf[::r, ::r]

    u_fine   = np.empty((N_samples, Nt, Nx,   Ny  ), np.float32)
    u_coarse = np.empty((N_samples, Nt, Nx_c, Ny_c), np.float32)

    for s in trange(N_samples, desc=f"max_modes={max_modes:2d}"):
        raw_f, raw_c = np.empty((Nt, Nx, Ny), np.float32), np.empty((Nt, Nx_c, Ny_c), np.float32)

        # initial conditions
        f0_f, g0_f = sample_simple_ic(Xf, Yf, max_modes=max_modes)
        f0_c       = f0_f[::r, ::r]

        # ------- leap-frog initial step (fine) -------
        u_nm1_f = f0_f.copy()
        lap0_f  = (np.roll(f0_f, +1, 0) + np.roll(f0_f, -1, 0) - 2 * f0_f) / dx_f ** 2 + \
                  (np.roll(f0_f, +1, 1) + np.roll(f0_f, -1, 1) - 2 * f0_f) / dy_f ** 2
        u_n_f   = u_nm1_f + 0.5 * c2dt2 * lap0_f

        # ------- coarse grid -------
        u_nm1_c = f0_c.copy()
        lap0_c  = (np.roll(f0_c, +1, 0) + np.roll(f0_c, -1, 0) - 2 * f0_c) / dx_c ** 2 + \
                  (np.roll(f0_c, +1, 1) + np.roll(f0_c, -1, 1) - 2 * f0_c) / dy_c ** 2
        u_n_c   = u_nm1_c + 0.5 * c2dt2 * lap0_c

        raw_f[0], raw_f[1] = u_nm1_f, u_n_f
        raw_c[0], raw_c[1] = u_nm1_c, u_n_c

        for n in range(1, Nt - 1):
            lap_f = (np.roll(u_n_f, +1, 0) + np.roll(u_n_f, -1, 0) - 2 * u_n_f) / dx_f ** 2 + \
                    (np.roll(u_n_f, +1, 1) + np.roll(u_n_f, -1, 1) - 2 * u_n_f) / dy_f ** 2
            u_np1_f = 2 * u_n_f - u_nm1_f + c2dt2 * lap_f

            lap_c = (np.roll(u_n_c, +1, 0) + np.roll(u_n_c, -1, 0) - 2 * u_n_c) / dx_c ** 2 + \
                    (np.roll(u_n_c, +1, 1) + np.roll(u_n_c, -1, 1) - 2 * u_n_c) / dy_c ** 2
            u_np1_c = 2 * u_n_c - u_nm1_c + c2dt2 * lap_c

            u_nm1_f, u_n_f = u_n_f, u_np1_f
            u_nm1_c, u_n_c = u_n_c, u_np1_c
            raw_f[n + 1], raw_c[n + 1] = u_n_f, u_n_c

        # per-trajectory normalisation
        u_min, u_max = raw_f.min(), raw_f.max()
        if u_max > u_min:
            raw_f = (raw_f - u_min) / (u_max - u_min)
            raw_c = (raw_c - u_min) / (u_max - u_min)
        else:
            raw_f.fill(0.0); raw_c.fill(0.0)

        u_fine[s], u_coarse[s] = raw_f, raw_c

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/u_fine_max_modes={max_modes}.npy",   u_fine)
    np.save(f"{out_dir}/u_coarse_max_modes={max_modes}.npy", u_coarse)

# ================================================================
# Coordinate → Fourier features
# ================================================================
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):                               # (B,H,W,2)
        proj = 2 * math.pi * torch.matmul(coords, self.B)    # (B,H,W,mapping_size)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)                        # (B,2*mapping_size,H,W)

# ---------------------------------------------------------------
def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid                                              # (B,H,W,2)

# ================================================================
# Fourier Neural Operator 2-D spectral layer
# ================================================================
class FourierLayer(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            / (in_ch * out_ch)
        )

    @staticmethod
    def compl_mul2d(inp, w):                                 # (B,IC,H,W) × (IC,OC,H,W)
        return torch.einsum('bixy,ioxy->boxy', inp, w)

    def forward(self, x):                                    # (B,C,H,W)  real
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.size(-1))                 # W_freq = W//2+1

        out_ft = torch.zeros(
            B, self.weight.size(1), H, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])

# ---------------------------------------------------------------
class ConvBlock(nn.Module):
    """[Conv → GELU] × 2 (keeps H×W)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

# ================================================================
# ↓↓↓ PixelShuffle-based up-sample block ↓↓↓
# ================================================================
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale ** 2), 3, padding=1)
        self.pix  = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.pix(self.conv(x)))

# ================================================================
# U-Net with Fourier bottleneck + PixelShuffle up-sampling
# NOTE: -- No pooling inside the bottleneck (only in encoder) --
# ================================================================
class SuperResUNet(nn.Module):
    def __init__(
        self,
        in_channels=101,
        lift_dim=128,
        mapping_size=64,
        mapping_scale=5.0,
        final_scale=2        # ← auto-detected from data
    ):
        super().__init__()

        # -------- lift ---------------
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        # -------- encoder ------------
        self.enc1 = ConvBlock(lift_dim,        lift_dim)        # keep  (Hc)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)    # pool → (Hc/2)
        self.pool = nn.MaxPool2d(2)

        # -------- bottleneck ---------
    
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
            nn.GELU()
        )

        # -------- decoder ------------
        # up1 keeps spatial dims (upscale=1) so it matches e2
        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)                    # cat(up1,e2)

        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)             # ×2  (Hc/2 → Hc)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)               # cat(up2,e1)

        self.dec0 = nn.Sequential(                                       # Hc → Hc×final_scale
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )

        # -------- output head --------
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)                      # 11-channel output
        )

    # -----------------------------------------------------------
    def forward(self, x):                                 # (B,11,Hc,Wc) normalised
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)   # lift
        x = self.lift(x)

        e1 = self.enc1(x)               # Hc
        e2 = self.enc2(self.pool(e1))   # Hc/2

        # ---- bottleneck ----
        b  = self.bottleneck(e2)        # Hc/2

        # ---- decoder ----
        d2 = self.up1(b)                             # Hc/2  (spatially matches e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up2(d2)                            # Hc
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.dec0(d1)                           # Hf
        return self.out_head(d0)                     # (B,11,Hf,Wf)  normalised

# =============================================================
# ---------- (3)  EVALUATION SWEEP ----------------------------
# =============================================================
BASE_DATA_DIR = Path("/pscratch/sd/h/hbassi/wave_dataset")
MODEL_PTH     = Path("/pscratch/sd/h/hbassi/models/2d_wave_FUnet_best_PS_FT_32to128_v1_low.pth")
STATS_PTH     = Path("./data/2d_wave_funet_phase1_stats_32to128_v1_low.pt")
DEVICE        = 'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- build / load model ----------
model = SuperResUNet(final_scale=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
model.eval()

stats       = torch.load(STATS_PTH, map_location=DEVICE)
data_mean   = stats["data_mean"].squeeze(0).to(DEVICE)
data_std    = stats["data_std" ].squeeze(0).to(DEVICE)

# ---------- arrays to hold summary MAE ----------
modes_list, mae_up_list, mae_pred_list = [], [], []

# ---------- generate + evaluate for max_modes = 3 … 20 -------
for max_modes in trange(3, 21):
    # ⓵ generate if not yet on disk
    out_dir = BASE_DATA_DIR / f"eval_max_modes_{max_modes}"
    if not (out_dir / f"u_fine_max_modes={max_modes}.npy").exists():
        generate_wave_dataset(max_modes=max_modes, out_dir=out_dir)

    # ⓶ load coarse / fine
    u_cg  = np.load(out_dir / f"u_coarse_max_modes={max_modes}.npy")[:, :101]  # (N,101,128,128)
    u_fg  = np.load(out_dir / f"u_fine_max_modes={max_modes}.npy")[:, :101]    # (N,101,256,256)
    N, Nt, Nc, _ = u_cg.shape
    scale = u_fg.shape[2] // Nc

    # ⓷ feed through network
    with torch.no_grad():
        x_in   = torch.tensor(u_cg, dtype=torch.float32, device=DEVICE)
        x_norm = (x_in - data_mean) / data_std
        y_pred = model(x_norm).cpu() * data_std + data_mean               # (N,101,256,256)

    # ⓸ MAE accumulation
    mae_up_all, mae_pred_all = 0.0, 0.0
    for case in range(N):
        for t in range(Nt):
            cg      = u_cg [case, t]
            cg_up   = zoom(cg, scale, order=3)
            fg_true = u_fg [case, t]
            fg_pred = y_pred[case, t].numpy()

            mae_up_all   += np.abs(cg_up  - fg_true).mean()
            mae_pred_all += np.abs(fg_pred - fg_true).mean()
    denom              = N * Nt
    mae_up, mae_pred   = mae_up_all / denom, mae_pred_all / denom
    modes_list.append(max_modes)
    mae_up_list.append(mae_up)
    mae_pred_list.append(mae_pred)
    print(f"[max_modes={max_modes:2d}]  MAE_up = {mae_up:.3e},  MAE_pred = {mae_pred:.3e}")

    # ⓹ diagnostic plots for sample #17 every Δt = 0.10
    case_idx      = 17
    time_stride   = 10
    fig_dir       = Path("./figures/wave/funet") / f"eval_max_modes_{max_modes}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for t in range(0, Nt, time_stride):
        cg      = u_cg [case_idx, t]
        cg_up   = zoom(cg, scale, order=1)
        fg_pred = y_pred[case_idx, t].numpy()
        fg_true = u_fg [case_idx, t]

        vmin, vmax = fg_true.min(), fg_true.max()
        err_up   = np.abs(cg_up  - fg_true)
        err_pred = np.abs(fg_pred - fg_true)
        mae_u, mae_p = err_up.mean(), err_pred.mean()
        err_max = max(err_up.max(), err_pred.max())

        fig, ax = plt.subplots(2, 4, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 1]})
        # fields row
        for a, im, ttl in zip(ax[0],
                              [cg, cg_up, fg_pred, fg_true],
                              ["coarse CG", "bicubic ↑", "prediction", "true FG"]):
            h = a.imshow(im, origin="lower", vmin=vmin, vmax=vmax)
            a.set_title(ttl, fontsize=9);  a.axis("off")
        fig.colorbar(h, ax=ax[0, :], orientation="vertical", fraction=0.02, pad=0.02,
                     shrink=0.8, label="u")

        # error row
        for a, im, ttl in zip([ax[1,1], ax[1,2]],
                              [err_up, err_pred],
                              [f"|CG↑ − FG|  MAE={mae_u:.2e}",
                               f"|pred − FG| MAE={mae_p:.2e}"]):
            he = a.imshow(im, origin="lower", cmap="jet", vmin=0, vmax=err_max)
            a.set_title(ttl, fontsize=9);  a.axis("off")
        ax[1,0].axis("off"); ax[1,3].axis("off")
        fig.colorbar(he, ax=[ax[1,1], ax[1,2]], orientation="vertical",
                     fraction=0.02, pad=0.02, shrink=0.8, label="|error|")
        fig.suptitle(f"max_modes = {max_modes}   |   sample #17   |   t = {t*0.01:.2f}",
                     y=0.97, fontsize=12)
        fig.savefig(fig_dir / f"ic17_t{t:03d}_maxModes{max_modes}.pdf",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

# =============================================================
# ---------- (4)  SUMMARY PLOT --------------------------------
# =============================================================
plt.figure(figsize=(6,4))
plt.plot(modes_list, mae_up_list,   "o-", label="bicubic ↑")
plt.plot(modes_list, mae_pred_list, "s-", label="prediction")
plt.xlabel("maximum number of sine modes in IC")
plt.ylabel("overall MAE")
plt.title("Error growth vs spectral richness of ICs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/wave/funet/mae_vs_maxModes.pdf", dpi=300)
plt.show()
