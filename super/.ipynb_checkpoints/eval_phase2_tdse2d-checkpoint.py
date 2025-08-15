#!/usr/bin/env python
# ===============================================================
#  eval_phase2_tdse2d.py  (Phase-1 single-shot ➜ Phase-2 roll-out)
#  ---------------------------------------------------------------
#  GT | Pred | Bicubic | |Pred-err| |Bicubic-err| + total errors
# ===============================================================
import os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from skimage.metrics import structural_similarity as ssim
from numpy.fft import fftfreq, fft2, ifft2, rfft2
from sklearn.decomposition import PCA
from matplotlib import gridspec
import imageio.v2 as imageio

# # ---------------------------------------------------------------- Paths / settings
# PATH_FINE_LOCAL      = './tdse2d_traj0_256_64to256.npy'   # fine-grid 256×256 (complex64)
# PATH_COARSE_LOCAL    = './tdse2d_traj0_64_64to256.npy'    # true coarse-grid 64×64 (complex64)

# PATH_MODEL_PHASE2    = '/pscratch/sd/h/hbassi/models/tdse_phase2_best_model_t=75_64to256.pth'
# PATH_MODEL_PHASE1    = '/pscratch/sd/h/hbassi/models/2d_tdse_FUnet_best_PS_FT_64to256_500_t=75.pth'
# PATH_STATS_PHASE2    = './data/tdse2d_phase2_norm_amp_t=75_64to256.pt'
# PATH_STATS_PHASE1    = './data/2d_tdse_funet_phase1_stats_64to256_v1.pt'

# ---------------------------------------------------------------- Paths / settings
PATH_FINE_LOCAL      = './tdse2d_traj0_256.npy'   # fine-grid 256×256 (complex64)
PATH_COARSE_LOCAL    = './tdse2d_traj0_128.npy'    # true coarse-grid 64×64 (complex64)

PATH_MODEL_PHASE2    = '/pscratch/sd/h/hbassi/models/tdse_phase2_best_model_t=75.pth'
PATH_MODEL_PHASE1    = '/pscratch/sd/h/hbassi/models/2d_tdse_FUnet_best_PS_FT_128to256_500_t=75.pth'
PATH_STATS_PHASE2    = './data/tdse2d_phase2_norm_amp_t=75.pt'
PATH_STATS_PHASE1    = './data/2d_tdse_funet_phase1_stats_128to256_v1.pt'

DEVICE           = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
WIN              = 75          # temporal window for the Phase-2 model
NUM_EXTRA        = 5000        # steps *beyond* recorded data
SAVE_PRED_FILE   = './tdse_phase2_preds_traj0_128to256.npy'
FIG_OUT_DIR      = './figs_tdse_phase2_eval'
os.makedirs(FIG_OUT_DIR, exist_ok=True)

# ===============================================================
#  Torch Strang-split steppers (Dirichlet BCs, 4-term K series)
# ===============================================================
class TDSECoarseStepper:
    def __init__(self, N=128, dom_a=-80.0, dom_b=40.0,
                 dt_phys=0.01 * 2.4e-3 * 1e-15,
                 to_atomic=2.4188843265857e-17):
        self.N  = N
        self.dt = torch.tensor(dt_phys / to_atomic, dtype=torch.float32)
        self.dx = torch.tensor((dom_b - dom_a) / (N - 1), dtype=torch.float32)
        xs = torch.linspace(dom_a, dom_b, N, dtype=torch.float32)
        xmat, ymat = torch.meshgrid(xs, xs, indexing='ij')
        pot = ( -((xmat + 10.0)**2 + 1.0)**-0.5
                -((ymat + 10.0)**2 + 1.0)**-0.5
                +((xmat - ymat)**2 + 1.0)**-0.5 )
        self.v_phase = torch.exp(-1j * self.dt * pot).to(torch.complex64)
        self.a_coeff = (1j * self.dt / 2).to(torch.complex64)
        self._dev_cached = None

    @staticmethod
    def _shift_and_zero(t, shift, dim):
        out = torch.roll(t, shifts=shift, dims=dim)
        if shift > 0:
            sl = [slice(None)] * t.ndim; sl[dim] = slice(0, shift)
            out[tuple(sl)] = 0
        elif shift < 0:
            sl = [slice(None)] * t.ndim; sl[dim] = slice(shift, None)
            out[tuple(sl)] = 0
        return out

    def _laplacian_4th(self, psi):
        dx2 = self.dx * self.dx
        p2 = self._shift_and_zero(psi,  2, 3)
        p1 = self._shift_and_zero(psi,  1, 3)
        m1 = self._shift_and_zero(psi, -1, 3)
        m2 = self._shift_and_zero(psi, -2, 3)
        lap_x = (-p2 + 16*p1 - 30*psi + 16*m1 - m2) / (12*dx2)

        p2 = self._shift_and_zero(psi,  2, 2)
        p1 = self._shift_and_zero(psi,  1, 2)
        m1 = self._shift_and_zero(psi, -1, 2)
        m2 = self._shift_and_zero(psi, -2, 2)
        lap_y = (-p2 + 16*p1 - 30*psi + 16*m1 - m2) / (12*dx2)

        return lap_x + lap_y

    def _apply_K(self, psi):
        a_psi = self.a_coeff * self._laplacian_4th(psi)
        out, a_pow = psi + a_psi, a_psi
        for div in (2.0, 6.0, 24.0):
            a_pow = self.a_coeff * self._laplacian_4th(a_pow)
            out   = out + a_pow / div
        return out

    @torch.no_grad()
    def step(self, psi):
        if self._dev_cached != psi.device:
            self.v_phase = self.v_phase.to(psi.device)
            self.a_coeff = self.a_coeff.to(psi.device)
            self.dx      = self.dx.to(psi.device)
            self._dev_cached = psi.device
        psi_out = self._apply_K(psi) * self.v_phase
        return self._apply_K(psi_out)

class TDSEFineStepper(TDSECoarseStepper):
    def __init__(self): super().__init__(N=256)

# ===============================================================
#  Network components  (unchanged definitions)
# ===============================================================
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    def forward(self, coords):
        proj = 2 * math.pi * torch.matmul(coords, self.B)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)

def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)

class FourierLayer(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat) /
            (in_ch * out_ch)
        )
    @staticmethod
    def compl_mul2d(inp, w): return torch.einsum('bixy,ioxy->boxy', inp, w)
    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1, m2 = min(self.modes1, H), min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(
            B, self.weight.size(1), H, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2]
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
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale**2), 3, padding=1)
        self.pix  = nn.PixelShuffle(upscale); self.act = nn.GELU()
    def forward(self, x): return self.act(self.pix(self.conv(x)))

class SuperResUNet(nn.Module):
    def __init__(self, in_channels=1, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0, final_scale=2):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, 1)
        self.enc1 = ConvBlock(lift_dim, lift_dim)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(lift_dim, lift_dim * 2)
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, 32, 32), nn.GELU()
        )
        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, 1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)
        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)
        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )
    def forward(self, x):
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)
        x  = self.lift(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([self.up1(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))
        d0 = self.dec0(d1)
        return self.out_head(d0)

# ===============================================================
#  Load stats & models
# ===============================================================
stats_p2 = torch.load(PATH_STATS_PHASE2, map_location=DEVICE)
mean_p2, std_p2 = stats_p2['mean'].to(DEVICE), stats_p2['std'].to(DEVICE)

stats_p1 = torch.load(PATH_STATS_PHASE1, map_location=DEVICE)
mean_p1 = stats_p1['data_mean'].to(DEVICE).squeeze(0)
std_p1  = stats_p1['data_std' ].to(DEVICE).squeeze(0)

model_p2 = SuperResUNet(in_channels=WIN, final_scale=2).to(DEVICE)
model_p2.load_state_dict(torch.load(PATH_MODEL_PHASE2, map_location=DEVICE))
model_p2.eval()

model_p1 = SuperResUNet(in_channels=WIN, final_scale=2).to(DEVICE)
model_p1.load_state_dict(torch.load(PATH_MODEL_PHASE1, map_location=DEVICE))
model_p1.eval()

# ===============================================================
#  Load fine & coarse trajectories
# ===============================================================
fine_all   = torch.tensor(np.load(PATH_FINE_LOCAL),
                          dtype=torch.complex64, device=DEVICE)        # (T,256,256)
coarse_all = torch.tensor(np.load(PATH_COARSE_LOCAL),
                          dtype=torch.complex64, device=DEVICE)        # (T,64,64)

T_total = fine_all.size(0); assert T_total > WIN

# ---------- windows for the NN path (identical to before) ------
fine_window            = fine_all[:WIN]            # (75,256,256) complex
coarse_window_complex  = fine_all[:WIN, ::2, ::2]  # (75,64,64)  complex  (drives NN)
coarse_window_amp      = coarse_window_complex.abs().float()

# ---------- NEW: ground-truth coarse windows -------------------
true_coarse_window_complex = coarse_all[:WIN]          # (75,64,64) complex
# (amplitude only needed for plotting & bicubic)
true_coarse_window_amp     = true_coarse_window_complex.abs().float()

# ---------- steppers ------------------------------------------
stepper_coarse = TDSECoarseStepper()   # 64×64
stepper_fine   = TDSEFineStepper()     # 256×256

total_rollout = (T_total - WIN) + NUM_EXTRA

# ---------------------------------------------------------------
def bicubic_up(c_amp):
    return F.interpolate(
        c_amp.unsqueeze(0).unsqueeze(0), scale_factor=2,
        mode='bicubic', align_corners=False
    ).squeeze(0).squeeze(0)

# ===============================================================
#  ----------------  P H A S E 1  -------------------------------
# ===============================================================
with torch.no_grad():
    norm_in_p1 = (coarse_window_amp.unsqueeze(0) - mean_p1) / std_p1
    phase1_pred_sequence = (model_p1(norm_in_p1) * std_p1 + mean_p1)  # (1,75,256,256)

phase1_pred_sequence    = phase1_pred_sequence.squeeze(0).cpu()          # (75,256,256)
phase1_bicubic_sequence = torch.stack([bicubic_up(true_coarse_window_amp[i]).cpu()
                                       for i in range(WIN)])              # (75,256,256)
true_early_sequence     = fine_all[:WIN].abs().float().cpu()              # (75,256,256)

# ===============================================================
#  ----------------  P H A S E 2  -------------------------------
# ===============================================================
phase2_pred_frames    = []
phase2_true_frames    = []
phase2_bicubic_frames = []
coarse_amp_frames     = []        # will store TRUE coarse amplitudes for plots

for step_idx in trange(total_rollout, desc='Phase-2 roll'):

    # -----------------------------------------------------------
    # 1) Propagate the NN’s *down-sampled-fine* coarse window
    #    (this drives the Phase-2 network; unchanged)
    # -----------------------------------------------------------
    next_coarse_complex = stepper_coarse.step(coarse_window_complex.unsqueeze(0))
    next_coarse_complex = next_coarse_complex.squeeze(0)[-1]          # (64,64) complex
    next_coarse_amp     = next_coarse_complex.abs().float()

    # slide NN coarse histories
    coarse_window_complex = torch.cat(
        [coarse_window_complex[1:], next_coarse_complex.unsqueeze(0)], dim=0)
    coarse_window_amp = torch.cat(
        [coarse_window_amp[1:], next_coarse_amp.unsqueeze(0)], dim=0)

    # -----------------------------------------------------------
    # 2) Propagate the *true* coarse-grid window (NEW baseline)
    # -----------------------------------------------------------
    next_true_coarse_complex = stepper_coarse.step(
        true_coarse_window_complex.unsqueeze(0)
    ).squeeze(0)[-1]                                                  # (64,64) complex
    next_true_coarse_amp = next_true_coarse_complex.abs().float()

    # store for later gallery plotting
    coarse_amp_frames.append(next_true_coarse_amp.cpu())

    # slide true coarse history window
    true_coarse_window_complex = torch.cat(
        [true_coarse_window_complex[1:], next_true_coarse_complex.unsqueeze(0)], dim=0)

    # -----------------------------------------------------------
    # 3) Neural lift ➜ predict fine amplitude (unchanged)
    # -----------------------------------------------------------
    with torch.no_grad():
        norm_in_p2 = (coarse_window_amp.unsqueeze(0) - mean_p2) / std_p2
        out_amp = model_p2(norm_in_p2) * std_p2 + mean_p2
    phase2_pred_next = out_amp[:, -1].squeeze(0)
    phase2_pred_frames.append(phase2_pred_next.cpu())

    # -----------------------------------------------------------
    # 4) Bicubic baseline from *true* coarse dynamics (NEW)
    # -----------------------------------------------------------
    phase2_bicubic_frames.append(bicubic_up(next_true_coarse_amp).cpu())

    # -----------------------------------------------------------
    # 5) Fine-grid ground-truth propagation (unchanged)
    # -----------------------------------------------------------
    last_fine_complex = fine_window[-1:].unsqueeze(0)                 # (1,1,256,256)
    true_next_complex = stepper_fine.step(last_fine_complex).squeeze(0).squeeze(0)
    phase2_true_frames.append(true_next_complex.abs().float().cpu())

    # slide fine history
    fine_window = torch.cat([fine_window[1:], true_next_complex.unsqueeze(0)], dim=0)

# stack to tensors ---------------------------------------------------------
phase2_pred_sequence    = torch.stack(phase2_pred_frames)
phase2_bicubic_sequence = torch.stack(phase2_bicubic_frames)
phase2_true_sequence    = torch.stack(phase2_true_frames)

# ===============================================================
#  Combine sequences (Phase-1 + Phase-2)
# ===============================================================
pred_sequence    = torch.cat([phase1_pred_sequence,    phase2_pred_sequence],    0)
bicubic_sequence = torch.cat([phase1_bicubic_sequence, phase2_bicubic_sequence], 0)
true_sequence    = torch.cat([true_early_sequence,     phase2_true_sequence],    0)

# ===============================================================
#  Error metrics
# ===============================================================
eps = 1e-12
def spectral_mse_2d(a, b):
    return np.mean((np.abs(rfft2(a)) - np.abs(rfft2(b))) ** 2)

def ms_ssim_2d(a, b):
    return ssim(a, b, multiscale=True, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False,
                data_range=b.max() - b.min())

def psnr(pred, ref):
    mse = np.mean((pred - ref) ** 2)
    if mse == 0: return np.inf
    peak = ref.max() - ref.min()
    return 10.0 * math.log10(peak * peak / mse)

# --- probability densities ------------------------------------
prob_pred_seq = phase2_pred_sequence.numpy()**2                 # (T₂,256,256)
prob_true_seq = phase2_true_sequence.numpy()**2

# --- marginal density ρᵧ(y) (= Σₓ ρ) --------------------------
rho_y_pred_seq = np.sum(prob_pred_seq, axis=2)                  # (T₂,256)
rho_y_true_seq = np.sum(prob_true_seq, axis=2)

# --- dipole moments -------------------------------------------
La, Lb = -80.0, 40.0
Ny, Nx = prob_true_seq.shape[1:]
h      = (Lb - La)/(Nx - 1)
xgrid  = np.linspace(La, Lb, Nx)
ygrid  = np.linspace(La, Lb, Ny)
X, Y   = np.meshgrid(xgrid, ygrid, indexing='xy')

d_true_seq = np.stack([
    np.sum(prob_true_seq * X, axis=(1, 2)),
    np.sum(prob_true_seq * Y, axis=(1, 2))
], axis=-1) * h**2

d_pred_seq = np.stack([
    np.sum(prob_pred_seq * X, axis=(1, 2)),
    np.sum(prob_pred_seq * Y, axis=(1, 2))
], axis=-1) * h**2

# ---------- helper utilities -----------------------------------
def rel_l2(pred, ref):
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps)

def batch_metric(pred_seq, ref_seq, fn):
    return np.mean([fn(p, r) for p, r in zip(pred_seq, ref_seq)])
#import pdb; pdb.set_trace()
# ---------- metrics: probability field -------------------------
rel_l2_prob  = rel_l2(pred_sequence, true_sequence)
ssim_prob    = batch_metric(prob_pred_seq, prob_true_seq,
                            lambda a, b: ssim(a, b, data_range=b.ptp()))
spec_prob    = batch_metric(prob_pred_seq, prob_true_seq, spectral_mse_2d)
psnr_prob    = batch_metric(prob_pred_seq, prob_true_seq,
                            lambda a, b: psnr(a, b))
# ---------- metrics: marginal density ρᵧ ------------------------
rel_l2_marg  = rel_l2(rho_y_pred_seq, rho_y_true_seq)
# ---------- metrics: dipole moment -----------------------------
rel_l2_dip   = rel_l2(d_pred_seq, d_true_seq)


# ===============================================================
#  Baseline (upsampled) metrics – added
# ===============================================================
prob_bic_seq = phase2_bicubic_sequence.numpy()**2            # (T₂,256,256)
rho_y_bic_seq = np.sum(prob_bic_seq, axis=2)                 # (T₂,256)

d_bic_seq = np.stack([
    np.sum(prob_bic_seq * X, axis=(1, 2)),
    np.sum(prob_bic_seq * Y, axis=(1, 2))
], axis=-1) * h**2

rel_l2_prob_up = rel_l2(bicubic_sequence, true_sequence)
ssim_prob_up   = batch_metric(prob_bic_seq, prob_true_seq,
                              lambda a, b: ssim(a, b, data_range=b.ptp()))
spec_prob_up   = batch_metric(prob_bic_seq, prob_true_seq, spectral_mse_2d)
psnr_prob_up   = batch_metric(phase2_bicubic_sequence.numpy(),
                              phase2_true_sequence.numpy(), psnr)

rel_l2_marg_up = rel_l2(rho_y_bic_seq, rho_y_true_seq)
rel_l2_dip_up  = rel_l2(d_bic_seq,     d_true_seq)


# ---------- print summary --------------------------------------
print('======================================================')
print('Phase-2 probability-density metrics')
print(f'  Relative L2        : pred = {rel_l2_prob   :.6e} | up = {rel_l2_prob_up :.6e}')
print(f'  SSIM   (mean)      : pred = {ssim_prob    :.6e} | up = {ssim_prob_up  :.6e}')
print(f'  Spectral MSE(mean) : pred = {spec_prob    :.6e} | up = {spec_prob_up  :.6e}')
print(f'  PSNR  (mean)       : pred = {psnr_prob    :.6e} | up = {psnr_prob_up  :.6e}')
print('------------------------------------------------------')
print('Phase-2 marginal ρ(y) metrics')
print(f'  Relative L2        : pred = {rel_l2_marg  :.6e} | up = {rel_l2_marg_up :.6e}')
print('Phase-2 dipole-moment metrics')
print(f'  Relative L2        : pred = {rel_l2_dip   :.6e} | up = {rel_l2_dip_up  :.6e}')
print('======================================================')

# ===============================================================
#  --------  P H A S E-2  P L O T   G A L L E R Y  --------------
# ===============================================================
def tex_sci(x, sig=2):
    """Return `x` as $a\\times10^{b}$ with `sig` significant digits."""
    if x == 0: return "0"
    exp = int(np.floor(np.log10(abs(x))))
    coef = x / 10**exp
    return rf"{coef:.{sig}g}\times10^{{{exp}}}"

out_dir_phase2 = os.path.join(FIG_OUT_DIR, "phase2_plots")
os.makedirs(out_dir_phase2, exist_ok=True)

plot_stride = 250            # save every 250-th Phase-2 frame
scale       = 2              # 64 ➜ 256 up-sampling factor

print("Generating Phase-2 evaluation figures …")
coarse_phase2_sequence = torch.stack(coarse_amp_frames)  # (T₂,64,64)

for idx in range(0, phase2_pred_sequence.size(0), plot_stride):

    # ---------- data & up-sampling ------------------------------
    rho_cg   = coarse_phase2_sequence[idx].numpy()
    rho_up   = F.interpolate(torch.tensor(rho_cg)[None, None],
                             scale_factor=scale,
                             mode="bicubic",
                             align_corners=False).squeeze().numpy()
    rho_pred = phase2_pred_sequence[idx].numpy()
    rho_fine = phase2_true_sequence[idx].numpy()

    # ---------- colour limits -----------------------------------
    vmin, vmax = rho_fine.min(), rho_fine.max()

    # ---------- error maps & scale ------------------------------
    err_up   = np.abs(rho_up   - rho_fine)
    err_pred = np.abs(rho_pred - rho_fine)
    err_max  = max(err_up.max(), err_pred.max())

    # ---------- relative L2 errors ------------------------------
    rl2_up   = np.linalg.norm(rho_up   - rho_fine) / (np.linalg.norm(rho_fine) + eps)
    rl2_pred = np.linalg.norm(rho_pred - rho_fine) / (np.linalg.norm(rho_fine) + eps)

    # ---------- figure layout -----------------------------------
    fig = plt.figure(figsize=(12, 6))
    gs  = gridspec.GridSpec(
        2, 4,
        width_ratios=[1, 1, 1, 0.05],
        wspace=0.05,
        hspace=0.25
    )

    # axes -------------------------------------------------------
    ax00 = fig.add_subplot(gs[0, 0]); ax01 = fig.add_subplot(gs[0, 1]); ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0]); ax11 = fig.add_subplot(gs[1, 1]); ax12 = fig.add_subplot(gs[1, 2])

    # ---------- top row: fields ---------------------------------
    field_panels = [rho_cg, rho_up, rho_pred]
    field_titles = [
        "True coarse dynamics",
        "Upsampled coarse dynamics",
        "Predicted fine dynamics"
    ]

    for ax, img, ttl in zip([ax00, ax01, ax02], field_panels, field_titles):
        h_field = ax.imshow(img, origin="lower", cmap="viridis")
        ax.set_title(ttl, fontsize=14, fontweight='bold')
        ax.set_xlabel(r"$x$", fontsize=12, fontweight='bold', labelpad=-7)
        ax.set_ylabel(r"$y$", fontsize=12, fontweight='bold', labelpad=-7)
        ax.set_xticks([0, img.shape[1]-1]); ax.set_xticklabels([0, img.shape[1]], fontsize=10)
        ax.set_yticks([0, img.shape[0]-1]); ax.set_yticklabels([img.shape[0], 0], fontsize=10)
        ax.grid(color='white', linewidth=0.5, linestyle='--', alpha=0.3)

    # ---------- bottom row: fine truth & errors -----------------
    error_panels = [rho_fine, err_up, err_pred]
    error_titles = [
        "True fine dynamics",
        rf"Upsampled (error = ${tex_sci(rl2_up)}$)",
        rf"FUnet (error = ${tex_sci(rl2_pred)}$)"
    ]

    for ax, img, ttl in zip([ax10, ax11, ax12], error_panels, error_titles):
        if ttl.startswith("True"):
            h_err = ax.imshow(img, origin="lower", cmap="viridis")
        else:
            h_err = ax.imshow(img, origin="lower", cmap="jet", vmin=0, vmax=err_max)
        ax.set_title(ttl, fontsize=12, fontweight='bold')
        ax.set_xlabel(r"$x$", fontsize=12, fontweight='bold', labelpad=-7)
        ax.set_ylabel(r"$y$", fontsize=12, fontweight='bold', labelpad=-7)
        ax.set_xticks([0, img.shape[1]-1]); ax.set_xticklabels([0, img.shape[1]], fontsize=10)
        ax.set_yticks([0, img.shape[0]-1]); ax.set_yticklabels([img.shape[0], 0], fontsize=10)
        ax.grid(color='white', linewidth=0.5, linestyle='--', alpha=0.3)

    # ---------- colour-bars ------------------------------------
    cax_field = fig.add_subplot(gs[0, 3])
    cax_err   = fig.add_subplot(gs[1, 3])

    fig.colorbar(h_field, cax=cax_field, orientation="vertical", label=r"$|\Psi|$")
    fig.colorbar(h_err,   cax=cax_err,   orientation="vertical", label="error")

    # ---------- save -------------------------------------------
    fig.savefig(
        os.path.join(out_dir_phase2, f"phase2_frame_{idx:05d}.pdf"),
        bbox_inches="tight", dpi=300
    )
    plt.close(fig)

print(f"Phase-2 figure gallery written to {out_dir_phase2}")

# ===============================================================
#  --------  Animation (Phase-1 + Phase-2) ----------------------
# ===============================================================
png_dir  = os.path.join(FIG_OUT_DIR, 'frames'); os.makedirs(png_dir, exist_ok=True)
gif_name = os.path.join(FIG_OUT_DIR, 'tdse_phase1_phase2_vs_bicubic.gif')

rel_l2_pred = (torch.linalg.vector_norm(pred_sequence - true_sequence, ord=2, dim=(1,2)) /
               (torch.linalg.vector_norm(true_sequence, ord=2, dim=(1,2)) + 1e-12)).numpy()
rel_l2_bic  = (torch.linalg.vector_norm(bicubic_sequence - true_sequence, ord=2, dim=(1,2)) /
               (torch.linalg.vector_norm(true_sequence, ord=2, dim=(1,2)) + 1e-12)).numpy()

def plot_frame(idx):
    truth = true_sequence[idx].numpy()
    pred  = pred_sequence[idx].numpy()
    bic   = bicubic_sequence[idx].numpy()
    err_p = np.abs(pred - truth)
    err_b = np.abs(bic  - truth)
    vmax  = max(truth.max(), pred.max(), bic.max())

    fig, ax = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    ax[0].imshow(truth, cmap='viridis'); ax[0].set_title('Ground truth')
    ax[1].imshow(pred,  cmap='viridis'); ax[1].set_title('Predicted')
    ax[2].imshow(bic,   cmap='viridis'); ax[2].set_title('Upsampled')
    ax[3].imshow(err_p, cmap='inferno'); ax[3].set_title('Prediction error')
    ax[4].imshow(err_b, cmap='inferno'); ax[4].set_title('Upsampled error')
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fname = os.path.join(png_dir, f'frame_{idx:05d}.png')
    plt.savefig(fname, dpi=110); plt.close()
    return fname

print('Rendering PNG frames …')
frame_idx = list(range(0, WIN)) + list(range(WIN, pred_sequence.size(0), 100))
if (pred_sequence.size(0) - 1) not in frame_idx:
    frame_idx.append(pred_sequence.size(0) - 1)
png_files = [plot_frame(i) for i in frame_idx]

print(f'Creating GIF  ➜  {gif_name}')
with imageio.get_writer(gif_name, mode='I') as writer:
    for idx, fname in zip(frame_idx, png_files):
        img = imageio.imread(fname)
        writer.append_data(img, {'duration': 5.0 if idx == WIN else 0.07})
print('GIF complete.')
