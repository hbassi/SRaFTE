import torch
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

# -----------------------------------
# Configuration (must match training)
# -----------------------------------
Nc = 32                  # coarse grid size
Nf = 128                 # fine grid size
nu = 0.1               # diffusion coefficient
dt = 0.01                # time step (coarse & fine)
f_coefficient = 0.01     # forcing amplitude
device = torch.device('cpu')


# Paths (adjust as needed)
COARSE_STATS_PATH = f'./data/2d_heat_phase2_stats_nu={nu}.pt'
MODEL_CHECKPOINT = '/pscratch/sd/h/hbassi/models/fine_tuning_best_FUnet_2d_heat_smooth2.pth'
FINE_DATA_PATH   = '/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k_test.npy'#'/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k.npy'
OUTPUT_DIR       = './extrapolation_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Precompute Fourier wavenumbers and forcing on coarse grid
# ------------------------------------------------------------
kx_c_1d = torch.fft.fftfreq(Nc, d=1.0 / Nc) * (2.0 * math.pi)        # (Nc,)
ky_c_1d = kx_c_1d.clone()
kx_c, ky_c = torch.meshgrid(kx_c_1d, ky_c_1d, indexing='ij')         # (Nc, Nc)
k2_c = kx_c ** 2 + ky_c ** 2                                          # (Nc, Nc)
k2_c[0, 0] = 1e-14                                                   # avoid division by zero
exp_c = torch.exp(-nu * k2_c * dt).to(device)                         # (Nc, Nc)

xs_c = (torch.arange(Nc, dtype=torch.float64) + 0.5) / Nc             # (Nc,)
ys_c = (torch.arange(Nc, dtype=torch.float64) + 0.5) / Nc             # (Nc,)
Xc, Yc = torch.meshgrid(xs_c, ys_c, indexing='ij')                    # (Nc, Nc)
f_xy_c = torch.sin(2 * math.pi * Xc) * torch.sin(2 * math.pi * Yc)    # (Nc, Nc)
f_hat_c = torch.fft.fft2(f_xy_c).to(device)                            # (Nc, Nc), complex128
k2_c = k2_c.to(device)

# ------------------------------------------------------------
# Precompute Fourier wavenumbers and forcing on fine grid
# ------------------------------------------------------------
kx_f_1d = torch.fft.fftfreq(Nf, d=1.0 / Nf) * (2.0 * math.pi)        # (Nf,)
ky_f_1d = kx_f_1d.clone()
kx_f, ky_f = torch.meshgrid(kx_f_1d, ky_f_1d, indexing='ij')         # (Nf, Nf)
k2_f = kx_f ** 2 + ky_f ** 2                                          # (Nf, Nf)
k2_f[0, 0] = 1e-14                                                     # avoid division by zero
exp_f = torch.exp(-nu * k2_f * dt).to(device)                         # (Nf, Nf)

xs_f = (torch.arange(Nf, dtype=torch.float64) + 0.5) / Nf             # (Nf,)
ys_f = (torch.arange(Nf, dtype=torch.float64) + 0.5) / Nf             # (Nf,)
Xf, Yf = torch.meshgrid(xs_f, ys_f, indexing='ij')                     # (Nf, Nf)
f_xy_f = torch.sin(2 * math.pi * Xf) * torch.sin(2 * math.pi * Yf)    # (Nf, Nf)
f_hat_f = torch.fft.fft2(f_xy_f).to(device)                            # (Nf, Nf), complex128
k2_f = k2_f.to(device)

# ================================================================
# Projection Operator: Fine -> Coarse (downsampling)
# ================================================================
def projection_operator(fine_data, factor=2):
    """
    Downsamples the input spatially by the given factor (applied channel–wise).
    fine_data: Tensor of shape [B, C, Hf, Wf] or [C, Hf, Wf].
    Returns: Tensor of shape [B, C, Hc, Wc] or [C, Hc, Wc].
    """
    return fine_data[..., ::factor, ::factor]

def coarse_time_step_heat(coarse_seq: torch.Tensor) -> torch.Tensor:
    """
    Advance one time step for every frame in a batch of coarse 2D heat-equation frames.

    Input:
        coarse_seq: Tensor of shape [B, Nc, Nc] or [B, T, Nc, Nc].
    Output:
        next_coarse: Tensor of the same shape (float32),
                     where each frame is advanced by one time step.
    """
    # If input has time dimension:
    if coarse_seq.ndim == 4:
        B, T, _, _ = coarse_seq.shape
        u_flat = coarse_seq.reshape(-1, Nc, Nc).to(torch.float64)  # [B*T, Nc, Nc]
    elif coarse_seq.ndim == 3:
        B = coarse_seq.shape[0]
        u_flat = coarse_seq.to(torch.float64).unsqueeze(1)  # [B, 1, Nc, Nc]
        u_flat = u_flat.reshape(-1, Nc, Nc)
    else:
        raise ValueError("coarse_seq must have shape [B, Nc, Nc] or [B, T, Nc, Nc]")

    # FFT in batch → [M, Nc, Nc], complex128
    u_hat = torch.fft.fft2(u_flat)

    # Multiply by integrating factor and add forcing
    u_hat = u_hat.to(exp_c.device) * exp_c.unsqueeze(0)
    numerator = f_hat_c * (1.0 - exp_c)                      # [Nc, Nc]
    denom = (nu * k2_c)                                      # [Nc, Nc]
    forcing_term = (numerator / denom) * f_coefficient       # [Nc, Nc]
    u_hat = u_hat + forcing_term.unsqueeze(0)                # broadcast

    # Inverse FFT → [M, Nc, Nc]
    u_next = torch.real(torch.fft.ifft2(u_hat))

    u_next = u_next.reshape(B, -1, Nc, Nc) if coarse_seq.ndim == 4 else u_next.reshape(B, Nc, Nc)
    return u_next.to(torch.float32)

def fine_time_step_heat_frame(u_f: torch.Tensor) -> torch.Tensor:
    """
    Advance one time step for a single fine-scale 2D heat-equation frame.

    Input:
        u_f: Tensor of shape [Hf, Wf], float32 or float64.
    Output:
        next_u_f: Tensor of shape [Hf, Wf], float32.
    """
    # Ensure float64 for FFT accuracy
    u_f64 = u_f.to(torch.float64)

    # FFT → [Hf, Wf], complex128
    u_hat = torch.fft.fft2(u_f64)

    # Multiply by fine integrating factor and add forcing
    u_hat = u_hat.to(exp_f.device) * exp_f
    numerator_f = f_hat_f * (1.0 - exp_f)                     # [Nf, Nf]
    denom_f = (nu * k2_f)                                     # [Nf, Nf]
    forcing_term_f = (numerator_f / denom_f) * f_coefficient  # [Nf, Nf]
    u_hat = u_hat + forcing_term_f                            # broadcast

    # Inverse FFT → [Hf, Wf]
    u_next = torch.real(torch.fft.ifft2(u_hat))
    return u_next.to(torch.float32)

# ================================================================
# Model Definition (must match training)
# ---------------------------------------------------------------
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):                               # (B,H,W,2)
        proj = 2 * math.pi * torch.matmul(coords, self.B)    # (B,H,W,map_size)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)                        # (B,2*map_size,H,W)

def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid                                              # (B,H,W,2)

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

    def forward(self, x):                                    # (B,C,H,W) real
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
        return torch.fft.irfft2(out_ft, s=(H, W))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale ** 2), 3, padding=1)
        self.pix  = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.pix(self.conv(x)))

class SuperResUNet(nn.Module):
    def __init__(
        self,
        in_channels=101,
        lift_dim=128,
        mapping_size=64,
        mapping_scale=5.0,
        final_scale=4
    ):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        self.enc1 = ConvBlock(lift_dim,        lift_dim)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
            nn.GELU()
        )

        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)

        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)

        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )

        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, x):                                 # (B, in_channels, Hc, Wc)
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

# ================================================================
# Data‐loading helper (loads full fine‐scale trajectories)
# ================================================================
def load_fine_data_all():
    """
    Loads fine-scale heat‐equation trajectories.
    Assumes the data is stored in a NumPy array of shape (num_samples, T, Hf, Wf).
    Returns:
        inputs: Tensor of shape [num_samples, T, Hf, Wf] (float32)
    """
    fine_data_np = np.load(FINE_DATA_PATH)  # shape: (num_samples, T, Hf, Wf)
    return torch.tensor(fine_data_np, dtype=torch.float32)

# ================================================================
# Evaluation: long‐time extrapolation on one trajectory
# ================================================================
def evaluate_long_extrap(sample_idx: int, num_extrap_steps: int):
    """
    For a given trajectory index, extrapolate its fine‐scale solution
    for num_extrap_steps beyond the provided horizon.

    Every 100 steps, plot:
      - predicted fine‐grid frame
      - solver‐extrapolated fine‐grid frame (ground truth)
      - absolute error heatmap
    """
    # -- Load statistics (mean, std) from training --
    stats = torch.load(COARSE_STATS_PATH, map_location=device)
    data_mean = stats['data_mean'].to(device)[:, :101]   # shape: [1, C, 1, 1]
    data_std  = stats['data_std'].to(device)[:, :101]    # shape: [1, C, 1, 1]

    # -- Instantiate model & load weights --
    model = SuperResUNet(in_channels=data_mean.shape[1], final_scale=Nf // Nc).to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # -- Load full fine‐scale data and pick one trajectory --
    fine_all = load_fine_data_all()  # [num_samples, T, Hf, Wf]
    num_samples, T_horizon, Hf, Wf = fine_all.shape

    if sample_idx < 0 or sample_idx >= num_samples:
        raise IndexError(f"sample_idx must be in [0, {num_samples-1}]")

    # Extract the entire trajectory (shape [T_horizon, Hf, Wf])
    trajectory_fine = fine_all[sample_idx].to(device)  # float32

    # The model requires a window of length T_win as input (in_channels = T_win)
    T_win = data_mean.shape[1]  # e.g., 101
    if T_horizon < T_win:
        raise ValueError(f"Trajectory length ({T_horizon}) is shorter than window length ({T_win})")

    # Initialize a buffer of the last T_win fine‐frames as the "history"
    # We'll start at t = 0,…,T_win-1 from the true data, then predict t=T_win, T_win+1, …
    history = trajectory_fine[:T_win].clone()  # shape [T_win, Hf, Wf]

    # Initialize the solver‐extrapolated "true" frame at t = T_win-1
    true_frame = trajectory_fine[T_win - 1].clone()  # [Hf, Wf]

    # Container for saving predicted fine‐frames (beyond the horizon)
    predictions = []

    # For steps from t = T_win to T_win + num_extrap_steps - 1:
    for step in trange(num_extrap_steps, desc="Extrapolating"):
        # 1) Advance the "true" fine‐scale solver by one step:
        true_frame = fine_time_step_heat_frame(true_frame)  # now at t = T_win + step

        # 2) Prepare the last T_win fine frames as a single input tensor:
        #    shape [1, T_win, Hf, Wf]
        fine_window = history.clone()                      # [T_win, Hf, Wf]
        fine_window = fine_window.unsqueeze(0)             # [1, T_win, Hf, Wf]

        # 3) Project the entire fine_window to coarse:
        #    coarse_window: [1, T_win, Nc, Nc]
        coarse_window = projection_operator(fine_window, factor=Hf // Nc)

        # 4) Step each coarse time‐slice forward by one dt:
        #    coarse_next: [1, T_win, Nc, Nc]
        coarse_next = coarse_time_step_heat(coarse_window)

        # 5) Normalize coarse_next (per‐channel statistics):
        norm_in = (coarse_next - data_mean) / data_std  # broadcasting works
        print(norm_in.min().item(), norm_in.max().item())

        # 6) Run model → output has shape [1, T_win, Hf, Wf]
        with torch.no_grad():
            output_norm = model(norm_in)                  # [1, T_win, Hf, Wf]

        # 7) De-normalize:
        pred_fine_window = output_norm * data_std + data_mean  # [1, T_win, Hf, Wf]
        pred_fine_window = pred_fine_window.squeeze(0)         # [T_win, Hf, Wf]

        # 8) The “next” predicted fine‐frame corresponds to pred_fine_window[T_win-1]
        next_fine = pred_fine_window[-1].clone()               # [Hf, Wf]
        predictions.append(next_fine.cpu().numpy())

        # -- Plot every 100 steps --
        if (step + 1) % 100 == 0:
            t_global = T_win + step
            pred_np = next_fine.cpu().numpy()
            true_np = true_frame.cpu().numpy()
            abs_err = np.abs(pred_np - true_np)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im0 = axe k s[0].imshow(pred_np, origin='lower')
            axes[0].set_title(f"Predicted (t={t_global})")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(true_np, origin='lower')
            axes[1].set_title(f"Solver True (t={t_global})")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(abs_err, origin='lower')
            axes[2].set_title(f"|Error| (t={t_global})")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_step_{t_global}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)

        # 9) Append next_fine to history, and drop the oldest:
        history = torch.cat([history[1:].clone(), next_fine.unsqueeze(0)], dim=0)  # [T_win, Hf, Wf]

    # Convert list to NumPy array: shape [num_extrap_steps, Hf, Wf]
    predictions_np = np.stack(predictions, axis=0)

    # Save predicted frames to disk (e.g., as a .npy)
    out_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_pred_{num_extrap_steps}steps.npy")
    np.save(out_path, predictions_np)
    print(f"Saved predicted trajectory (shape {predictions_np.shape}) to:\n  {out_path}")

    return predictions_np

# ================================================================
# If run as script, perform one example extrapolation
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate long‐time extrapolation for one heat‐equation trajectory"
    )
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of trajectory to extrapolate (0-based)")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Number of time steps to extrapolate beyond training horizon")
    args = parser.parse_args()

    _ = evaluate_long_extrap(sample_idx=args.sample_idx,
                             num_extrap_steps=args.num_steps)
