import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
scale            = 4                  # upsampling factor (32 → 128)
Nf               = 128
Nc               = Nf // scale
c                = 0.5
dt               = 0.01
dx_c, dy_c       = 1.0 / Nc, 1.0 / Nc
dx_f, dy_f       = 1.0 / Nf, 1.0 / Nf
c2dt2            = (c * dt)**2
device           = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# Paths
STATS_PATH       = './data/wave_phase2_stats_c=0.5.pt'
MODEL_CHECKPOINT = '/pscratch/sd/h/hbassi/models/2d_wave_FUnet_best_v1_phase2_medium_sf=4.pth'
FINE_DATA_PATH   = '/pscratch/sd/h/hbassi/wave_dataset_multi_sf_modes=4_kmax=4/u_fine.npy'
OUTPUT_DIR       = './extrapolation_outputs_wave'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Operators
# ------------------------------------------------------------
def projection_operator(fine, factor):
    # fine: [T, Hf, Wf]
    return fine[..., ::factor, ::factor]  # [T, Hc, Wc]

def coarse_time_step_wave(coarse_prev, coarse_curr):
    B, T, Hc, Wc = coarse_curr.shape
    M = B * T
    u_nm1 = coarse_prev.reshape(M, Hc, Wc)
    u_n   = coarse_curr.reshape(M, Hc, Wc)
    lap = (
        (torch.roll(u_n,  +1, 1) + torch.roll(u_n,  -1, 1) - 2*u_n) / dx_c**2
      + (torch.roll(u_n,  +1, 2) + torch.roll(u_n,  -1, 2) - 2*u_n) / dy_c**2
    )
    u_np1 = 2*u_n - u_nm1 + c2dt2 * lap
    return u_np1.view(B, T, Hc, Wc)

def fine_time_step_wave_frame(u_nm1, u_n):
    lap = (
        (torch.roll(u_n, +1, 0) + torch.roll(u_n, -1, 0) - 2*u_n) / dx_f**2
      + (torch.roll(u_n, +1, 1) + torch.roll(u_n, -1, 1) - 2*u_n) / dy_f**2
    )
    return 2*u_n - u_nm1 + c2dt2 * lap

# ================================================================
# Coordinate → Fourier features
# ================================================================
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
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid

# ================================================================
# Fourier Neural Operator 2-D spectral layer & U-Net blocks
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
    def compl_mul2d(inp, w):
        return torch.einsum('bixy,ioxy->boxy', inp, w)
    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(B, self.weight.size(1), H, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
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
        self.conv = nn.Conv2d(in_ch, out_ch*(upscale**2), 3, padding=1)
        self.ps   = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self, x): return self.act(self.ps(self.conv(x)))

class SuperResUNet(nn.Module):
    def __init__(self, in_channels=101, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0,
                 final_scale=4):
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
            nn.GELU()
        )
        self.up1  = PixelShuffleUpsample(lift_dim*2, lift_dim*2, upscale=1)
        self.dec2 = ConvBlock(lift_dim*4, lift_dim)
        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim, upscale=2)
        self.dec1 = ConvBlock(lift_dim*2, lift_dim//2)
        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim//2, lift_dim//4, upscale=final_scale),
            ConvBlock(lift_dim//4, lift_dim//4)
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim//4, 32, 3, padding=1),
            nn.GELU(),
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
        d2 = self.up1(b); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up2(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = self.dec0(d1)
        return self.out_head(d0)

# ------------------------------------------------------------
# Load data and stats
# ------------------------------------------------------------
def load_fine_trajectories():
    arr = np.load(FINE_DATA_PATH)  # shape: (N, T, Hf, Wf)
    return torch.tensor(arr, dtype=torch.float32, device=device)

stats = torch.load(STATS_PATH, map_location=device)
data_mean = stats['mean']  # [1, T_win, 1, 1]
data_std  = stats['std']

# ------------------------------------------------------------
# Extrapolation routine
# ------------------------------------------------------------
def evaluate_long_extrap_wave(sample_idx: int, num_steps: int, plot_interval: int = 100):
    # load model
    model = SuperResUNet(in_channels=data_mean.shape[1], final_scale=scale).to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # load full fine trajectory
    fine_all = load_fine_trajectories()  # [N, T_h, Hf, Wf]
    N, T_h, Hf, Wf = fine_all.shape
    if not (0 <= sample_idx < N):
        raise IndexError(f"sample_idx must be in [0, {N-1}]")

    traj = fine_all[sample_idx]  # [T_h, Hf, Wf]
    T_win = data_mean.shape[1]

    # initial history & solver state
    history = traj[:T_win].clone()      # [T_win, Hf, Wf]
    u_nm1_f = history[-2].clone()
    u_n_f   = history[-1].clone()

    predictions   = []
    error_history = []  # collect MAE at each step

    for step in trange(num_steps, desc="Extrapolating"):
        # 1) True solver step
        u_np1 = fine_time_step_wave_frame(u_nm1_f, u_n_f)
        u_nm1_f, u_n_f = u_n_f, u_np1

        # 2) Model prediction
        coarse_win = projection_operator(history, scale)
        cw = coarse_win.unsqueeze(0)
        cp, cc = torch.cat([cw[:, :1], cw[:, :-1]], 1), cw
        cn = coarse_time_step_wave(cp, cc)

        norm_in = (cn - data_mean) / data_std
        with torch.no_grad():
            out_norm = model(norm_in)
        pred_window = out_norm * data_std + data_mean
        pred_window = pred_window.squeeze(0)
        next_pred   = pred_window[-1]
        predictions.append(next_pred.cpu().numpy())

        # compute MAE and store
        err_np  = np.abs(next_pred.cpu().numpy() - u_np1.cpu().numpy())
        mae     = err_np.mean()
        error_history.append(mae)

        # 3) Plot every plot_interval, with inset MAE
        if (step+1) % plot_interval == 0:
            t_global = T_win + step
            pred_np   = next_pred.cpu().numpy()
            true_np   = u_np1.cpu().numpy()

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            im0 = ax[0].imshow(pred_np, origin='lower')
            ax[0].set_title(f"Prediction (t={t_global*dt:.2f})")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            im1 = ax[1].imshow(true_np, origin='lower')
            ax[1].set_title(f"True FG (t={t_global*dt:.2f})")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            im2 = ax[2].imshow(err_np, origin='lower', cmap='hot')
            ax[2].set_title(f"Error (t={t_global*dt:.2f})")
            ax[2].text(
                0.05, 0.95, f"MAE = {mae:.3e}",
                transform=ax[2].transAxes,
                fontsize=10, color='white',
                va='top', ha='left',
                bbox=dict(facecolor='black', alpha=0.6, pad=3)
            )
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"sample{sample_idx}_step{t_global*dt:.2f}.pdf"),
                dpi=150
            )
            plt.close(fig)

        # update history
        history = torch.cat([history[1:], next_pred.unsqueeze(0)], dim=0)

    # save error history array
    errors_np = np.array(error_history)
    np.save(os.path.join(OUTPUT_DIR, f"sample{sample_idx}_error_history.npy"), errors_np)

    # plot accumulated error over time
    times = (np.arange(1, num_steps+1) * dt) + T_win*dt
    plt.figure(figsize=(6,4))
    plt.plot(times, errors_np, marker='o', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"sample{sample_idx}_error_curve.pdf"), dpi=150)
    plt.close()

    return np.stack(predictions, axis=0), errors_np

# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave eqn long-term extrapolation")
    parser.add_argument("--sample_idx",    type=int, default=777, help="Trajectory index")
    parser.add_argument("--num_steps",     type=int, default=400, help="Extrap steps")
    parser.add_argument("--plot_interval", type=int, default=100, help="Plot every N steps")
    args = parser.parse_args()

    preds, errs = evaluate_long_extrap_wave(
        sample_idx    = args.sample_idx,
        num_steps     = args.num_steps,
        plot_interval = args.plot_interval
    )
