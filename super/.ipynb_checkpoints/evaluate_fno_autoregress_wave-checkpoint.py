import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
from tqdm import trange

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
Nf     = 128
scale  = 4                  # unused in this script
Nc     = Nf // scale
dt     = 0.01
dx_f   = 1.0 / Nf
dy_f   = 1.0 / Nf
c      = 0.5
c2dt2  = (c * dt)**2
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# FNO hyperparameters (must match training)
in_ch, out_ch = 101, 101
modes1, modes2 = 16, 16
width         = 128

# Paths
MODEL_CHECKPOINT = '/pscratch/sd/h/hbassi/models/2d_wave_FNO_best_v1_phase2_medium_sf=4.pth'
FINE_DATA_PATH   = '/pscratch/sd/h/hbassi/wave_dataset_multi_sf_modes=4_kmax=4/u_fine.npy'
OUTPUT_DIR       = './fno_extrap_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Fine-grid wave solver (2nd-order finite differences)
# ------------------------------------------------------------
def fine_time_step_wave_frame(u_nm1, u_n):
    lap = (
        (torch.roll(u_n, +1, 0) + torch.roll(u_n, -1, 0) - 2*u_n) / dx_f**2
      + (torch.roll(u_n, +1, 1) + torch.roll(u_n, -1, 1) - 2*u_n) / dy_f**2
    )
    return 2*u_n - u_nm1 + c2dt2 * lap


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
        #x = self.conv1(x) + self.w1(x)
        #x = nn.GELU()(x)
        #x = self.conv2(x) + self.w2(x)
        #x = nn.GELU()(x)
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


# ------------------------------------------------------------
# Load trajectories
# ------------------------------------------------------------
def load_fine_trajectories():
    arr = np.load(FINE_DATA_PATH)         # (N, T, H, W)
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ------------------------------------------------------------
# Autoregressive evaluation
# ------------------------------------------------------------
def evaluate_long_extrap_wave(sample_idx: int, num_steps: int, plot_interval: int = 100):
    # instantiate and load model
    model = FNO2d(in_ch, out_ch, modes1, modes2, width).to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # load data
    fine_all = load_fine_trajectories()  # [N, T_h, H, W]
    N, T_h, H, W = fine_all.shape
    assert 0 <= sample_idx < N

    traj = fine_all[sample_idx]          # [T_h, H, W]
    # initial history of length in_ch:
    history = traj[:in_ch].clone()       # [in_ch, H, W]
    # solver initial state from last two of history
    u_nm1_f = history[-2].clone()
    u_n_f   = history[-1].clone()

    preds = []
    errors = []

    for step in trange(num_steps, desc="Extrapolating"):
        # solver ground truth
        u_np1 = fine_time_step_wave_frame(u_nm1_f, u_n_f)
        u_nm1_f, u_n_f = u_n_f, u_np1

        # model prediction
        window = history.unsqueeze(0)      # [1, in_ch, H, W]
        with torch.no_grad():
            out = model(window)            # [1, out_ch, H, W]
        # take last channel as next prediction
        next_pred = out[0, -1, :, :]      # [H, W]
        preds.append(next_pred.cpu().numpy())

        # error vs solver
        err_map = np.abs(next_pred.cpu().numpy() - u_np1.cpu().numpy())
        mae     = err_map.mean()
        errors.append(mae)

        # slide history window
        history = torch.cat([history[1:], next_pred.unsqueeze(0)], dim=0)  # [in_ch, H, W]

        # plotting
        if (step+1) % plot_interval == 0:
            t = step + in_ch
            fig, ax = plt.subplots(1,3,figsize=(12,4))
            im0 = ax[0].imshow(preds[-1], origin='lower')
            ax[0].set_title(f"FNO Pred (t={t*dt:.2f})")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            im1 = ax[1].imshow(u_np1.cpu().numpy(), origin='lower')
            ax[1].set_title(f"Solver True (t={t*dt:.2f})")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            im2 = ax[2].imshow(err_map, origin='lower', cmap='hot')
            ax[2].set_title(f"|Error| MAE={mae:.3e}")
            ax[2].text(
                0.05, 0.95, f"MAE={mae:.2e}",
                transform=ax[2].transAxes,
                color='white', va='top', ha='left',
                bbox=dict(facecolor='black', alpha=0.6, pad=3)
            )
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"fno_sample{sample_idx}_step{t}.png"), dpi=150)
            plt.close(fig)

    # save predictions and error history
    preds_np  = np.stack(preds, axis=0)   # [num_steps, H, W]
    errors_np = np.array(errors)          # [num_steps]
    np.save(os.path.join(OUTPUT_DIR, f"fno_sample{sample_idx}_preds.npy"), preds_np)
    np.save(os.path.join(OUTPUT_DIR, f"fno_sample{sample_idx}_errors.npy"), errors_np)

    # plot MAE vs time
    times = (np.arange(1, num_steps+1) + in_ch) * dt
    plt.figure(figsize=(6,4))
    plt.plot(times, errors_np, marker='o', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'FNO Autoregressive MAE (sample {sample_idx})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fno_sample{sample_idx}_error_curve.pdf"), dpi=150)
    plt.close()

    return preds_np, errors_np


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNO2d autoregressive evaluation")
    parser.add_argument("--sample_idx",    type=int, default=777,   help="Trajectory index")
    parser.add_argument("--num_steps",     type=int, default=400, help="Number of extrapolation steps")
    parser.add_argument("--plot_interval", type=int, default=100, help="Plot every N steps")
    args = parser.parse_args()

    evaluate_long_extrap_wave(
        sample_idx    = args.sample_idx,
        num_steps     = args.num_steps,
        plot_interval = args.plot_interval
    )
