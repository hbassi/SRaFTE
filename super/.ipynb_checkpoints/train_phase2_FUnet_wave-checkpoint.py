import os
import math
import logging
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(
    filename='training_wave_32to128_phase2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# Wave equation parameters (must match data‐generation)
c        = 0.5
dt       = 0.01
Lx       = 1.0
Ly       = 1.0

# Upsampling scale
scale    = 4               # coarse 32 → fine 128
Nx_fine  = 128
Nx_coarse = Nx_fine // scale
dx_c     = Lx / Nx_coarse
dy_c     = Ly / Nx_coarse
c2dt2    = (c * dt)**2

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "/pscratch/sd/h/hbassi/wave_dataset_multi_sf_modes=4_kmax=4"

# ------------------------------------------------------------
# Simple downsampling projection
# ------------------------------------------------------------
def projection_operator(fine, factor):
    # fine: [B, T, Hf, Wf]
    return fine[..., ::factor, ::factor]  # [B, T, Hc, Wc]

# ------------------------------------------------------------
# One‐step coarse‐wave propagator (vectorized)
# ------------------------------------------------------------
def coarse_time_step_wave(coarse_prev, coarse_curr):
    """
    coarse_prev, coarse_curr: [B, T, Hc, Wc]
    returns u_np1:        [B, T, Hc, Wc]
    via u_{n+1} = 2 u_n - u_{n-1} + c^2 dt^2 Δ u_n
    """
    B, T, Hc, Wc = coarse_curr.shape
    M = B * T

    u_nm1 = coarse_prev.reshape(M, Hc, Wc)
    u_n   = coarse_curr.reshape(M, Hc, Wc)

    lap = (
        (torch.roll(u_n, +1, 1) + torch.roll(u_n, -1, 1) - 2*u_n) / dx_c**2
      + (torch.roll(u_n, +1, 2) + torch.roll(u_n, -1, 2) - 2*u_n) / dy_c**2
    )
    u_np1 = 2*u_n - u_nm1 + c2dt2 * lap

    return u_np1.reshape(B, T, Hc, Wc)
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
# PixelShuffle-based up-sample block 
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

        # ---- bottleneck (no extra pooling) ----
        b  = self.bottleneck(e2)        # Hc/2

        # ---- decoder ----
        d2 = self.up1(b)                             # Hc/2  (spatially matches e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up2(d2)                            # Hc
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.dec0(d1)                           # Hf
        return self.out_head(d0)                     # (B,11,Hf,Wf)  normalised

# ------------------------------------------------------------
# Load only fine‐scale data, form input/output pairs
# ------------------------------------------------------------
def load_fine_data_all():
    """
    Returns:
      inputs: [N, T, Hf, Wf]  (u(t=0..T-1))
      targets:[N, T, Hf, Wf]  (u(t=1..T))
    """
    u_fine = np.load(os.path.join(DATA_DIR, "u_fine.npy"))
    inp    = torch.tensor(u_fine[:, :-1], dtype=torch.float32)
    tgt    = torch.tensor(u_fine[:, 1: ], dtype=torch.float32)
    return inp, tgt

# ------------------------------------------------------------
# Phase-2 Training: project→step→super-resolve
# ------------------------------------------------------------
def train_finetune_wave():
    # load fine data
    inputs, targets = load_fine_data_all()  # [N,T,128,128]
    N, T, Hf, Wf   = inputs.shape
    print("Data shapes:", inputs.shape, targets.shape)

    # compute normalization stats on inputs (fine)
    data_mean = inputs.mean(dim=(0,2,3), keepdim=True)  # [1,T,1,1]
    data_std  = inputs.std (dim=(0,2,3), keepdim=True).clamp_min(1e-8)
    torch.save({'mean':data_mean,'std':data_std},
               f'./data/wave_phase2_stats_c={c}.pt')

    # dataset & loaders
    ds = TensorDataset(inputs, targets)
    n_train = int(0.9 * N)
    train_ds, val_ds = random_split(ds, [n_train, N-n_train])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4)

    # model, optimizer, scheduler, loss
    model = SuperResUNet(in_channels=T, final_scale=scale).to(device)
    model.load_state_dict(torch.load('/pscratch/sd/h/hbassi/models/2d_wave_FUnet_best_v1_medium_sf=4.pth'))
    opt   = optim.AdamW(model.parameters(), lr=7e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-6)
    crit  = nn.L1Loss()

    best_val = float('inf')
    epochs   = 2000

    for ep in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0.0
        for fine_seq, target_seq in train_loader:
            fine_seq   = fine_seq.to(device)    # [B,T,128,128]
            target_seq = target_seq.to(device)  # [B,T,128,128]

            # project fine → coarse
            coarse = projection_operator(fine_seq, scale)  # [B,T,32,32]

            # build coarse_prev, coarse_curr
            coarse_prev = torch.cat([coarse[:, :1], coarse[:, :-1]], dim=1)
            coarse_curr = coarse

            # step coarse one Δt
            coarse_next = coarse_time_step_wave(coarse_prev, coarse_curr)  # [B,T,32,32]

            # normalize & forward
            norm_in = (coarse_next - data_mean.to(device)) / data_std.to(device)
            out     = model(norm_in)  # [B,T,128,128]
            pred    = out * data_std.to(device) + data_mean.to(device)

            loss = crit(pred, target_seq)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fine_seq, target_seq in val_loader:
                fine_seq   = fine_seq.to(device)
                target_seq = target_seq.to(device)

                coarse      = projection_operator(fine_seq, scale)
                coarse_prev = torch.cat([coarse[:, :1], coarse[:, :-1]], dim=1)
                coarse_curr = coarse
                coarse_next = coarse_time_step_wave(coarse_prev, coarse_curr)

                norm_in = (coarse_next - data_mean.to(device)) / data_std.to(device)
                out     = model(norm_in)
                pred    = out * data_std.to(device) + data_mean.to(device)

                val_loss += crit(pred, target_seq).item()
        val_loss /= len(val_loader)

        # logging & checkpoint
        if ep % 10 == 0:
            msg = f"Epoch {ep} | Train {train_loss:.6f} | Val {val_loss:.6f}"
            print(msg)
            logging.info(msg)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"/pscratch/sd/h/hbassi/models/2d_wave_FUnet_best_v1_phase2_medium_sf=4.pth")
            logging.info(f"New best val {best_val:.6f} at epoch {ep}")

if __name__ == "__main__":
    train_finetune_wave()
