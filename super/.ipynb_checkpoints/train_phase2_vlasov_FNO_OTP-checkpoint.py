#!/usr/bin/env python
# ================================================================
#  train_phase2_FNO_OTP_vlasov.py   (One‑Time‑Propagator FNO)
# ----------------------------------------------------------------
#  Learns  f_e(t+Δt)  ≈  FNO(  f_e(t‑100:…:t)  )   on the 128×128 grid
#  ─ same Fourier‑layer stack as the Phase‑1 SR‑FNO but WITHOUT upscaling
# ================================================================
import os, sys, math, contextlib
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

# -----------------------------------------------------------------
# 0 ▸  PATHS  (adjust to your environment)
# -----------------------------------------------------------------
SAVE_DIR   = "/pscratch/sd/h/hbassi/models"
os.makedirs(SAVE_DIR, exist_ok=True)
CONFIGS  = [(8, 8)]

PATH_FINE_E = (
   f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_fine_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
)
STATS_PATH  = "./data/2d_vlasov_fno_otp_stats.pt"
WEIGHTS_BEST = os.path.join(SAVE_DIR, "2d_vlasov_two-stream_FNO_OTP_phase2_best.pth")

# -----------------------------------------------------------------
# 1 ▸  HYPER‑PARAMETERS
# -----------------------------------------------------------------
HIST_LEN   = 200            # channels (t‑100 … t)
BATCH_SIZE = 2
EPOCHS     = 2000
LR         = 5e-4
M0, M1     = 16, 16         # Fourier modes per axis
WIDTH      = 64             # channel width in hidden layers
LAYERS     = 3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------
# 2 ▸  FNO BUILDING BLOCKS  (reuse SpectralConv2d)
# -----------------------------------------------------------------
from train_phase1_FNO_2d_vlasov import SpectralConv2d   # provides FFT‑based conv

class FNO2dOTP(nn.Module):
    """Three‑layer Fourier Neural Operator (no upscaling)."""
    def __init__(self, in_ch=HIST_LEN, out_ch=1,
                 modes1=M0, modes2=M1, width=WIDTH, layers=LAYERS):
        super().__init__()
        self.width = width
        self.fc0 = nn.Conv2d(in_ch, width, 1)          # lift

        self.spectral_convs = nn.ModuleList(
            SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)
        )
        self.ws = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.act = nn.GELU()

        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        # Input shape: (B, C=101, 128,128)
        x = self.fc0(x)                                 # (B, width, H, W)
        for spec, w in zip(self.spectral_convs, self.ws):
            x = spec(x) + w(x)
            x = self.act(x)
        x = self.act(self.fc1(x))
        return self.fc2(x)                              # (B, 1, 128,128)

# -----------------------------------------------------------------
# 3 ▸  DATA LOADING
# -----------------------------------------------------------------
def load_dataset():
    """
    Returns  TensorDataset(  inp , tgt  )
      * inp : (B, 101, 128,128)  fine‑grid history
      * tgt : (B,  1, 128,128)   next‑time electron slice
    Saves mean/std for normalisation.
    """
    fine_e = np.load(PATH_FINE_E, mmap_mode="r")        # (N_traj, T,128,128)

    # --- build tensors --------------------------------------------------
    #  For demo: use ALL trajectories and first 102 frames so that
    #  frame k (0‑100) → target frame k+1
    inp  = torch.from_numpy(fine_e[:, :HIST_LEN     ]).float()  # (N,T=101,H,W)
    tgt  = torch.from_numpy(fine_e[:, 1:HIST_LEN+1 ]).float()   # (N,T=101,H,W)
    # keep only last slice of tgt per sample
    # shape: (N,1,128,128)
    tgt  = tgt[:, -1:, :, :]
    # flatten traj dim into batch
    B, T, H, W = inp.shape
    inp = inp.view(-1, HIST_LEN, H, W)
    tgt = tgt.view(-1, 1,        H, W)

    # --- normalisation stats (channel‑wise over spatial dims) ----------
    data_mean = inp.mean(dim=(0, 2, 3), keepdim=True)
    data_std  = inp.std (dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    torch.save({'data_mean': data_mean, 'data_std': data_std}, STATS_PATH)

    print("Dataset shapes:", inp.shape, tgt.shape)
    return TensorDataset(inp, tgt), data_mean, data_std

def get_loaders(batch=BATCH_SIZE):
    ds, μ, σ = load_dataset()
    n_tr = int(0.9 * len(ds))
    tr, va = random_split(ds, [n_tr, len(ds) - n_tr])
    return (DataLoader(tr, batch, shuffle=True, pin_memory=True),
            DataLoader(va, batch, pin_memory=True),
            μ.to(DEVICE), σ.to(DEVICE))

# -----------------------------------------------------------------
# 4 ▸  TRAIN LOOP
# -----------------------------------------------------------------
def train_fno_otp():
    tr_loader, va_loader, μ, σ = get_loaders()
    model = FNO2dOTP().to(DEVICE)
    crit  = nn.L1Loss()
    opt   = optim.AdamW(model.parameters(), lr=LR)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, 1e-6)

    best_val = math.inf
    for ep in trange(EPOCHS):
        # ---------- training ---------------------------------
        model.train(); tr_loss = 0.0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = (x - μ) / σ
            pred = model(x) * σ[:, :1] + μ[:, :1]        # denorm
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        # ---------- validation -------------------------------
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = (x - μ) / σ
                pred = model(x) * σ[:, :1] + μ[:, :1]
                va_loss += crit(pred, y).item()
        va_loss /= len(va_loader); sch.step()

        if ep % 50 == 0:
            print(f"[{ep:4d}] train={tr_loss:.6e} | val={va_loss:.6e}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), WEIGHTS_BEST)
            print(f"   ↳ saved new best (val={best_val:.3e})")

    print("Training complete. Best validation:", best_val)

# -----------------------------------------------------------------
if __name__ == "__main__":
    train_fno_otp()
