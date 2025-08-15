#!/usr/bin/env python
# ===============================================================
#  baseline_unet_superres_training.py
#  ---------------------------------------------------------------
#  Baseline image‑to‑image super‑resolution with a plain U‑Net
#  that upsamples coarse Vlasov–Poisson snapshots (32×32) to
#  fine snapshots (128×128).  The original 4‑D dataset (B,T,H,W)
#  is reshaped to (B·T, 1, H, W) so each time slice is treated
#  as an independent sample.
# ===============================================================

import os, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

# ---------------------------------------------------------------- Paths / settings
CONFIGS = [(8, 8)]  # (m_x, m_y) identifiers used in filenames

#PATH_CG = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_coarse_32_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
#PATH_FG = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_fine_128_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
PATH_CG = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_coarse_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_FG = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_fine_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
DIR_MODELS = "/pscratch/sd/h/hbassi/models"
DIR_LOGS   = "./logs"
os.makedirs(DIR_MODELS, exist_ok=True)
os.makedirs(DIR_LOGS,   exist_ok=True)

BATCH_SIZE   = 256
NUM_EPOCHS   = 50
LR_INIT      = 3e-4
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ===============================================================
# 1 ▸ Dataset loader
# ===============================================================
def load_data():
    """
    Returns
    -------
    X : torch.FloatTensor, shape (N, 1, Hc, Wc)
        Coarse‑grid inputs (Z‑scored later).
    Y : torch.FloatTensor, shape (N, 1, Hf, Wf)
        Fine‑grid targets.
    """
    input_cg  = np.load(PATH_CG)  # (B,T,Hc,Wc)
    target_fg = np.load(PATH_FG)  # (B,T,Hf,Wf)
    if input_cg.shape[:2] != target_fg.shape[:2]:
        raise ValueError("Coarse and fine arrays must share (B,T) dimensions.")

    B, T, Hc, Wc = input_cg.shape
    _, _, Hf, Wf = target_fg.shape

    X = torch.from_numpy(input_cg).float().reshape(B * T, 1, Hc, Wc)
    Y = torch.from_numpy(target_fg).float().reshape(B * T, 1, Hf, Wf)
    return X, Y

# ===============================================================
# 2 ▸ U‑Net building blocks
# ===============================================================
def conv_block(in_ch, out_ch, k=3, act=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch)
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)

class UNetSR(nn.Module):
    """
    Vanilla U‑Net.  Internally restores the original input
    resolution and then applies a *single* bilinear upsample by
    `upscale_factor` → final output matches fine grid.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64, upscale_factor: int = 4):
        super().__init__()
        assert upscale_factor in (2, 4, 8), "upscale_factor must be 2, 4, or 8"
        self.upscale_factor = upscale_factor

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = nn.Sequential(conv_block(in_ch, base_ch),
                                  conv_block(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)       #  1/2

        self.enc2 = nn.Sequential(conv_block(base_ch, base_ch * 2),
                                  conv_block(base_ch * 2, base_ch * 2))
        self.pool2 = nn.MaxPool2d(2)       #  1/4

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = nn.Sequential(conv_block(base_ch * 2, base_ch * 4),
                                         conv_block(base_ch * 4, base_ch * 4))

        # ── Decoder (transpose conv) ─────────────────────────────
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)  # 1/2
        self.dec2 = nn.Sequential(conv_block(base_ch * 4, base_ch * 2),
                                  conv_block(base_ch * 2, base_ch * 2))

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)       # 1/1
        self.dec1 = nn.Sequential(conv_block(base_ch * 2, base_ch),
                                  conv_block(base_ch, base_ch))

        self.out_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_head(d1)                     # (N,1,Hc,Wc)
        # Final bilinear upsample to fine resolution
        out = F.interpolate(out,
                            scale_factor=self.upscale_factor,
                            mode='bilinear',
                            align_corners=False)
        return out

# ===============================================================
# 3 ▸ Training loop
# ===============================================================
def train():
    # ---------------- Data ----------------
    print("Loading data …")
    X, Y = load_data()
    print(f"Coarse shape {tuple(X.shape)}   Fine shape {tuple(Y.shape)}")

    upscale_factor = Y.shape[-1] // X.shape[-1]
    assert upscale_factor in (2, 4, 8), "Only ×2, ×4, ×8 factors supported."
    assert Y.shape[-2] // X.shape[-2] == upscale_factor, "Non‑uniform upscale factors."

    # Z‑score coarse data
    mean, std = X.mean(), X.std().clamp_min(1e-8)
    X = (X - mean) / std
    torch.save({'mean': mean, 'std': std}, os.path.join(DIR_LOGS, '2d_vlasov_two-stream_unet_ssr_stats.pt'))

    dataset = TensorDataset(X, Y)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print(f"Samples → Train={len(train_ds)}  Val={len(val_ds)}  |  upscale ×{upscale_factor}")

    # ---------------- Model / Optimiser ----------------
    model = UNetSR(in_ch=1, base_ch=64, upscale_factor=upscale_factor).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=NUM_EPOCHS,
                                                     eta_min=1e-6)

    best_val = float('inf')
    tr_hist, va_hist = [], []

    for epoch in trange(NUM_EPOCHS, desc="Epochs", ascii=True):
        # ── Train ───────────────────────────────────────────────
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred * std + mean, yb)  # undo normalisation in loss
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        tr_loss = running / len(train_loader)
        tr_hist.append(tr_loss)

        # ── Validate every 50 epochs ───────────────────────────
        if epoch % 5 == 0:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                    outv = model(xv)
                    vloss = criterion(outv * std + mean, yv)
                    val_running += vloss.item()
            val_loss = val_running / len(val_loader)
            va_hist.append(val_loss)

            print(f"Epoch {epoch:4d} | train_loss {tr_loss:.6f} | val_loss {val_loss:.6f}")

            # Checkpoint
            ckpt_path = os.path.join(DIR_MODELS, f"2d_vlasov_two-stream_baseline_unet_sr_ep{epoch:04d}.pth")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': tr_loss},
                       ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(DIR_MODELS, "2d_vlasov_two-stream_baseline_unet_sr_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  ↳ new best val_loss ({best_val:.6f}) saved")

            # Persist loss curves
            np.save(os.path.join(DIR_LOGS, 'unet_2d_vlasov_two-stream_train_loss.npy'), np.array(tr_hist))
            np.save(os.path.join(DIR_LOGS, 'unet_2d_vlasov_two-stream_val_loss.npy'),   np.array(va_hist))

# ----------------------------------------------------------------
if __name__ == "__main__":
    train()
