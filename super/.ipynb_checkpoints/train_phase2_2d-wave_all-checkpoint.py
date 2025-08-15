

# ───────────────────────── imports ────────────────────────────
import argparse, math, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import trange
import models

torch.set_float32_matmul_precision("high")

# ───────── configuration ─────────────────────────────────────
# Wave‑equation parameters (must match data generation)
c          = 0.5                     # wave speed
dt         = 0.01
Lx = Ly    = 1.0

# Upsampling
UPSCALE    = 8                       # fine 128 → coarse 32
Nx_fine    = 128                     # (only for comments)
Nx_coarse  = Nx_fine // UPSCALE
dx_c       = Lx / Nx_coarse
dy_c       = Ly / Nx_coarse
c2dt2      = (c * dt) ** 2

T_CONTEXT  = 100                     # same 100‑frame context
CKPT_TPL   = "/pscratch/sd/h/hbassi/models/best_{model}_{tag}.pth"
TAG        = "2d-wave_high_freq_sf=8"               # Phase‑1 tag (adjust if needed)

# ───────── projection & coarse solver ────────────────────────
def projection_operator(fine, factor=UPSCALE):
    return fine[..., ::factor, ::factor]

def coarse_time_step_wave(coarse_prev, coarse_curr):
    B, T, Hc, Wc = coarse_curr.shape
    M = B * T                     # flatten batch and channel dims
    u_nm1 = coarse_prev.reshape(M, Hc, Wc)
    u_n   = coarse_curr.reshape(M, Hc, Wc)

    lap = (
        (torch.roll(u_n, +1, 1) + torch.roll(u_n, -1, 1) - 2*u_n) / dx_c**2 +
        (torch.roll(u_n, +1, 2) + torch.roll(u_n, -1, 2) - 2*u_n) / dy_c**2
    )
    u_np1 = 2*u_n - u_nm1 + c2dt2 * lap
    return u_np1.reshape(B, T, Hc, Wc)

# ───────── dataset loader ────────────────────────────────────
class FineTraj(Dataset):
    def __init__(self, path, T=T_CONTEXT):
        data = np.load(path)                     # (B, T+1, H, W)
        self.inputs  = torch.tensor(data[:, :T]    , dtype=torch.float32)
        self.targets = torch.tensor(data[:, 1:T+1] , dtype=torch.float32)

    def __len__(self):          return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

# ───────── model factory (unchanged) ─────────────────────────
def build_model(kind, in_ch):
    if   kind=="funet": return models.SuperResUNet(in_channels=in_ch, final_scale=UPSCALE)
    elif kind=="unet":  return models.UNetSR(in_ch=in_ch, upscale_factor=UPSCALE)
    elif kind=="edsr":  return models.EDSR(in_ch=in_ch, n_feats=128, n_res_blocks=16,
                                           upscale_factor=UPSCALE,
                                           mean=np.zeros(in_ch, dtype=np.float32), std=np.ones(in_ch, dtype=np.float32))
    elif kind=="fno":   return models.FNO2dSR(in_ch=in_ch, modes1=8, modes2=8,
                                              upscale_factor=UPSCALE)
    else: raise ValueError(kind)

# ───────── training loop ─────────────────────────────────────
def train(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds   = FineTraj(cfg.fine)
    tr_ds, va_ds = random_split(ds, [int(0.9*len(ds)), len(ds)-int(0.9*len(ds))],
                                generator=torch.Generator().manual_seed(0))
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size)

    # normalisation (reuse fine‑field stats)
    mean = ds.inputs.mean((0,2,3), keepdim=True).to(dev)
    std  = ds.inputs.std ((0,2,3), keepdim=True).clamp_min(1e-8).to(dev)
    torch.save({"mean": mean, "std": std}, f"./data/phase2_{cfg.model}_stats_{cfg.tag}.pt")

    # model (warm‑start from Phase‑1)
    model = build_model(cfg.model, in_ch=T_CONTEXT).to(dev)
    ph1_path = CKPT_TPL.format(model=cfg.model, tag=cfg.tag)
    model.load_state_dict(torch.load(ph1_path, map_location="cpu"))
    print(f"Loaded Phase‑1 weights from {ph1_path}")

    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    loss_fn = nn.L1Loss()
    best = float('inf'); save_dir = Path(cfg.save_dir); save_dir.mkdir(exist_ok=True)

    for ep in trange(cfg.epochs+1):
        # —— train ——
        model.train(); tr = 0.
        for fin, tgt in tr_ld:
            fin, tgt = fin.to(dev), tgt.to(dev)

            coarse   = projection_operator(fin, UPSCALE)                 # (B,T,Hc,Wc)

            # construct u_{n-1}, u_n for every time channel
            coarse_prev = torch.cat([coarse[:, :1], coarse[:, :-1]], dim=1)  # duplicate first frame
            coarse_tp   = coarse_time_step_wave(coarse_prev, coarse)         # u_{n+1}

            pred  = model(((coarse_tp-mean)/std).float()) * std + mean
            loss  = loss_fn(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item()
        tr /= len(tr_ld)

        # —— validate ——
        if ep % cfg.val_every == 0:
            model.eval(); va = 0.
            with torch.no_grad():
                for fin, tgt in va_ld:
                    fin, tgt = fin.to(dev), tgt.to(dev)
                    coarse   = projection_operator(fin, UPSCALE)
                    coarse_prev = torch.cat([coarse[:, :1], coarse[:, :-1]], dim=1)
                    coarse_tp   = coarse_time_step_wave(coarse_prev, coarse)
                    pred = model(((coarse_tp-mean)/std).float()) * std + mean
                    va  += loss_fn(pred, tgt).item()
            va /= len(va_ld)
            print(f"E{ep:04d}  train {tr:.4e} | val {va:.4e}")

            torch.save({"epoch":ep,"model":model.state_dict(),
                        "opt":opt.state_dict(),"val":va},
                       save_dir/f"wave_{cfg.model}_phase2_ep{ep}.pth")
            if va < best:
                best = va
                torch.save(model.state_dict(),
                           save_dir/f"wave_best_{cfg.model}_phase2.pth")
        sch.step()

# ───────── CLI ───────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser("Phase‑2 2‑D wave‑equation trainer")
    p.add_argument("--model", choices=["funet","unet","edsr","fno"], required=True)
    p.add_argument("--fine", default="/pscratch/sd/h/hbassi/"
                                   "wave_dataset_multi_sf_modes=10_kmax=7/u_fine.npy")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--tag", default=TAG)              # Phase‑1 tag
    p.add_argument("--save_dir", default="/pscratch/sd/h/hbassi/models")
    return p.parse_args()

if __name__ == "__main__":
    train(cli())
