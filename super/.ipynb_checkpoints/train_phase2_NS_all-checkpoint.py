# ───────────────────────── imports ────────────────────────────
import argparse, math, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import trange
import models

#torch.set_float32_matmul_precision("high")
CKPT_TPL  = "/pscratch/sd/h/hbassi/models/best_{model}_{tag}.pth"
TAG       = "nu1e-4_k7.5" 
# ───────── data + PDE parameters ──────────────────────────────
NU         = 1e-4
K_CUTOFF   = 7.5
DT_COARSE  = 0.01          # Euler step on the coarse grid
UPSCALE    = 4             # fine 256 → coarse 64
T_CONTEXT  = 100           # we still feed 100 channels to the net

# ───────── auxiliary functions ────────────────────────────────
def projection(fine, factor=UPSCALE):
    """Simple strided subsampling channel‑wise."""
    return fine[..., ::factor, ::factor]

def coarse_step_ns(u, dt, dx, dy, nu):
    """
    One Euler step of 2‑D NS vorticity on the coarse grid.
    u : (B, C, H, W) real
    """
    B, C, H, W = u.shape
    k = torch.fft.fftfreq(H, d=dx, device=u.device) * 2 * math.pi
    KX, KY = torch.meshgrid(k, k, indexing='ij')
    k2  = KX**2 + KY**2
    k2[0,0] = 1e-10
    KX, KY, k2 = [t.unsqueeze(0).unsqueeze(0) for t in (KX, KY, k2)]

    ω_hat = torch.fft.fft2(u)
    ψ_hat = -ω_hat / k2
    u_x   =  torch.fft.ifft2(1j*KY*ψ_hat).real
    u_y   = -torch.fft.ifft2(1j*KX*ψ_hat).real
    dωdx  =  torch.fft.ifft2(1j*KX*ω_hat).real
    dωdy  =  torch.fft.ifft2(1j*KY*ω_hat).real
    adv   = u_x*dωdx + u_y*dωdy
    lapω  = torch.fft.ifft2(-k2*ω_hat).real

    # simple diagonal forcing (same as Phase‑1 script)
    x = torch.linspace(0,1,H, device=u.device)
    y = torch.linspace(0,1,W, device=u.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    forcing = 0.025*(torch.sin(2*math.pi*(X+Y)) + torch.cos(2*math.pi*(X+Y)))
    forcing = forcing.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    return u + dt*(-adv + nu*lapω + forcing)

# ───────── dataset loader ─────────────────────────────────────
class FineTraj(Dataset):
    def __init__(self, path, T=T_CONTEXT):
        data = np.load(path)                     # (B,T+1, C, H, W)
        self.inputs  = torch.tensor(data[:500, :T]  , dtype=torch.float32)
        self.targets = torch.tensor(data[:500, 1:T+1], dtype=torch.float32)

    def __len__(self):          return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

# ───────── model factory ─────────────────────────────────────
def build_model(kind, in_ch):
    if   kind=="funet": return models.SuperResUNet(in_channels=in_ch, final_scale=UPSCALE)
    elif kind=="unet":  return models.UNetSR(in_ch=in_ch, upscale_factor=UPSCALE)
    elif kind=="edsr":  return models.EDSR(in_ch=in_ch, n_feats=128, n_res_blocks=16,
                                           upscale_factor=UPSCALE,
                                           mean=np.zeros(in_ch, dtype=np.float32), std=np.ones(in_ch, dtype=np.float32))
    elif kind=="fno":   return models.FNO2dSR(in_ch=in_ch, modes1=16, modes2=16,
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

    # stats from Phase‑1 mean/std of coarse inputs (reuse fine mean/std for simplicity)
    mean = ds.inputs.mean((0,2,3), keepdim=True).to(dev)
    std  = ds.inputs.std ((0,2,3), keepdim=True).clamp_min(1e-8).to(dev)
    torch.save({"mean": mean, "std": std}, f"./data/phase2_{cfg.model}_stats_{cfg.tag}.pt")

    # model – start from best Phase‑1 weights
    model = build_model(cfg.model, in_ch=T_CONTEXT).to(dev)
    ph1_path = CKPT_TPL.format(model=cfg.model, tag=cfg.tag)
    model.load_state_dict(torch.load(ph1_path, map_location="cpu"))
    print(f"Loaded Phase‑1 weights from {ph1_path}")

    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    loss_fn = nn.L1Loss()
    best = float('inf'); save_dir = Path(cfg.save_dir); save_dir.mkdir(exist_ok=True)

    # coarse‑grid spacing
    Hf = ds.inputs.shape[-1]; Hc = Hf//UPSCALE
    dx = dy = 1.0 / Hc

    for ep in trange(cfg.epochs+1):
        # —— train ——
        model.train(); tr = 0.
        for fin, tgt in tr_ld:
            fin, tgt = fin.to(dev), tgt.to(dev)
            coarse   = projection(fin[:], UPSCALE)             # last fine frame → coarse
            coarse_tp= coarse_step_ns(coarse, DT_COARSE, dx, dy, NU)
            pred     = model(((coarse_tp-mean)/std).float())*std + mean
            loss     = loss_fn(pred, tgt[:])
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item()
        tr /= len(tr_ld)

        # —— validate ——
        if ep % cfg.val_every == 0:
            model.eval(); va = 0.
            with torch.no_grad():
                for fin, tgt in va_ld:
                    fin, tgt = fin.to(dev), tgt.to(dev)
                    coarse   = projection(fin[:], UPSCALE)
                    coarse_tp= coarse_step_ns(coarse, DT_COARSE, dx, dy, NU)
                    pred     = model(((coarse_tp-mean)/std).float())*std + mean
                    va      += loss_fn(pred, tgt[:]).item()
            va /= len(va_ld)
            print(f"E{ep:04d}  train {tr:.4e} | val {va:.4e}")

            torch.save({"epoch":ep,"model":model.state_dict(),
                        "opt":opt.state_dict(),"val":va},
                       save_dir/f"NS_{cfg.model}_{cfg.tag}_phase2_ep{ep}.pth")
            if va<best:
                best=va; torch.save(model.state_dict(),
                                    save_dir/f"NS_best_{cfg.model}_{cfg.tag}_phase2.pth")
        sch.step()

# ───────── CLI ───────────────────────────────────────────────
def cli():
    p=argparse.ArgumentParser("Phase‑2 Navier–Stokes trainer")
    p.add_argument("--model", choices=["funet","unet","edsr","fno"], required=True)
    p.add_argument("--fine", default="/pscratch/sd/h/hbassi/"
                                 "NavierStokes_fine_256_nu0.0001_k7.5.npy")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--tag", default="nu1e-4_k7.5")          # Phase‑1 tag
    p.add_argument("--save_dir", default="/pscratch/sd/h/hbassi/models")
    return p.parse_args()

if __name__ == "__main__":
    train(cli())