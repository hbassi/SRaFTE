# ───────────── imports ─────────────
import argparse, math, os, logging
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

import models                   

torch.set_float32_matmul_precision("high")

# ═══════════════════════════════════════════════════════════════
# 1 ▸ Heat-equation coarse-grid propagator
#    (spectral integrating-factor – same as your Funet script)
# ═══════════════════════════════════════════════════════════════
def make_coarse_solver(Nc: int, nu: float, dt: float, device):
    kx_1d = torch.fft.fftfreq(Nc, d=1.0 / Nc) * (2.0 * math.pi)
    ky_1d = kx_1d.clone()
    kx, ky = torch.meshgrid(kx_1d, ky_1d, indexing='ij')
    k2 = (kx ** 2 + ky ** 2).to(device)
    k2[0, 0] = 1e-14                       # avoid /0
    exp_fac = torch.exp(-nu * k2 * dt)     # (Nc,Nc)

    # forcing term  f(x,y) = sin(2πx) sin(2πy)
    xs = (torch.arange(Nc, device=device, dtype=torch.float64) + 0.5) / Nc
    ys = xs.clone()
    Xc, Yc = torch.meshgrid(xs, ys, indexing='ij')
    f_xy = torch.sin(2 * math.pi * Xc) * torch.sin(2 * math.pi * Yc)
    f_hat = torch.fft.fft2(f_xy)
    forcing_term = ((1.0 - exp_fac) * f_hat) / (nu * k2)

    def step(u_c):  # u_c: (B,1,Nc,Nc) float64/32
        u_hat = torch.fft.fft2(u_c.squeeze(1).to(torch.float64))
        u_hat = u_hat * exp_fac + forcing_term
        u_next = torch.real(torch.fft.ifft2(u_hat))
        return u_next.to(torch.float32)         # (B,1,Nc,Nc)

    return step


# ═══════════════════════════════════════════════════════════════
# 2 ▸ Data loading helpers
# ═══════════════════════════════════════════════════════════════
def load_fine_sequences(path, T=100):
    data = np.load(path)                                     # (B_all, T_total, H, W)
    data = data[:, :T + 1]                                  
    inputs  = torch.tensor(data[:,  :T ], dtype=torch.float32)  # (B,T,H,W)
    targets = torch.tensor(data[:, 1:T+1], dtype=torch.float32)
    return inputs, targets


def compute_norm_stats(x):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std  = x.std (dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    return mean, std


# ═══════════════════════════════════════════════════════════════
# 3 ▸ Model factory  (re-uses Phase-1 architectures)
# ═══════════════════════════════════════════════════════════════
def build_model(kind: str, in_ch: int, upscale: int):
    kind = kind.lower()
    if   kind == "funet":
        return models.SuperResUNet(in_channels=in_ch, final_scale=upscale)
    elif kind == "unet":
        return models.UNetSR (in_ch=in_ch, upscale_factor=upscale)
    elif kind == "edsr":
        return models.EDSR   (in_ch=in_ch, n_feats=128, n_res_blocks=16,
                              upscale_factor=upscale,
                              mean=np.zeros(in_ch, dtype=np.float32),
                              std =np.ones (in_ch, dtype=np.float32))
    elif kind == "fno":
        return models.FNO2dSR(in_ch=in_ch, modes1=16, modes2=16,
                              upscale_factor=upscale)
    else:
        raise ValueError(f"unknown model '{kind}'")


# ═══════════════════════════════════════════════════════════════
# 4 ▸ Training loop
# ═══════════════════════════════════════════════════════════════
def train(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    inputs, targets = load_fine_sequences(cfg.fine_data, cfg.time_window)
    B, T, Hf, Wf = inputs.shape
    upscale = Hf // cfg.Nc
    mean, std = compute_norm_stats(inputs)
    torch.save({"mean": mean, "std": std},
               Path(cfg.ckpt_dir) / f"phase2_stats_{cfg.model}_{cfg.tag}.pt")

    ds = TensorDataset(inputs, targets)
    tr_len = int(0.9 * len(ds))
    va_len = len(ds) - tr_len
    tr_ds, va_ds = random_split(ds, [tr_len, va_len],
                                generator=torch.Generator().manual_seed(0))
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    # ---------- physics ----------
    coarse_step = make_coarse_solver(cfg.Nc, cfg.nu, cfg.dt, dev)

    # ---------- model ----------
    model = build_model(cfg.model, in_ch=T, upscale=upscale).to(dev)
    # load Phase-1 weights
    phase1_ckpt = Path(cfg.phase1_dir) / f"best_{cfg.model}_{cfg.phase1_tag}.pth"
    model.load_state_dict(torch.load(phase1_ckpt, map_location=dev))

    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                               eta_min=1e-6)
    loss_fn = nn.L1Loss()
    best = float("inf")

    for ep in trange(cfg.epochs):
        # ── train ───────────────────────────────────────────
        model.train(); tr_loss = 0.0
        for xb, yb in tr_ld:                                   # (B,T,Hf,Wf)
            xb, yb = xb.to(dev), yb.to(dev)                    # fine grids
            opt.zero_grad()

            fine_curr = xb                                     # (B,T,Hf,Wf)
            coarse_curr = fine_curr[..., ::upscale, ::upscale] # (B,T,Nc,Nc)
            coarse_tp1  = coarse_step(coarse_curr)
                                               # (B,T,Nc,Nc)

            norm_in = (coarse_tp1 - mean.to(dev)) / std.to(dev)
            pred_tp1 = model(norm_in) * std.to(dev) + mean.to(dev)

            loss = loss_fn(pred_tp1, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── validate ───────────────────────────────────────
        if ep % cfg.val_every == 0:
            model.eval(); va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb, yb = xb.to(dev), yb.to(dev)

                    coarse_tp1 = coarse_step(
                        xb[..., ::upscale, ::upscale]
                    )
                    norm_in = (coarse_tp1 - mean.to(dev)) / std.to(dev)
                    pred_tp1 = model(norm_in) * std.to(dev) + mean.to(dev)
                    va_loss += loss_fn(pred_tp1, yb).item()
            va_loss /= len(va_ld)

            logging.info(f"E{ep:04d}  train {tr_loss:.6e} | val {va_loss:.6e}")
            # ---- save checkpoints ----
            torch.save({"epoch": ep,
                        "model": model.state_dict(),
                        "opt":   opt.state_dict(),
                        "val":   va_loss},
                       Path(cfg.ckpt_dir) / f"{cfg.model}_{cfg.tag}_ep{ep}.pth")
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(),
                           Path(cfg.ckpt_dir) / f"best_{cfg.model}_{cfg.tag}.pth")


# ═══════════════════════════════════════════════════════════════
# 5 ▸ CLI
# ═══════════════════════════════════════════════════════════════
def cli():
    p = argparse.ArgumentParser("Phase-2 trainer – 2-D heat equation")
    p.add_argument("--model", required=True,
                   choices=["funet", "unet", "edsr", "fno"])
    p.add_argument("--fine_data",
                   default="/pscratch/sd/h/hbassi/dataset/"
                           "2d_heat_eqn_fine_all_smooth_gauss_1k.npy")
    p.add_argument("--phase1_dir", default="/pscratch/sd/h/hbassi/models")
    p.add_argument("--phase1_tag", default="heat_nuNA_gauss1k",
                   help="tag used when saving Phase-1 checkpoints")
    p.add_argument("--ckpt_dir",  default="/pscratch/sd/h/hbassi/models")
    p.add_argument("--tag",       default="heat_phase2")
    # physics / grid
    p.add_argument("--Nc", type=int, default=32,
                   help="coarse grid size (matches Phase-1 training)")
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.01)
    # training
    p.add_argument("--epochs",     type=int,   default=2500)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--val_every",  type=int,   default=100)
    p.add_argument("--time_window",type=int,   default=100,
                   help="number of time steps used as channels")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ------- logging -------
    args = cli()
    logging.basicConfig(
        filename=f"{args.ckpt_dir}/train_{args.model}_{args.tag}.log",
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s"
    )
    logging.info("Starting Phase-2 training")
    train(args)
