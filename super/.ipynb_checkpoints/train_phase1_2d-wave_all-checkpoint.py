# ───────────────────────── imports ────────────────────────────
import argparse, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import trange
import models                      

torch.set_float32_matmul_precision("high")

# ────────────────────── 1 ▸ Data set ──────────────────────────
class HeatDataset(Dataset):
    def __init__(self, coarse_path, fine_path, T=None):
        self.X = np.load(coarse_path)     # (B,Tc,Hc,Wc)
        self.Y = np.load(fine_path)       # (B,Tf,Hf,Wf)
        assert self.X.shape[:2] == self.Y.shape[:2], "B,T mismatch!"
        self.B, self.T = self.X.shape[:2]
        if T is not None:
            self.T = min(self.T, T)
            self.X = self.X[:, :self.T]
            self.Y = self.Y[:, :self.T]

    def __len__(self):  return self.B
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y


# ────────────────────── 2 ▸ Stats util ────────────────────────
def dataset_stats(loader):
    cnt, mean, M2 = 0, None, None
    for x, _ in loader:
        bsz = x.size(0)
        x = x.reshape(bsz, x.size(1), -1)        # (B,T,H*W)
        if mean is None:
            mean = torch.zeros_like(x[0, :, 0])
            M2   = torch.zeros_like(mean)
        cnt_new = bsz * x.size(-1)
        μb      = x.mean(dim=(0, 2))
        σ2b     = x.var (dim=(0, 2), unbiased=False)
        if cnt == 0:
            mean, M2 = μb, σ2b * cnt_new
        else:
            δ   = μb - mean
            tot = cnt + cnt_new
            mean += δ * cnt_new / tot
            M2   += σ2b * cnt_new + δ**2 * cnt * cnt_new / tot
        cnt += cnt_new
    return mean.view(1, -1, 1, 1), (M2 / cnt).sqrt().clamp_min(1e-8).view(1, -1, 1, 1)


# ───────────────────── 3 ▸ Model factory ──────────────────────
def build_model(kind, in_ch, upscale):
    kind = kind.lower()
    if   kind == "funet":
        return models.SuperResUNet(in_channels=in_ch, final_scale=upscale)
    elif kind == "unet":
        return models.UNetSR (in_ch=in_ch,      upscale_factor=upscale)
    elif kind == "edsr":
        return models.EDSR   (in_ch=in_ch, n_feats=128, n_res_blocks=16,
                              upscale_factor=upscale,
                              mean=np.zeros(in_ch, dtype=np.float32), std=np.ones(in_ch, dtype=np.float32))
    elif kind == "fno":
        return models.FNO2dSR(in_ch=in_ch, modes1=8, modes2=8,
                              upscale_factor=upscale)
    else:
        raise ValueError(f"unknown model '{kind}'")

# ───────────────────── 4 ▸ Training loop ──────────────────────
def train(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = HeatDataset(cfg.coarse, cfg.fine, cfg.time_horizon)
    upscale = ds.Y.shape[-1] // ds.X.shape[-1]
    print(f"Loaded {len(ds)} samples  |  upscale ×{upscale}")

    tr_len = int(len(ds) * 0.9)
    va_len = len(ds) - tr_len
    tr_ds, va_ds = random_split(ds, [tr_len, va_len],
                                generator=torch.Generator().manual_seed(0))
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    mean, std = dataset_stats(tr_ld)
    torch.save({"mean": mean, "std": std},
               Path(cfg.save_dir) / f"stats_{cfg.model}_{cfg.tag}.pt")

    model = build_model(cfg.model, ds.T, upscale).to(dev)
    opt   = optim.AdamW(model.parameters(), lr=cfg.lr)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                 eta_min=1e-6)
    loss_fn = nn.L1Loss()
    best = float("inf")
    save_dir = Path(cfg.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    tr_hist, va_hist = [], []

    for ep in trange(cfg.epochs + 1):
        # ── train ───────────────────────────────────────────
        model.train(); tr_loss = 0.0
        for x, y in tr_ld:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            pred = model((x - mean.to(dev)) / std.to(dev))
            pred = pred * std.to(dev) + mean.to(dev)
            loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld); tr_hist.append(tr_loss)

        # ── validation / checkpoint ──────────────────────
        if ep % cfg.val_every == 0:
            model.eval(); va_loss = 0.0
            with torch.no_grad():
                for x, y in va_ld:
                    x, y = x.to(dev), y.to(dev)
                    pred = model((x - mean.to(dev)) / std.to(dev))
                    pred = pred * std.to(dev) + mean.to(dev)
                    va_loss += loss_fn(pred, y).item()
            va_loss /= len(va_ld); va_hist.append((ep, va_loss))
            print(f"E{ep:04d}  train {tr_loss:.5e} | val {va_loss:.5e}")

            ckpt = {"epoch": ep, "model": model.state_dict(),
                    "opt": opt.state_dict(), "val": va_loss}
            torch.save(ckpt, save_dir / f"{cfg.model}_{cfg.tag}_ep{ep}.pth")
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(),
                           save_dir / f"best_{cfg.model}_{cfg.tag}.pth")
        sch.step()

    np.save(save_dir / f"{cfg.model}_{cfg.tag}_train.npy", np.array(tr_hist))
    np.save(save_dir / f"{cfg.model}_{cfg.tag}_val.npy",   np.array(va_hist))


# ───────────────────── 5 ▸ CLI ────────────────────────────────
def cli():
    p = argparse.ArgumentParser("Phase‑1 trainer – 2‑D heat equation")
    p.add_argument("--model", required=True,
                   choices=["funet", "unet", "edsr", "fno"])
    p.add_argument("--coarse", default="/pscratch/sd/h/hbassi/"
                                      f"wave_dataset_multi_sf_modes=10_kmax=7/u_coarse_sf={8}.npy")
    p.add_argument("--fine",   default="/pscratch/sd/h/hbassi/"
                                      "wave_dataset_multi_sf_modes=10_kmax=7/u_fine.npy")
    p.add_argument("--epochs",      type=int,   default=5000)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--val_every",   type=int,   default=100)
    p.add_argument("--time_horizon",type=int,   default=100,
                   help="truncate T if the files contain more steps")
    p.add_argument("--save_dir",    default="/pscratch/sd/h/hbassi/models")
    p.add_argument("--tag",         default="2d-wave_high_freq_sf=8")
    return p.parse_args()

if __name__ == "__main__":
    train(cli())
