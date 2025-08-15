
# ───────────────────────────── imports ──────────────────────────
import argparse, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import trange
import models
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

# ─────────────────────── 1 ▸  In‑memory dataset  ────────────────
class NSDataset(Dataset):
    def __init__(self, coarse_path, fine_path, T=None):
        #  ↓↓↓  NO mmap_mode  ↓↓↓
        self.X = np.load(coarse_path)              # (B,Tc,Hc,Wc)
        self.Y = np.load(fine_path)                # (B,Tf,Hf,Wf)
        assert self.X.shape[:2] == self.Y.shape[:2], "B,T mismatch!"
        self.B, self.T = self.X.shape[:2]
        if T is not None:
            self.T = min(self.T, T)
            self.X = self.X[:, :self.T]
            self.Y = self.Y[:, :self.T]

    def __len__(self):  return self.B

    def __getitem__(self, idx):
        # avoid extra copies; torch can read directly from the NumPy view
        x = torch.from_numpy(self.X[idx]).float()  # (T,Hc,Wc)
        y = torch.from_numpy(self.Y[idx]).float()  # (T,Hf,Wf)
        return x, y


# ──────────────── 2 ▸  Utility: mean / std  ─────────────────────
def dataset_stats(loader):
    cnt, mean, M2 = 0, None, None
    for x, _ in loader:
        bsz = x.size(0)
        x = x.view(bsz, x.size(1), -1)             # (B,T,H*W)
        if mean is None:
            mean = torch.zeros_like(x[0, :, 0])
            M2   = torch.zeros_like(mean)
        cnt_new    = bsz * x.size(-1)
        mean_batch = x.mean(dim=(0, 2))
        var_batch  = x.var(dim=(0, 2), unbiased=False)
        if cnt == 0:
            mean, M2 = mean_batch, var_batch * cnt_new
        else:
            delta = mean_batch - mean
            tot   = cnt + cnt_new
            mean += delta * cnt_new / tot
            M2   += var_batch * cnt_new + delta**2 * cnt * cnt_new / tot
        cnt += cnt_new
    return mean.view(1, -1, 1, 1).float(), (M2 / cnt).sqrt().clamp_min(1e-8).view(1, -1, 1, 1).float()


# ─────────────── 3 ▸  Model factory  ────────────────────────────
def build_model(name, in_ch):
    name = name.lower()
    if   name == "funet":
        return models.SuperResUNet(in_channels=in_ch)
    elif name == "unet":
        return models.UNetSR(in_ch=in_ch, upscale_factor=4)
    elif name == "edsr":
        return models.EDSR(
            in_ch=in_ch, n_feats=128, n_res_blocks=16, upscale_factor=4,
            mean=np.zeros(in_ch, dtype=np.float32), std=np.ones(in_ch, dtype=np.float32)
        )
    elif name == "fno":
        return models.FNO2dSR(in_ch=in_ch, modes1=16, modes2=16, upscale_factor=4)
    else:
        raise ValueError(f"Unknown model '{name}'.")


# ─────────────── 4 ▸  Training loop  ────────────────────────────
def train(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    ds = NSDataset(cfg.coarse, cfg.fine, cfg.time_horizon)
    tr_len = int(len(ds) * 0.9)
    va_len = len(ds) - tr_len
    tr_ds, va_ds = random_split(ds, [tr_len, va_len],
                                generator=torch.Generator().manual_seed(0))

    # <<< num_workers=0 so tensors stay in one process >>>
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    mean, std = dataset_stats(tr_ld)
    mean = mean.float()
    std = std.float()
    torch.save({"mean": mean, "std": std}, f"stats_{cfg.tag}.pt")

    model = build_model(cfg.model, ds.T).to(dev)
    opt   = optim.AdamW(model.parameters(), lr=cfg.lr)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    loss_fn = nn.L1Loss()
    best = float("inf")
    save_dir = Path(cfg.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    train_log, val_log = [], []

    for ep in trange(cfg.epochs + 1):
        # ── training ────────────────────────────────────────────
        model.train(); tr_loss = 0.0
        for x, y in tr_ld:
            x, y  = x.to(dev).float(), y.to(dev).float()
            x_hat = (x - mean.to(dev)) / std.to(dev)
            pred  = model(x_hat)
            pred  = pred * std.to(dev) + mean.to(dev)
            #x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            #pred = model(((x - mean.double().to(dev)).double()) / std.double().to(dev))
            #pred = pred * std.double().to(dev) + mean.double().to(dev)
            loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld); train_log.append(tr_loss)

        # ── validation / checkpoint ────────────────────────────
        if ep % cfg.val_every == 0:
            model.eval(); va_loss = 0.0
            with torch.no_grad():
                for x, y in va_ld:
                    x, y  = x.to(dev).float(), y.to(dev).float()
                    x_hat = (x - mean.to(dev)) / std.to(dev)
                    pred  = model(x_hat)
                    pred  = pred * std.to(dev) + mean.to(dev)
                    va_loss += loss_fn(pred, y).item()
            va_loss /= len(va_ld); val_log.append((ep, va_loss))
            print(f"E{ep:04d}  train {tr_loss:.5e} | val {va_loss:.5e}")

            ckpt = {"epoch": ep, "model": model.state_dict(),
                    "opt": opt.state_dict(), "val": va_loss}
            torch.save(ckpt, save_dir / f"{cfg.model}_{cfg.tag}_ep{ep}.pth")
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(),
                           save_dir / f"best_{cfg.model}_{cfg.tag}.pth")
        sch.step()

    np.save(save_dir / f"{cfg.model}_{cfg.tag}_train.npy", np.array(train_log))
    np.save(save_dir / f"{cfg.model}_{cfg.tag}_val.npy",   np.array(val_log))


# ─────────────── 5 ▸  CLI  ─────────────────────────────────────
def cli():
    p = argparse.ArgumentParser("Phase‑1 operator trainer (RAM mode)")
    p.add_argument("--model",  required=True,
                   choices=["funet", "unet", "edsr", "fno"])
    p.add_argument("--coarse", default="/pscratch/sd/h/hbassi/"
                                      "NavierStokes_coarse_128_nu0.0001_k7.5_training_data.npy")
    p.add_argument("--fine",   default="/pscratch/sd/h/hbassi/"
                                      "NavierStokes_fine_512_nu0.0001_k7.5_training_data.npy")
    p.add_argument("--epochs",      type=int, default=5000)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--val_every",   type=int, default=100)
    p.add_argument("--time_horizon",type=int, default=100,
                   help="truncate T if the file contains more steps than needed")
    p.add_argument("--save_dir",    default="/pscratch/sd/h/hbassi/models")
    p.add_argument("--tag",         default="128to512_nu1e-4_k7.5")
    return p.parse_args()

if __name__ == "__main__":
    train(cli())
