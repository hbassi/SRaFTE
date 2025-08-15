#!/usr/bin/env python
# ===============================================================
#  train_phase1_baseline_FNO.py
#  ---------------------------------------------------------------
#  Phase‑1 super‑resolution baseline using a **plain 2‑D Fourier
#  Neural Operator (FNO)**.  The input is the full 101‑slice
#  coarse history   (B,101,32,32)  and the network predicts the
#  corresponding fine history      (B,101,128,128).
#
#  Aside from swapping the model to FNO, the data pipeline,
#  normalisation, logging, and moment diagnostics mirror the
#  original FUnet script for apples‑to‑apples comparison.
# ===============================================================

import os, math, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
import torch.fft                          as fft
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

CONFIGS = [(8, 8)]               # (m_x, m_y)
SAVE_DIR = "/pscratch/sd/h/hbassi/models"
LOG_DIR  = "./logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ===============================================================
# 1 ▸ Data loader (unchanged)
# ===============================================================
def load_data():
    # inp = np.load(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_coarse_32_fixed_timestep_"
    #               f"mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy")
    # tgt = np.load(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_fine_128_fixed_timestep_"
    #               f"mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy")
    inp = np.load(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_coarse_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy")
    tgt = np.load(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_fine_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy")
    return (torch.tensor(inp, dtype=torch.float32)[:, :200],   # (N,T,Hc,Wc)
            torch.tensor(tgt, dtype=torch.float32)[:, :200])   # (N,T,Hf,Wf)

# ===============================================================
# 2 ▸ Moment helper (unchanged)
# ===============================================================
ve_lims = (-6.0, 6.0)
def compute_moments_torch(f, q=-1.0):
    v  = torch.linspace(*ve_lims, steps=f.size(-1), device=f.device)
    dv = v[1] - v[0]
    rho = q * f.sum(dim=-1)          * dv
    J   = q * (f * v).sum(dim=-1)    * dv
    M2  = q * (f * v**2).sum(dim=-1) * dv
    return rho, J, M2

# ===============================================================
# 3 ▸ 2‑D Spectral Convolution layer (FNO core)
# ===============================================================
class SpectralConv2d(nn.Module):
    """Fourier layer that keeps only the lowest `modes` coefficients."""
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.scale = 1 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            self.scale * torch.randn(in_ch, out_ch, modes1, modes2,
                                     dtype=torch.cfloat)
        )

    def compl_mul2d(self, x, w):
        # x: (B, in_ch, H, W_freq); w: (in_ch,out_ch,m1,m2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = fft.rfft2(x)                   # (B,C,H,W//2+1)
        m1, m2 = self.modes1, self.modes2

        out_ft = torch.zeros(B, self.weight.size(1), H, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2]
        )
        out = fft.irfft2(out_ft, s=x.shape[-2:])
        return out

# ===============================================================
# 4 ▸ Baseline FNO‑SR network
# ===============================================================
class FNO2dSR(nn.Module):
    """
    Plain FNO backbone followed by a bilinear upsample to the fine
    resolution.  No coordinate embedding is used to keep parity
    with the minimal baseline UNet.
    """
    def __init__(self, in_ch=101, width=64, modes1=16, modes2=16,
                 upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.lin0 = nn.Conv2d(in_ch, width, 1)
        self.fno_blocks = nn.ModuleList(
            [nn.ModuleDict({
                "spec": SpectralConv2d(width, width, modes1, modes2),
                "w":    nn.Conv2d(width, width, 1)
            }) for _ in range(3)]
        )
        self.act = nn.GELU()
        self.lin1 = nn.Conv2d(width, in_ch, 1)

    def forward(self, x):
        x = self.lin0(x)
        for blk in self.fno_blocks:
            x = self.act(blk["spec"](x) + blk["w"](x))
        x = self.lin1(x)                       # coarse (Hc,Wc)
        x = nn.functional.interpolate(
            x, scale_factor=self.upscale_factor,
            mode='bilinear', align_corners=False
        )
        return x

# ===============================================================
# 5 ▸ Training loop (mostly unchanged)
# ===============================================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data …")
    X, Y = load_data()                        # (N,T,Hc,Wc) / (N,T,Hf,Wf)
    print("Coarse", X.shape, " Fine", Y.shape)

    upscale_factor = Y.shape[2] // X.shape[2]
    assert Y.shape[3] // X.shape[3] == upscale_factor

    # reshape → channels=101
    X = X.permute(0, 1, 3, 2).contiguous()     # (N,101,32,32) after view
    Y = Y.permute(0, 1, 3, 2).contiguous()     # (N,101,128,128)

    # per‑channel stats
    data_mean = X.mean(dim=(0, 2, 3), keepdim=True)
    data_std  = X.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    torch.save({'data_mean': data_mean, 'data_std': data_std},
               "./data/2d_vlasov_two-stream_phase1_FNO_stats.pt")

    ds = TensorDataset(X, Y)
    tr_ds, va_ds = random_split(ds, [int(0.9 * len(ds)), len(ds) - int(0.9 * len(ds))])
    tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=16)

    model = FNO2dSR(in_ch=200, width=64, modes1=16, modes2=16,
                    upscale_factor=upscale_factor).to(device)
    crit  = nn.L1Loss()
    opt   = optim.AdamW(model.parameters(), lr=5e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-6)

    best_val = float('inf')
    tr_hist, va_hist = [], []

    for ep in trange(5001, desc="Epochs", ascii=True):
        # ───── train ────────────────────────────────────────────
        model.train(); running = 0.0
        m0_max = m1_max = m2_max = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            xb_nrm = (xb - data_mean.to(device)) / data_std.to(device)
            pred = model(xb_nrm) * data_std.to(device) + data_mean.to(device)

            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            running += loss.item()

            rho_o, J_o, M2_o = compute_moments_torch(pred, q=-1.0)
            rho_t, J_t, M2_t = compute_moments_torch(yb,   q=-1.0)
            m0_max = max(m0_max, (rho_o - rho_t).abs().max().item())
            m1_max = max(m1_max, (J_o   - J_t).abs().max().item())
            m2_max = max(m2_max, (M2_o  - M2_t).abs().max().item())

        if ep % 100 == 0:
            tr_loss = running / len(tr_ld)
            tr_hist.append(tr_loss)

            # ───── val ─────────────────────────────────────────
            model.eval(); val_run = 0.0
            vm0 = vm1 = vm2 = 0.0
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    xb_nrm = (xb - data_mean.to(device)) / data_std.to(device)
                    pred = model(xb_nrm) * data_std.to(device) + data_mean.to(device)
                    val_run += crit(pred, yb).item()

                    rho_o, J_o, M2_o = compute_moments_torch(pred, q=-1.0)
                    rho_t, J_t, M2_t = compute_moments_torch(yb,   q=-1.0)
                    vm0 = max(vm0, (rho_o - rho_t).abs().max().item())
                    vm1 = max(vm1, (J_o   - J_t).abs().max().item())
                    vm2 = max(vm2, (M2_o  - M2_t).abs().max().item())

            va_loss = val_run / len(va_ld)
            va_hist.append(va_loss)
            print(f"[{ep:4d}] train={tr_loss:.6e} | val={va_loss:.6e}")
            print(f"     moments train max: ρ {m0_max:.2e}, J {m1_max:.2e}, M2 {m2_max:.2e}")
            print(f"     moments val   max: ρ {vm0:.2e}, J {vm1:.2e}, M2 {vm2:.2e}")

            # ───── checkpoint ─────────────────────────────────
            ckpt = {
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'optim_state_dict': opt.state_dict(), 'val_loss': va_loss
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, f"2d_vlasov_two-stream_FNO_phase1_ep{ep:04d}.pth"))
            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR, "2d_vlasov_two-stream_FNO_phase1_best.pth"))
                print("   ↳ saved new best")

            # log histories
            np.save(os.path.join(LOG_DIR, "2d_vlasov_two-stream_FNO_train_loss.npy"), np.array(tr_hist))
            np.save(os.path.join(LOG_DIR, "2d_vlasov_two-stream_FNO_val_loss.npy"),   np.array(va_hist))

# ----------------------------------------------------------------
if __name__ == "__main__":
    train_model()
