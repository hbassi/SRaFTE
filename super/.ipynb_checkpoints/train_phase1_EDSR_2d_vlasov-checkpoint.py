#!/usr/bin/env python
# ===============================================================
#  train_phase1_baseline_EDSR.py
#  ---------------------------------------------------------------
#  Phase‑1 super‑resolution baseline using a 2‑D EDSR network.
#  Input  : full 101‑slice coarse history (B,101,32,32)
#  Output : full 101‑slice fine   history (B,101,128,128)
#  The pipeline, normalisation, logging, and diagnostics mirror
#  the original FUnet script for fair comparison.
# ===============================================================

import os, math, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

CONFIGS   = [(8, 8)]          # (m_x, m_y)
SAVE_DIR  = "/pscratch/sd/h/hbassi/models"
LOG_DIR   = "./logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ===============================================================
# 1 ▸ Data loader
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
# 2 ▸ Moment helper
# ===============================================================
VE_LIMS = (-6.0, 6.0)
def compute_moments_torch(f, q=-1.0):
    v  = torch.linspace(*VE_LIMS, steps=f.size(-1), device=f.device)
    dv = v[1] - v[0]
    rho = q * f.sum(dim=-1)          * dv
    J   = q * (f * v).sum(dim=-1)    * dv
    M2  = q * (f * v**2).sum(dim=-1) * dv
    return rho, J, M2

# ===============================================================
# 3 ▸ EDSR components
# ===============================================================
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias)

class ShiftMean(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        c = mean.shape[0]
        self.register_buffer('mean', torch.tensor(mean).view(1, c, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, c, 1, 1))
    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean) / self.std
        if mode == 'add':
            return x * self.std + self.mean
        raise NotImplementedError

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                 act=nn.ReLU(True), res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            act,
            conv(n_feats, n_feats, kernel_size)
        )
        self.res_scale = res_scale
    def forward(self, x):
        return x + self.body(x) * self.res_scale

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats):
        m = []
        if (scale & (scale - 1)) == 0:        # scale = 2^n
            for _ in range(int(math.log2(scale))):
                m += [conv(n_feats, 4 * n_feats, 3), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [conv(n_feats, 9 * n_feats, 3), nn.PixelShuffle(3)]
        else:
            raise NotImplementedError
        super().__init__(*m)

class EDSR(nn.Module):
    def __init__(self, in_ch, n_feats, n_res_blocks,
                 upscale_factor, mean, std, conv=default_conv):
        super().__init__()
        self.shift = ShiftMean(mean, std)
        m_head = [conv(in_ch, n_feats, 3)]
        m_body = [ResBlock(conv, n_feats, 3) for _ in range(n_res_blocks)]
        m_body += [conv(n_feats, n_feats, 3)]
        m_tail = [Upsampler(conv, upscale_factor, n_feats),
                  conv(n_feats, in_ch, 3)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    def forward(self, x):
        x = self.shift(x, 'sub')
        x = self.head(x)
        res = self.body(x) + x
        x = self.tail(res)
        x = self.shift(x, 'add')
        return x

# ===============================================================
# 4 ▸ Training loop
# ===============================================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data …")
    X, Y = load_data()                          # (N,T,Hc,Wc)/(N,T,Hf,Wf)
    print("Coarse", X.shape, " Fine", Y.shape)

    upscale_factor = Y.shape[2] // X.shape[2]
    assert Y.shape[3] // X.shape[3] == upscale_factor

    # reshape to channel‑first tensors (N,101,32,32) and (N,101,128,128)
    X = X.permute(0, 1, 3, 2).contiguous()
    Y = Y.permute(0, 1, 3, 2).contiguous()

    # per‑channel statistics
    mean = X.mean(dim=(0, 2, 3)).cpu().numpy()
    std  = X.std(dim=(0, 2, 3)).clamp_min(1e-8).cpu().numpy()
    torch.save({'mean': mean, 'std': std}, "./data/2d_vlasov_two-stream_phase1_EDSR_stats.pt")

    ds = TensorDataset(X, Y)
    tr_ds, va_ds = random_split(ds, [int(0.9 * len(ds)), len(ds) - int(0.9 * len(ds))])
    tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=16)

    model = EDSR(in_ch=200, n_feats=64, n_res_blocks=4,
                 upscale_factor=upscale_factor, mean=mean, std=std).to(device)
    crit = nn.L1Loss()
    opt  = optim.AdamW(model.parameters(), lr=5e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-6)

    best_val = float('inf')
    tr_hist, va_hist = [], []

    for ep in trange(5001, desc="Epochs", ascii=True):
        # ── Train ───────────────────────────────────────────────
        model.train(); run_loss = 0.0
        m0_max = m1_max = m2_max = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            run_loss += loss.item()

            rho_o, J_o, M2_o = compute_moments_torch(pred, q=-1.0)
            rho_t, J_t, M2_t = compute_moments_torch(yb,   q=-1.0)
            m0_max = max(m0_max, (rho_o - rho_t).abs().max().item())
            m1_max = max(m1_max, (J_o   - J_t).abs().max().item())
            m2_max = max(m2_max, (M2_o  - M2_t).abs().max().item())

        if ep % 100 == 0:
            tr_loss = run_loss / len(tr_ld)
            tr_hist.append(tr_loss)

            # ── Validation ────────────────────────────────────
            model.eval(); val_sum = 0.0
            vm0 = vm1 = vm2 = 0.0
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_sum += crit(pred, yb).item()

                    rho_o, J_o, M2_o = compute_moments_torch(pred, q=-1.0)
                    rho_t, J_t, M2_t = compute_moments_torch(yb,   q=-1.0)
                    vm0 = max(vm0, (rho_o - rho_t).abs().max().item())
                    vm1 = max(vm1, (J_o   - J_t).abs().max().item())
                    vm2 = max(vm2, (M2_o  - M2_t).abs().max().item())

            va_loss = val_sum / len(va_ld)
            va_hist.append(va_loss)
            print(f"[{ep:4d}] train={tr_loss:.6e} | val={va_loss:.6e}")
            print(f"     moments train max: ρ {m0_max:.2e}, J {m1_max:.2e}, M2 {m2_max:.2e}")
            print(f"     moments val   max: ρ {vm0:.2e}, J {vm1:.2e}, M2 {vm2:.2e}")

            # ── Checkpoint ────────────────────────────────────
            ckpt = {
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': opt.state_dict(),
                'val_loss': va_loss
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, f"2d_vlasov_two-stream_EDSR_phase1_ep{ep:04d}.pth"))
            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR, "2d_vlasov_two-stream_EDSR_phase1_best.pth"))
                print("   ↳ saved new best")

            np.save(os.path.join(LOG_DIR, "2d_vlasov_two-stream_EDSR_train_loss.npy"), np.array(tr_hist))
            np.save(os.path.join(LOG_DIR, "2d_vlasov_two-stream_EDSR_val_loss.npy"),   np.array(va_hist))

# ----------------------------------------------------------------
if __name__ == "__main__":
    train_model()
