#!/usr/bin/env python
# ===============================================================
#  train_phase2_tdse2d.py
#  ---------------------------------------------------------------
#  Phase-2 predictor–corrector training for 2-D TDSE super-res
#  Workflow per mini-batch:
#    1) load fine state psi_256^n  (256 x 256 complex grid)
#    2) downsample to 128 x 128            -> psi_128^n
#    3) take one Strang-splitting step     -> psi_128^{n+1}
#    4) feed |psi_128^{n+1}| into U-Net    -> pred_256^{n+1}
#    5) L1 loss against |psi_256^{n+1}|
# ===============================================================
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

# ---------------------------------------------------------------- Paths
PATH_FG   = '/pscratch/sd/h/hbassi/tdse2d/64to128/grid256_dataset.npy'
MODEL_DIR = '/pscratch/sd/h/hbassi/models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('./data', exist_ok=True)

# ===============================================================
#  Torch implementation of a one-step Strang-split propagator on
#  the coarse 128 x 128 grid (4th-order finite-difference Laplacian)
# ===============================================================
# ===============================================================
#  Torch Strang-split steppers (Dirichlet BCs, 4-term K series)
# ===============================================================
class TDSECoarseStepper:
    def __init__(self, N=128, dom_a=-80.0, dom_b=40.0,
                 dt_phys=0.01 * 2.4e-3 * 1e-15,
                 to_atomic=2.4188843265857e-17):
        self.N  = N
        self.dt = torch.tensor(dt_phys / to_atomic, dtype=torch.float32)
        self.dx = torch.tensor((dom_b - dom_a) / (N - 1), dtype=torch.float32)
        xs = torch.linspace(dom_a, dom_b, N, dtype=torch.float32)
        xmat, ymat = torch.meshgrid(xs, xs, indexing='ij')
        pot = ( -((xmat + 10.0)**2 + 1.0)**-0.5
                -((ymat + 10.0)**2 + 1.0)**-0.5
                +((xmat - ymat)**2 + 1.0)**-0.5 )
        self.v_phase = torch.exp(-1j * self.dt * pot).to(torch.complex64)
        self.a_coeff = (1j * self.dt / 2).to(torch.complex64)
        self._dev_cached = None

    @staticmethod
    def _shift_and_zero(t, shift, dim):
        out = torch.roll(t, shifts=shift, dims=dim)
        if shift > 0:
            sl = [slice(None)] * t.ndim; sl[dim] = slice(0, shift)
            out[tuple(sl)] = 0
        elif shift < 0:
            sl = [slice(None)] * t.ndim; sl[dim] = slice(shift, None)
            out[tuple(sl)] = 0
        return out

    def _laplacian_4th(self, psi):
        dx2 = self.dx * self.dx
        p2 = self._shift_and_zero(psi,  2, 3)
        p1 = self._shift_and_zero(psi,  1, 3)
        m1 = self._shift_and_zero(psi, -1, 3)
        m2 = self._shift_and_zero(psi, -2, 3)
        lap_x = (-p2 + 16*p1 - 30*psi + 16*m1 - m2) / (12*dx2)

        p2 = self._shift_and_zero(psi,  2, 2)
        p1 = self._shift_and_zero(psi,  1, 2)
        m1 = self._shift_and_zero(psi, -1, 2)
        m2 = self._shift_and_zero(psi, -2, 2)
        lap_y = (-p2 + 16*p1 - 30*psi + 16*m1 - m2) / (12*dx2)

        return lap_x + lap_y

    def _apply_K(self, psi):
        a_psi = self.a_coeff * self._laplacian_4th(psi)
        out, a_pow = psi + a_psi, a_psi
        for div in (2.0, 6.0, 24.0):
            a_pow = self.a_coeff * self._laplacian_4th(a_pow)
            out   = out + a_pow / div
        return out

    @torch.no_grad()
    def step(self, psi):
        if self._dev_cached != psi.device:
            self.v_phase = self.v_phase.to(psi.device)
            self.a_coeff = self.a_coeff.to(psi.device)
            self.dx      = self.dx.to(psi.device)
            self._dev_cached = psi.device
        psi_out = self._apply_K(psi) * self.v_phase
        return self._apply_K(psi_out)
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
            torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat) /
            (in_ch * out_ch)
        )

    @staticmethod
    def compl_mul2d(inp, w):
        return torch.einsum('bixy,ioxy->boxy', inp, w)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(
            B, self.weight.size(1), H, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])


# ================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


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
class SuperResUNet(nn.Module):
    def __init__(self, in_channels=101, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0, final_scale=2):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size

        # ───── encoder ──────────────────────────────────────────────
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)
        self.enc1 = ConvBlock(lift_dim,        lift_dim)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)
        self.pool = nn.MaxPool2d(2)

        # ───── bottleneck (with FNO-style spectral layer) ──────────
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
            nn.GELU()
        )

        # ───── decoder ──────────────────────────────────────────────
        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)
        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)
        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)

        x  = self.lift(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(e2)

        d2 = self.dec2(torch.cat([self.up1(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))
        d0 = self.dec0(d1)
        return self.out_head(d0)

# ===============================================================
#  Dataset helper  (batched propagation to avoid OOM)
# ===============================================================
def load_phase2_data(max_traj: int = 500,
                     batch_size: int = 32,
                     device: torch.device | str = 'cuda:0'):
    """
    Returns
    -------
    psi_n   : complex64 tensor,  (B, 1, 256, 256)
    psi_np1 : complex64 tensor,  (B, 1, 256, 256)

    Notes
    -----
    * Only the first `max_traj` trajectories and their first 75 snapshots
      are touched on disk (via NumPy mem-mapping → no RAM spike).
    * One-step TDSE propagation is done in slices of size `batch_size` to
      keep GPU / host memory usage low.
    """
    # ---------- lazy on–disk view  ---------------------------------
    fg_mem = np.load(PATH_FG, mmap_mode="r")          # shape (Ntraj, Nt, 256, 256)
    psi_n_np = fg_mem[:max_traj, :75]                 # still lazy

    # ---------- flatten (traj, t) → batch dimension ----------------
    B_total = psi_n_np.shape[0] * psi_n_np.shape[1]   # = max_traj * 75
    psi_n_np = psi_n_np.reshape(B_total, 256, 256)    # (B, 256, 256)

    # ---------- torch tensor on *CPU* first ------------------------
    psi_n_cpu = torch.from_numpy(psi_n_np.astype(np.complex64))          # (B,256,256)
    psi_n_cpu = psi_n_cpu.unsqueeze(1)                                   # → (B,1,256,256)

    # ---------- stepper (only one copy on chosen device) -----------
    stepper = TDSECoarseStepper(N=256)

    # ---------- propagate in mini-batches --------------------------
    psi_np1_list = []
    with torch.no_grad():
        for k in range(0, B_total, batch_size):
            chunk = psi_n_cpu[k:k + batch_size].to(device)               # (b,1,256,256)
            psi_np1_chunk = stepper.step(chunk)                          # (b,1,256,256)
            psi_np1_list.append(psi_np1_chunk.cpu())                     # keep result on CPU

    psi_np1_cpu = torch.cat(psi_np1_list, dim=0)                         # (B,1,256,256)

    # ---------- final return (both on CPU; send to GPU later) -------
    return psi_n_cpu.reshape((max_traj, 75, 256, 256)), psi_np1_cpu.reshape((max_traj, 75, 256, 256))


# ===============================================================
#  Training loop
# ===============================================================
def train_phase2():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('[INFO] loading dataset ...')
    psi_n, psi_np1 = load_phase2_data()
    print(f'   psi_n   shape {psi_n.shape}')
    print(f'   psi_np1 shape {psi_np1.shape}')

    # Build dataset with both complex psi and real amplitudes
    amp_n   = psi_n.abs().float()
    amp_np1 = psi_np1.abs().float()
    dataset = TensorDataset(psi_n, amp_n, amp_np1)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # amplitude normalization
    mean = amp_n.mean((0, 2, 3), keepdim=True)
    std  = amp_n.std((0, 2, 3), keepdim=True).clamp_min(1e-8)
    torch.save({'mean': mean, 'std': std},
               './data/tdse2d_phase2_norm_amp_t=75_64to256.pt')

    # model, stepper, optimizer
    net     = SuperResUNet(in_channels=75,final_scale=4).to(device)
    net.load_state_dict(torch.load('/pscratch/sd/h/hbassi/models/2d_tdse_FUnet_best_PS_FT_64to256_500_t=75.pth', map_location=device))
    #net.load_state_dict(torch.load('/pscratch/sd/h/hbassi/models/tdse_phase2_best_model_t=75.pth', map_location=device))
    stepper = TDSECoarseStepper(N=64)
    criterion = nn.L1Loss()
    optimiz  = optim.AdamW(net.parameters(), lr=5e-4)
    sched    = optim.lr_scheduler.CosineAnnealingLR(optimiz,
                                                    T_max=2500, eta_min=1e-6)
   

    best_val = float('inf')

    for epoch in trange(2501):
        net.train()
        epoch_loss = 0.0

        for psi_full, amp, amp_next in loader:
            psi_full = psi_full.to(device)        # complex
            amp_next = amp_next.to(device)        # float
            optimiz.zero_grad()
            # 1) project to coarse grid
            psi_coarse = psi_full[:, :, ::4, ::4]     # (B,1,128,128) complex

            # 2) physics time step
            psi_coarse_next = stepper.step(psi_coarse)    # complex tensor
            #psi_fine_next = finestepper.step(psi_full)
            #target = psi_fine_next.abs().float()
            # 3) take amplitude and normalize
            amp_in = psi_coarse_next.abs().float()
            amp_norm = (amp_in - mean.to(device)) / std.to(device)

            # 4) NN prediction
            pred = net(amp_norm)
            pred = pred * std.to(device) + mean.to(device)

            # 5) loss and optimize
            loss = criterion(pred, amp_next)
            
            loss.backward()
            optimiz.step()
            sched.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f'Epoch {epoch:4d} | training L1 {avg_loss:.6e}')

            # checkpoint
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optim_state_dict': optimiz.state_dict()},
                       os.path.join(MODEL_DIR,
                       f'tdse_phase2_ckpt_{epoch:04d}_t=75_64to256.pth'))

            # best model tracking
            if avg_loss < best_val:
                best_val = avg_loss
                torch.save(net.state_dict(),
                           os.path.join(MODEL_DIR,
                           'tdse_phase2_best_model_t=75_64to256.pth'))
                print(f'   [NEW BEST] {best_val:.6e}')

# ----------------------------------------------------------------------
if __name__ == '__main__':
    train_phase2()
