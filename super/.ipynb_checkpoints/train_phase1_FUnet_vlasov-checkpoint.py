import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import math
CONFIGS  = [(8, 8)]  # (mx, my)

# ================================================================
# Data Loading
# ================================================================
def load_data():
    data = np.load('/pscratch/sd/h/hbassi/landau_ds_t=15.npz')
    input_CG = data['fe_coarse']
    target_FG = data['fe_fine']
    #input_CG  = np.load('/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_coarse_64_fixed_timestep_buneman_phase1_training_data_no_ion.npy')
    #target_FG = np.load('/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_fine_256_fixed_timestep_buneman_phase1_training_data_no_ion.npy')
    #input_CG = np.load(f'/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_coarse_32_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy')
    #target_FG = np.load(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_fine_128_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy")
    #import pdb; pdb.set_trace()
    input_tensor  = torch.tensor(input_CG,  dtype=torch.float32)      # (N, T, Hc, Wc)
    target_tensor = torch.tensor(target_FG, dtype=torch.float32)      # (N, T, Hf, Wf)
    return input_tensor[:, :75], target_tensor[:, :75]

# ================================================================
# Moments helper
# ================================================================
ve_lims = (-6.0, 6.0)
def compute_moments_torch(f, q=-1.0):
    """
    Compute the first three velocity moments of a batch of 1-D phase–space slices.
    f: Tensor of shape (..., Nx, Nv)
    Returns rho, J, M2 each of shape (..., Nx).
    """
    # f: (..., Nx, Nv)
    v = torch.linspace(ve_lims[0], ve_lims[1], steps=f.size(-1), device=f.device)
    dv = v[1] - v[0]
    rho = q * f.sum(dim=-1)             * dv      # (..., Nx)
    J   = q * (f * v).sum(dim=-1)       * dv
    M2  = q * (f * v**2).sum(dim=-1)    * dv
    return rho, J, M2

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
            torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            / (in_ch * out_ch)
        )

    @staticmethod
    def compl_mul2d(inp, w):
        return torch.einsum('bixy,ioxy->boxy', inp, w)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(B, self.weight.size(1), H, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
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
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

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
    def __init__(self, in_channels=75, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0, final_scale=2):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)
        self.enc1 = ConvBlock(lift_dim,        lift_dim)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
            nn.GELU()
        )
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
        x = self.lift(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([self.up1(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))
        d0 = self.dec0(d1)
        return self.out_head(d0)

# ================================================================
def spectral_loss(output, target):
    fft_o = torch.fft.rfft2(output)
    fft_t = torch.fft.rfft2(target)
    return torch.mean(torch.abs(torch.abs(fft_o) - torch.abs(fft_t)))

# ================================================================
# Training Loop
# ================================================================
def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------------- data ----------------
    print('Loading data …')
    input_tensor, target_tensor = load_data()
    print(f'inputs  {input_tensor.shape}   targets  {target_tensor.shape}')

    # -------- auto-detect upscale factor -------
    upscale_factor = target_tensor.shape[2] // input_tensor.shape[2]
    assert target_tensor.shape[3] // input_tensor.shape[3] == upscale_factor, \
           "Non-uniform upscale factors are not supported."

    # -------- mean-shift statistics (per-channel) -------
    data_mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)
    data_std  = input_tensor.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    torch.save({'data_mean': data_mean, 'data_std': data_std},
               './data/2d_vlasov_funet_phase1_stats_64to256_landau_v1.pt')

    dataset = TensorDataset(input_tensor, target_tensor)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    print(f'Train={len(train_ds)}  Val={len(val_ds)}  |  upscale ×{upscale_factor}')

    # ---------------- model ----------------
    model      = SuperResUNet(in_channels=75, final_scale=upscale_factor).to(device)
    num_epochs = 5000
    criterion = nn.L1Loss()
    optimizer  = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val = float('inf')
    tr_loss_hist, va_loss_hist = [], []
    tr_m0_hist, tr_m1_hist, tr_m2_hist = [], [], []
    va_m0_hist, va_m1_hist, va_m2_hist = [], [], []

    for epoch in trange(num_epochs + 1):
        # ---- training ----
        model.train()
        epoch_loss = 0.0
        train_m0_max = train_m1_max = train_m2_max = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            norm_in = (inputs - data_mean.to(device)) / data_std.to(device)
            outputs = model(norm_in)
            outputs = outputs * data_std.to(device) + data_mean.to(device)

            # compute epoch training loss
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # compute moment errors for this batch
            rho_o, J_o, M2_o = compute_moments_torch(outputs, q=-1.0)
            rho_t, J_t, M2_t = compute_moments_torch(targets, q=-1.0)
            err_rho = torch.abs(rho_o - rho_t).max().item()
            err_J   = torch.abs(J_o   - J_t).max().item()
            err_M2  = torch.abs(M2_o  - M2_t).max().item()
            train_m0_max = max(train_m0_max, err_rho)
            train_m1_max = max(train_m1_max, err_J)
            train_m2_max = max(train_m2_max, err_M2)

            loss.backward()
            optimizer.step()
            scheduler.step()

        # ---- validation and logging every 100 epochs ----
        if epoch % 100 == 0:
            # compute average train loss
            tr_loss = epoch_loss / len(train_loader)
            tr_loss_hist.append(tr_loss)
            tr_m0_hist.append(train_m0_max)
            tr_m1_hist.append(train_m1_max)
            tr_m2_hist.append(train_m2_max)

            model.eval()
            val_loss = 0.0
            val_m0_max = val_m1_max = val_m2_max = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    norm_in = (inputs - data_mean.to(device)) / data_std.to(device)
                    outputs = model(norm_in)
                    outputs = outputs * data_std.to(device) + data_mean.to(device)

                    val_loss += criterion(outputs, targets).item()

                    rho_o, J_o, M2_o = compute_moments_torch(outputs, q=-1.0)
                    rho_t, J_t, M2_t = compute_moments_torch(targets, q=-1.0)
                    err_rho = torch.abs(rho_o - rho_t).max().item()
                    err_J   = torch.abs(J_o   - J_t).max().item()
                    err_M2  = torch.abs(M2_o  - M2_t).max().item()
                    val_m0_max = max(val_m0_max, err_rho)
                    val_m1_max = max(val_m1_max, err_J)
                    val_m2_max = max(val_m2_max, err_M2)

            va_loss = val_loss / len(val_loader)
            va_loss_hist.append(va_loss)
            va_m0_hist.append(val_m0_max)
            va_m1_hist.append(val_m1_max)
            va_m2_hist.append(val_m2_max)

            print(f'Epoch {epoch:4d} | train_loss {tr_loss:.6f} | val_loss {va_loss:.6f}')
            print(f'  ↳ train moments max err: ρ {train_m0_max:.3e}, J {train_m1_max:.3e}, M2 {train_m2_max:.3e}')
            print(f'  ↳ val   moments max err: ρ {val_m0_max:.3e}, J {val_m1_max:.3e}, M2 {val_m2_max:.3e}')

            # ---- checkpointing ----
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict(),
                'train_loss': tr_loss,
                'val_loss': va_loss,
            }, f'/pscratch/sd/h/hbassi/models/2d_vlasov_landau_FUnet_ckpt_{epoch:04d}_PS_FT_32to128_t=75_no_ion_pth')

            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(),
                           '/pscratch/sd/h/hbassi/models/2d_vlasov_landau_FUnet_best_PS_FT_32to128_t=75_no_ion.pth')
                print(f'  ↳ new best val_loss ({best_val:.6f}) saved')

            # ---- save histories ----
            np.save('./logs/2d_vlasov_landau_FUnet_train_PS_FT_32to128_t=75_no_ion_loss.npy',
                    np.array(tr_loss_hist))
            np.save('./logs/2d_vlasov_FUnet_landau_val_PS_FT_32to128_t=75_no_ion_loss.npy',
                    np.array(va_loss_hist))

            np.save('./logs/2d_vlasov_FUnet_landau_train_PS_FT_32to128_t=75_no_ion_mom0.npy',
                    np.array(tr_m0_hist))
            np.save('./logs/2d_vlasov_FUnet_landau_train_PS_FT_32to128_t=75_no_ion_mom1.npy',
                    np.array(tr_m1_hist))
            np.save('./logs/2d_vlasov_FUnet_landau_train_PS_FT_32to128_t=75_no_ion_mom2.npy',
                    np.array(tr_m2_hist))

            np.save('./logs/2d_vlasov_FUnet_landau_val_PS_FT_32to128_t=75_no_ion_mom0.npy',
                    np.array(va_m0_hist))
            np.save('./logs/2d_vlasov_FUnet_landau_val_PS_FT_32to128_t=75_no_ion_mom1.npy',
                    np.array(va_m1_hist))
            np.save('./logs/2d_vlasov_FUnet_landau_val_PS_FT_32to128_t=75_no_ion_mom2.npy',
                    np.array(va_m2_hist))

# ----------------------------------------------------------------
if __name__ == '__main__':
    train_model()
