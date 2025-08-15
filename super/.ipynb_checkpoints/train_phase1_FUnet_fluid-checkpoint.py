import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import math

#torch.set_float32_matmul_precision('high')

# ================================================================
# Data Loading
# ================================================================
def load_data():
    data = np.load('/pscratch/sd/h/hbassi/fluid_data/fluid_dynamics_datasets_6to8.npz')
    input_CG  = data['coarse']
    target_FG = data['fine']
    #import pdb; pdb.set_trace()
    input_tensor  = torch.tensor(input_CG[:, :100, :-1, :-1],  dtype=torch.float32)      
    target_tensor = torch.tensor(target_FG[:, :100, :-1, :-1], dtype=torch.float32)      
    return input_tensor, target_tensor

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
    def __init__(self, in_channels=100, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0, final_scale=4):
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
    data_mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)   # (1,11,1,1)
    data_std  = input_tensor.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    torch.save({'data_mean': data_mean, 'data_std': data_std},
               './data/fluids_funet_phase1_stats.pt')

    dataset = TensorDataset(input_tensor, target_tensor)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    print(f'Train={len(train_ds)}  Val={len(val_ds)}  |  upscale ×{upscale_factor}')

    # ---------------- model ----------------
    model = SuperResUNet(final_scale=upscale_factor).to(device)
    num_epochs  = 5000
    criterion   = nn.L1Loss()
    optimizer   = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val = float('inf')
    tr_hist, va_hist = [], []

    # ---------------- loop -----------------
    for epoch in trange(num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # ---- normalise inputs, forward, denormalise outputs ----
            norm_in  = (inputs  - data_mean.to(device)) / data_std.to(device)
            outputs  = model(norm_in)
            outputs  = outputs * data_std.to(device) + data_mean.to(device)   # undo

            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # ---------------- validation ----------------
        if epoch % 100 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    norm_in = (inputs - data_mean.to(device)) / data_std.to(device)
                    outputs = model(norm_in)
                    outputs = outputs * data_std.to(device) + data_mean.to(device)
                    val_loss += criterion(outputs, targets).item()

            val_loss /= len(val_loader)
            tr_loss   = epoch_loss / len(train_loader)
            print(f'Epoch {epoch:4d} | train {tr_loss:.8f} | val {val_loss:.8f}')

            tr_hist.append(tr_loss)
            va_hist.append(val_loss)

            # ---- checkpointing ----
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict(),
                'train_loss': tr_loss,
                'val_loss': val_loss,
            }, f'/pscratch/sd/h/hbassi/models/fluids_FUnet_ckpt_{epoch:04d}_64to256.pth')

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           '/pscratch/sd/h/hbassi/models/fluids_FUnet_best_PS_FT_64to256.pth')
                print(f'  ↳ new best loss ({best_val:.8f}) saved')

            np.save('./logs/fluids_FUnet_train_PS_FT_64to256.npy', np.asarray(tr_hist))
            np.save('./logs/fluids_FUnet_val_PS_FT_64to256.npy',   np.asarray(va_hist))

# ----------------------------------------------------------------
if __name__ == '__main__':
    train_model()
