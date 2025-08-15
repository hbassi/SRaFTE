import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import math

torch.set_float32_matmul_precision('high')

# ----------------------------
# Data Loading and Preparation
# ----------------------------
nu = 1e-4
k_cutoff_coarse = 7.5
k_cutoff_fine   = 7.5

def load_data():
    input_CG = np.load(
        f'/pscratch/sd/h/hbassi/NavierStokes_coarse_32_nu={nu}_kcutoff={k_cutoff_coarse}_with_forcing_no_norm.npy'
    )[:, :100]  # assume shape (N,100,32,32)
    target_FG = np.load(
        f'/pscratch/sd/h/hbassi/NavierStokes_fine_128_nu={nu}_kcutoff={k_cutoff_fine}_with_forcing_no_norm.npy'
    )[:, :100]  # assume shape (N,100,128,128)
    input_tensor  = torch.tensor(input_CG,  dtype=torch.float32)
    target_tensor = torch.tensor(target_FG, dtype=torch.float32)
    return input_tensor, target_tensor

# ----------------------------
# Coordinate → Fourier features
# ----------------------------
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):                     # coords: (B,H,W,2)
        proj = 2 * math.pi * torch.matmul(coords, self.B)           # (B,H,W,mapping_size)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) # (B,H,W,2*mapping_size)
        return ff.permute(0, 3, 1, 2)                               # (B,2*mapping_size,H,W)

def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid  # (B,H,W,2)

# ================================================================
#            Fourier Neural Operator 2-D spectral layer
# ================================================================
class FourierLayer(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weight = nn.Parameter(torch.randn(
            in_ch, out_ch, modes1, modes2, dtype=torch.cfloat
        ) / (in_ch * out_ch))

    @staticmethod
    def compl_mul2d(inp, w):
        return torch.einsum("bixy,ioxy->boxy", inp, w)

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
        return torch.fft.irfft2(out_ft, s=(H, W))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale ** 2), 3, padding=1)
        self.pix  = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.pix(self.conv(x)))

# ================================================================
#    SuperResUNet: 32×32 → 8×8 → … → 128×128 with 4 upsamples
# ================================================================
class SuperResUNet(nn.Module):
    def __init__(self, in_channels=100, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0):
        super().__init__()
        # lift
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        # encoder
        self.enc1 = ConvBlock(lift_dim,        lift_dim)     # 32×32
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2) # 16×16
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),            # 8×8
            FourierLayer(lift_dim * 2, lift_dim * 2, 32, 32),
            nn.GELU()
        )

        # decoder stage 1: 8→16
        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)

        # decoder stage 2: 16→32
        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)

        # decoder stage 3: 32→64
        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )

        # decoder stage 4: 64→128
        self.up3  = PixelShuffleUpsample(lift_dim // 4, lift_dim // 4)
        self.dec3 = ConvBlock(lift_dim // 4, lift_dim // 4)

        # output head (128×128 → in_channels)
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)

        # lift
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)
        x = self.lift(x)

        # encoder
        e1 = self.enc1(x)                 # 32×32
        e2 = self.enc2(self.pool(e1))     # 16×16

        # bottleneck
        b  = self.bottleneck(self.pool(e2))  # 8×8

        # decoder 1: 8→16
        d2 = self.up1(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # decoder 2: 16→32
        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # decoder 3: 32→64
        d0 = self.dec0(d1)

        # decoder 4: 64→128
        d3 = self.up3(d0)
        d3 = self.dec3(d3)

        # final head
        return self.out_head(d3)  # (B, in_channels, 128, 128)

# ---------------------------------------------------------------
def spectral_loss(output, target):
    fft_output = torch.fft.rfft2(output)
    fft_target = torch.fft.rfft2(target)
    return torch.mean(torch.abs(torch.abs(fft_output) - torch.abs(fft_target)))

# ----------------------------
# Training Setup
# ----------------------------
def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    # Load data
    input_tensor, target_tensor = load_data()
    dataset = TensorDataset(input_tensor, target_tensor)
    train_ds, val_ds = random_split(dataset, [950, 50])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16)

    # Normalize stats
    data_mean = input_tensor.mean(dim=(0,2,3), keepdim=True)
    data_std  = input_tensor.std(dim=(0,2,3), keepdim=True).clamp_min(1e-8)
    torch.save({
    'data_mean': data_mean,
    'data_std':  data_std
}, f'stats_nu={nu}.pt')

    # Model, optimizer, scheduler
    model     = SuperResUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5500, eta_min=1e-6)
    criterion = nn.L1Loss()
    lambda_spec = 0.1

    for epoch in trange(5501):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            norm_in = (inputs - data_mean.to(device)) / data_std.to(device)
            outputs = model(norm_in)
            outputs = outputs * data_std.to(device) + data_mean.to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # record average training loss
        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Periodic validation, printing, and checkpointing
        if epoch % 100 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    out = model((inputs - data_mean.to(device)) / data_std.to(device))
                    out = out * data_std.to(device) + data_mean.to(device)
                    val_loss += criterion(out, targets).item()
            avg_val = val_loss / len(val_loader)
            validation_losses.append(avg_val)

            print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val:.8f}")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val,
            }, f"/pscratch/sd/h/hbassi/models/PSFUnet_NS_checkpoint_epoch_{epoch}_nu={nu}.pth")

            # Save best model
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(),
                           f"/pscratch/sd/h/hbassi/models/best_PSFUnet_NS_nu={nu}.pth")
                print(f"  → New best model (val loss {avg_val:.8f}) saved.")

        # Save loss logs every epoch
        np.save(f'./logs/PSFUnet_train_losses_nu={nu}.npy', np.array(training_losses))
        np.save(f'./logs/PSFUnet_val_losses_nu={nu}.npy',   np.array(validation_losses))

if __name__ == "__main__":
    train_model()
