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
    input_CG  = np.load(f"/pscratch/sd/h/hbassi/wave_dataset_multi_sf_modes=10_kmax=10/u_coarse_sf={4}.npy")
    target_FG = np.load('/pscratch/sd/h/hbassi/wave_dataset_multi_sf_modes=10_kmax=10/u_fine.npy')
    input_tensor  = torch.tensor(input_CG,  dtype=torch.float32)      # (N,11,Hc,Wc)
    target_tensor = torch.tensor(target_FG, dtype=torch.float32)      # (N,11,Hf,Wf)
    return input_tensor[:, :101], target_tensor[:, :101]

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

# ---------------------------------------------------------------
def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid                                              # (B,H,W,2)

# # ================================================================
# # Fourier Neural Operator 2-D spectral layer
# # ================================================================
# class FourierLayer(nn.Module):
#     def __init__(self, in_ch, out_ch, modes1, modes2):
#         super().__init__()
#         self.modes1, self.modes2 = modes1, modes2
#         self.weight = nn.Parameter(
#             torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
#             / (in_ch * out_ch)
#         )

#     @staticmethod
#     def compl_mul2d(inp, w):                                 # (B,IC,H,W) × (IC,OC,H,W)
#         return torch.einsum('bixy,ioxy->boxy', inp, w)

#     def forward(self, x):                                    # (B,C,H,W)  real
#         B, _, H, W = x.shape
#         x_ft = torch.fft.rfft2(x)

#         m1 = min(self.modes1, H)
#         m2 = min(self.modes2, x_ft.size(-1))                 # W_freq = W//2+1

#         out_ft = torch.zeros(
#             B, self.weight.size(1), H, x_ft.size(-1),
#             dtype=torch.cfloat, device=x.device
#         )
#         out_ft[:, :, :m1, :m2] = self.compl_mul2d(
#             x_ft[:, :, :m1, :m2],
#             self.weight[:, :, :m1, :m2]
#         )
#         return torch.fft.irfft2(out_ft, s=x.shape[-2:])

# # ---------------------------------------------------------------
# class ConvBlock(nn.Module):
#     """[Conv → GELU] × 2 (keeps H×W)."""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.GELU()
#         )
#     def forward(self, x): return self.block(x)

# # ================================================================
# # PixelShuffle-based up-sample block 
# # ================================================================
# class PixelShuffleUpsample(nn.Module):
#     def __init__(self, in_ch, out_ch, upscale=2):
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch * (upscale ** 2), 3, padding=1)
#         self.pix  = nn.PixelShuffle(upscale)
#         self.act  = nn.GELU()
#     def forward(self, x):
#         return self.act(self.pix(self.conv(x)))

# # ================================================================
# # U-Net with Fourier bottleneck + PixelShuffle up-sampling
# # ================================================================
# class SuperResUNet(nn.Module):
#     def __init__(
#         self,
#         in_channels=101,
#         lift_dim=128,
#         mapping_size=64,
#         mapping_scale=5.0,
#         final_scale=2        # ← auto-detected from data
#     ):
#         super().__init__()

#         # -------- lift ---------------
#         self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
#         lifted_ch = in_channels + 2 * mapping_size
#         self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

#         # -------- encoder ------------
#         self.enc1 = ConvBlock(lift_dim,        lift_dim)        # keep  (Hc)
#         self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)    # pool → (Hc/2)
#         self.pool = nn.MaxPool2d(2)

#         # -------- bottleneck ---------
#         self.bottleneck = nn.Sequential(
#             ConvBlock(lift_dim * 2, lift_dim * 2),
#             FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
#             nn.GELU()
#         )

#         # -------- decoder ------------
#         # up1 keeps spatial dims (upscale=1) so it matches e2
#         self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
#         self.dec2 = ConvBlock(lift_dim * 4, lift_dim)                    # cat(up1,e2)

#         self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)             # ×2  (Hc/2 → Hc)
#         self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)               # cat(up2,e1)

#         self.dec0 = nn.Sequential(                                       # Hc → Hc×final_scale
#             PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
#             ConvBlock(lift_dim // 4, lift_dim // 4)
#         )

#         # -------- output head --------
#         self.out_head = nn.Sequential(
#             nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, in_channels, 3, padding=1)                      # 11-channel output
#         )

#     # -----------------------------------------------------------
#     def forward(self, x):                                 # (B,11,Hc,Wc) normalised
#         B, _, H, W = x.shape
#         coords = get_coord_grid(B, H, W, x.device)
#         x = torch.cat([x, self.fourier_mapping(coords)], dim=1)   # lift
#         x = self.lift(x)

#         e1 = self.enc1(x)               # Hc
#         e2 = self.enc2(self.pool(e1))   # Hc/2

#         # ---- bottleneck (no extra pooling) ----
#         b  = self.bottleneck(e2)        # Hc/2

#         # ---- decoder ----
#         d2 = self.up1(b)                             # Hc/2  (spatially matches e2)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))

#         d1 = self.up2(d2)                            # Hc
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))

#         d0 = self.dec0(d1)                           # Hf
#         return self.out_head(d0)                     # (B,11,Hf,Wf)  normalised

import torch.nn.functional as F

# ------------------------------------------------------------------
# (Assume `FourierFeatureMapping` and `get_coord_grid`
#  are defined elsewhere in your codebase.)
# ------------------------------------------------------------------

# ================================================================
# Blur-pool down-sampling (antialiased)
# ================================================================
class BlurPool2d(nn.Module):
    """
    Depth-wise 3×3 Gaussian blur followed by 2× average-pool.
    Keeps the number of channels unchanged and halves H, W.
    """
    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor([1., 2., 1.])           #  [1 2 1]^T [1 2 1]
        kernel = (k[:, None] * k[None, :]).float()
        kernel /= kernel.sum()                   # normalise
        self.register_buffer(
            "weight",
            kernel[None, None, :, :].repeat(channels, 1, 1, 1)  # (C,1,3,3)
        )

    def forward(self, x):                       # (B,C,H,W) → (B,C,H/2,W/2)
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        x = F.conv2d(x, self.weight, stride=1, groups=x.shape[1])
        return F.avg_pool2d(x, kernel_size=2, stride=2)


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
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1, m2 = min(self.modes1, H), min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(
            B, self.weight.size(1), H, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2]
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
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ================================================================
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale ** 2), 3, padding=1)
        self.pix = nn.PixelShuffle(upscale)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.pix(self.conv(x)))


# ================================================================
# Helper for HANO-style multi-scale feature aggregation
# ================================================================
class ToFine(nn.Module):
    def __init__(self, in_ch, out_ch, up_factor: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=up_factor,
                              mode="bilinear", align_corners=False)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.proj(self.up(x))


# ================================================================
class SuperResUNet(nn.Module):
    """
    FUnet + HANO-style multi-scale fusion with antialiased blur-pool
    down-sampling in the encoder.
    """
    def __init__(
        self,
        in_channels: int = 101,
        lift_dim: int = 128,
        mapping_size: int = 64,
        mapping_scale: float = 5.0,
        final_scale: int = 2,
    ):
        super().__init__()

        # Fourier positional encoding & lift
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        # Encoder
        self.enc1 = ConvBlock(lift_dim, lift_dim)
        self.pool = BlurPool2d(lift_dim)               # ← replaces MaxPool2d
        self.enc2 = ConvBlock(lift_dim, lift_dim * 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
            nn.GELU(),
        )

        # Decoder
        self.up1 = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)

        self.up2 = PixelShuffleUpsample(lift_dim, lift_dim)      # ×2
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)

        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4),
        )

        # HANO-style fusion
        agg_ch = lift_dim // 4
        self.tofine_e1 = ToFine(lift_dim,        agg_ch, up_factor=final_scale)
        self.tofine_e2 = ToFine(lift_dim * 2,    agg_ch, up_factor=2 * final_scale)
        self.tofine_b  = ToFine(lift_dim * 2,    agg_ch, up_factor=2 * final_scale)
        self.tofine_d1 = ToFine(lift_dim // 2,   agg_ch, up_factor=final_scale)

        # Output head
        self.out_head = nn.Sequential(
            nn.Conv2d(agg_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )

    # ------------------------------------------------------------
    def forward(self, x):
        B, _, H, W = x.shape

        # positional encoding
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)
        x = self.lift(x)

        # encoder
        e1 = self.enc1(x)                    # (B, D, H,   W)
        e2 = self.enc2(self.pool(e1))        # (B, 2D, H/2, W/2)

        # bottleneck
        b = self.bottleneck(e2)              # (B, 2D, H/2, W/2)

        # decoder
        d2 = self.dec2(torch.cat([self.up1(b), e2], dim=1))      # (B, D,   H/2, W/2)
        d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))     # (B, D/2, H,   W)
        d0 = self.dec0(d1)                                       # (B, D/4, 2H,  2W)

        # HANO-style “sum of all levels” fusion
        fine = (
            d0
            + self.tofine_e1(e1)
            + self.tofine_e2(e2)
            + self.tofine_b(b)
            + self.tofine_d1(d1)
        )

        return self.out_head(fine)

# ================================================================
class H1RelativeLoss(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        # cached |k|² grid lives on the same device/dtype as the first forward() call
        self.register_buffer("_k2", None, persistent=False)

    def _build_k2(self, H: int, W: int, device, dtype):
        ky = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)  # (H,)
        kx = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype)  # (W,)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")                 # (H,W)
        self._k2 = (KY**2 + KX**2)                                     # (H,W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # shapes must match and end with (H,W)
        assert pred.shape == target.shape, "pred/target shapes differ"
        if self._k2 is None or self._k2.shape != pred.shape[-2:]:
            self._build_k2(*pred.shape[-2:], device=pred.device, dtype=pred.dtype)

        diff_fft = torch.fft.fftn(pred - target, dim=(-2, -1))
        targ_fft = torch.fft.fftn(target,        dim=(-2, -1))

        num = (self._k2 * diff_fft.abs()**2).sum(dim=(-2, -1))
        den = (self._k2 * targ_fft.abs()**2).sum(dim=(-2, -1)).clamp_min(self.eps)

        rel_err = num / den                       # [... batch/time …]
        return rel_err.mean()                     # scalar loss

# ================================================================
# Training Loop
# ================================================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
               './data/2d_wave_funet_phase1_stats_32to128_v1_high_multi-level_features_L1.pt')

    dataset = TensorDataset(input_tensor, target_tensor)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    print(f'Train={len(train_ds)}  Val={len(val_ds)}  |  upscale ×{upscale_factor}')

    # ---------------- model ----------------
    model = SuperResUNet(final_scale=upscale_factor).to(device)
    num_epochs  = 3000
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            }, f'/pscratch/sd/h/hbassi/models/2d_wave_FUnet_ckpt_{epoch:04d}_v1_high_sf=4_multi-level_features_L1.pth')

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           '/pscratch/sd/h/hbassi/models/2d_wave_FUnet_best_v1_high_sf=4_multi-level_features_L1.pth')
                print(f'  ↳ new best loss ({best_val:.8f}) saved')

            np.save('./logs/2d_wave_FUnet_train_PS_FT_32to128_v1_high_sf=4_multi-level_features_L1.npy', np.asarray(tr_hist))
            np.save('./logs/2d_wave_FUnet_val_PS_FT_32to128_v1_high_sf=4_multi-level_features_L1.npy',   np.asarray(va_hist))

# ----------------------------------------------------------------
if __name__ == '__main__':
    train_model()
