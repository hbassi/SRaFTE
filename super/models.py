import os, math, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
import torch.fft                          as fft
import torch.nn.functional as F
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
    def __init__(self, in_channels=101, lift_dim=128,
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
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=64, modes2=64),
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
# ===============================================================
# 2 ▸ U‑Net building blocks
# ===============================================================
def conv_block(in_ch, out_ch, k=3, act=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch)
    ]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)

class UNetSR(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 64, upscale_factor: int = 4):
        super().__init__()
        assert upscale_factor in (2, 4, 8), "upscale_factor must be 2, 4, or 8"
        self.upscale_factor = upscale_factor

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = nn.Sequential(conv_block(in_ch, base_ch),
                                  conv_block(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)       #  1/2

        self.enc2 = nn.Sequential(conv_block(base_ch, base_ch * 2),
                                  conv_block(base_ch * 2, base_ch * 2))
        self.pool2 = nn.MaxPool2d(2)       #  1/4

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = nn.Sequential(conv_block(base_ch * 2, base_ch * 4),
                                         conv_block(base_ch * 4, base_ch * 4))

        # ── Decoder (transpose conv) ─────────────────────────────
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)  # 1/2
        self.dec2 = nn.Sequential(conv_block(base_ch * 4, base_ch * 2),
                                  conv_block(base_ch * 2, base_ch * 2))

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)       # 1/1
        self.dec1 = nn.Sequential(conv_block(base_ch * 2, base_ch),
                                  conv_block(base_ch, base_ch))

        self.out_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_head(d1)                     # (N,1,Hc,Wc)
        # Final bilinear upsample to fine resolution
        out = F.interpolate(out,
                            scale_factor=self.upscale_factor,
                            mode='bilinear',
                            align_corners=False)
        return out
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
# 3 ▸ 2‑D Spectral Convolution layer (FNO core)
# ===============================================================
class SpectralConv2d(nn.Module):
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

class FNO2d(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch=1,
                 modes1=16,
                 modes2=16,
                 width=64,
                 n_layers=4):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.width  = width

        self.fc0 = nn.Conv2d(in_ch, width, 1)

        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.w_layers = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, out_ch, 1)

    def forward(self, x):
        """
        x : (B, in_ch, H, W)
        """
        x = self.fc0(x)                                   # lift to width channels
        for spec, w in zip(self.spectral_layers, self.w_layers):
            x = spec(x) + w(x)
            x = self.act(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class FNO2dAR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width):
        super(FNO2dAR, self).__init__()
        self.width = width
        # Lift the input (here, in_channels = T) to a higher-dimensional feature space.
        self.fc0 = nn.Linear(in_channels, self.width)

        # Fourier layers and pointwise convolutions 
        self.conv0 = SpectralConv2dAR(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)

        self.conv1 = SpectralConv2dAR(self.width, self.width, modes1, modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)

        self.conv2 = SpectralConv2dAR(self.width, self.width, modes1, modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        self.conv3 = SpectralConv2dAR(self.width, self.width, modes1, modes2)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # Permute to [B, H, W, T] so each spatial location has a feature vector of length T
        x = x.permute(0, 2, 3, 1)
        # Lift to higher-dimensional space
        x = self.fc0(x)
        # Permute to [B, width, H, W] for convolutional operations
        x = x.permute(0, 3, 1, 2)

        # Apply Fourier layers with local convolution
        x = self.conv0(x) + self.w0(x)
        x = nn.GELU()(x)
        x = self.conv1(x) + self.w1(x)
        x = nn.GELU()(x)
        x = self.conv2(x) + self.w2(x)
        x = nn.GELU()(x)
        x = self.conv3(x) + self.w3(x)

        # Permute back and project to output space
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

# Spectral convolution layer remains unchanged
class SpectralConv2dAR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2dAR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy, ioxy -> boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights
        )
        x = torch.fft.irfft2(out_ft, s=x.shape[-2:])
        return x