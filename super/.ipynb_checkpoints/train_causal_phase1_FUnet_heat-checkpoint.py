import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import math

# ================================================================
# User‐defined memory length ℓ
# ================================================================
memory_length = 10   # ← for example, ℓ = 10

# ================================================================
# Data Loading (unchanged from your original script)
# ================================================================
def load_data():
    input_CG  = np.load('/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_coarse_all_smooth_gauss_1k.npy')
    target_FG = np.load('/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k.npy')
    input_tensor  = torch.tensor(input_CG,  dtype=torch.float32)[:, :101]   # (N,101,Hc,Wc)
    target_tensor = torch.tensor(target_FG, dtype=torch.float32)[:, :101]   # (N,101,Hf,Wf)
    return input_tensor, target_tensor


# ================================================================
# Coordinate → Fourier features (unchanged)
# ================================================================
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):                               # coords: (B,H,W,2)
        proj = 2 * math.pi * torch.matmul(coords, self.B)    # (B,H,W,mapping_size)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)                        # (B, 2*mapping_size, H, W)


def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid  # (B, H, W, 2)


# ================================================================
# FourierLayer, ConvBlock, PixelShuffleUpsample (unchanged)
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
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2+1)
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
    """[Conv → GELU] × 2 (keeps H×W)."""
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
# U-Net with Fourier bottleneck + PixelShuffle upsampling
# (unchanged from your original except that forward() is replaced)
# ================================================================
class SuperResUNetCausalVec(nn.Module):
    def __init__(
        self,
        in_channels=101,
        lift_dim=128,
        mapping_size=64,
        mapping_scale=5.0,
        final_scale=2,
        modes1=32,
        modes2=32,
        out_channels=101
    ):
        super().__init__()

        # -------- lift ---------------
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        # -------- encoder ------------
        self.enc1 = ConvBlock(lift_dim,        lift_dim)        # keep (Hc, Wc)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)    # pool → (Hc/2, Wc/2)
        self.pool = nn.MaxPool2d(2)

        # -------- bottleneck ---------
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=modes1, modes2=modes2),
            nn.GELU()
        )

        # -------- decoder ------------
        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)

        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim, upscale=2)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)

        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )

        # -------- output head --------
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

        # -------- precompute the causal mask M[t,k] for t,k in [0..100] --------
        T_coarse = in_channels
        ℓ = memory_length
        mask = torch.zeros(T_coarse, T_coarse)  # shape (101,101)
        for t in range(T_coarse):
            start = max(0, t - ℓ)
            mask[t, start : t+1] = 1.0
        # We will register this on CPU (it’ll be moved to .device in forward)
        self.register_buffer('causal_mask', mask)  # shape (101,101)


    def forward(self, x_coarse):
        """
        x_coarse: (B, 101, Hc, Wc)
        Returns:  (B, 101, Hf, Wf)
        Each output channel t only "sees" coarse channels in [t-ℓ...t].
        """

        B, T_coarse, Hc, Wc = x_coarse.shape
        device = x_coarse.device
        ℓ = memory_length
        mask = self.causal_mask.to(device)                 # (101,101)

        # 1) Precompute Fourier features at coarse resolution once:
        coords = get_coord_grid(B, Hc, Wc, device)          # (B, Hc, Wc, 2)
        fourier_feats = self.fourier_mapping(coords)        # (B, 2*mapping, Hc, Wc)

        # 2) Build a giant masked‐input batch:
        #
        #    a) Expand x_coarse from (B,101,Hc,Wc) → (B,101,101,Hc,Wc)
        #       by unsqueezing at dim=1 and repeating along that axis:
        x_expanded = x_coarse.unsqueeze(1).repeat(1, T_coarse, 1, 1, 1)
        #    shape now (B, 101, 101, Hc, Wc), where the middle dim is "t index"

        #    b) Broadcast mask (101,101) → (1,101,101,1,1) and multiply:
        mask5 = mask.view(1, T_coarse, T_coarse, 1, 1)      # (1,101,101,1,1)
        x_masked_big = x_expanded * mask5                   # (B,101,101,Hc,Wc)
        #    Now x_masked_big[b,t,k,:,:] = x_coarse[b,k,:,:] if k∈[t-ℓ..t], else 0.

        #    c) Collapse the first two dims (B,101) → (B*101):
        x_masked_batch = x_masked_big.reshape(B * T_coarse, T_coarse, Hc, Wc)
        #    shape = (B*101, 101, Hc, Wc)

        # 3) Run the *entire* block through the same U-Net (lift/enc/bottle/dec/out):
        #
        #    We need to repeat the Fourier features for each “sub-batch.”
        #    fourier_feats is currently (B, 2*mapping, Hc, Wc).  We want shape
        #    (B*101, 2*mapping, Hc, Wc) to match x_masked_batch.
        fourier_feats_big = fourier_feats.unsqueeze(1).repeat(1, T_coarse, 1, 1, 1)
        #    shape (B,101, 2*mapping, Hc, Wc)
        fourier_feats_big = fourier_feats_big.reshape(B * T_coarse, -1, Hc, Wc)
        #    now (B*101, 2*mapping, Hc, Wc)

        #    a) lift
        lifted_big = torch.cat([x_masked_batch, fourier_feats_big], dim=1)
        #    shape (B*101, 101 + 2*mapping, Hc, Wc)
        lifted_big = self.lift(lifted_big)  # → (B*101, lift_dim, Hc, Wc)

        #    b) encoder
        e1_big = self.enc1(lifted_big)           # (B*101, lift_dim, Hc, Wc)
        e2_big = self.enc2(self.pool(e1_big))    # (B*101, 2*lift_dim, Hc/2, Wc/2)

        #    c) bottleneck
        b_big = self.bottleneck(e2_big)          # (B*101, 2*lift_dim, Hc/2, Wc/2)

        #    d) decoder
        d2_big = self.up1(b_big)                 # (B*101, 2*lift_dim, Hc/2, Wc/2)
        d2_big = self.dec2(torch.cat([d2_big, e2_big], dim=1))
        #    → (B*101, lift_dim, Hc/2, Wc/2)

        d1_big = self.up2(d2_big)                 # (B*101, lift_dim, Hc, Wc)
        d1_big = self.dec1(torch.cat([d1_big, e1_big], dim=1))
        #    → (B*101, lift_dim//2, Hc, Wc)

        d0_big = self.dec0(d1_big)                # (B*101, lift_dim//4, Hf, Wf)

        #    e) final out_head → 101 output channels
        out_101_big = self.out_head(d0_big)       # (B*101, 101, Hf, Wf)

        # 4) “Un‐pack” and pick the diagonal slice:
        #
        #    Reshape (B*101, 101, Hf, Wf) → (B, 101, 101, Hf, Wf)
        out_101_all = out_101_big.view(B, T_coarse, T_coarse, out_101_big.size(-2), out_101_big.size(-1))
        #    Now out_101_all[b, t, k, :, :] is the fine‐grid prediction (for time k)
        #    when input was masked to only include [k-ℓ..k] at the coarse‐side “t” index.
        #
        #    We want, for each output‐time t, to pick out channel k=t (the diagonal):
        #    i.e. out[b, t, :, :] = out_101_all[b, t, t, :, :].
        #
        #    We can do this by indexing.  One way is:
        diag_idx = torch.arange(T_coarse, device=device)       # (101,)
        out_diag = out_101_all[:, diag_idx, diag_idx, :, :]    # → (B, 101, Hf, Wf)

        return out_diag  # (B, 101, Hf, Wf)


# ================================================================
# (Optional) Spectral‐magnitude‐matching loss (unchanged)
# ================================================================
def spectral_loss(output, target):
    fft_o = torch.fft.rfft2(output)
    fft_t = torch.fft.rfft2(target)
    return torch.mean(torch.abs(torch.abs(fft_o) - torch.abs(fft_t)))


# ================================================================
# Training Loop (exactly as before, just instantiate SuperResUNetCausalVec)
# ================================================================
def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------------- data ----------------
    print('Loading data …')
    X_coarse, Y_fine = load_data()
    # X_coarse: (N, 101, Hc, Wc)
    # Y_fine:   (N, 101, Hf, Wf)
    print(f'coarse inputs: {X_coarse.shape}   fine targets: {Y_fine.shape}')

    # Determine upscale factor
    _, _, Hc, Wc = X_coarse.shape
    _, _, Hf, Wf = Y_fine.shape
    upscale_factor = Hf // Hc
    assert (Wf // Wc) == upscale_factor, "Non-uniform upscale factors are not supported."

    # Compute mean/std over all 101 coarse channels:
    data_mean = X_coarse.mean(dim=(0,2,3), keepdim=True)   # (1,101,1,1)
    data_std  = X_coarse.std(dim=(0,2,3), keepdim=True).clamp_min(1e-8)
    torch.save({'data_mean': data_mean, 'data_std': data_std},
               './data/2d_heat_funet_phase1_stats_causal_vec.pt')

    # Build dataset + split
    dataset = TensorDataset(X_coarse, Y_fine)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)  # maybe smaller bs to fit GPU
    val_loader   = DataLoader(val_ds,   batch_size=4)

    print(f'Train={len(train_ds)}  Val={len(val_ds)}  |  upscale ×{upscale_factor}  |  memory_length={memory_length}')

    # ---------------- model ----------------
    model = SuperResUNetCausalVec(
        in_channels   = 101,
        lift_dim      = 128,
        mapping_size  = 64,
        mapping_scale = 5.0,
        final_scale   = upscale_factor,
        modes1        = 32,
        modes2        = 32,
        out_channels  = 101
    ).to(device)

    num_epochs  = 5000
    criterion   = nn.L1Loss()
    optimizer   = optim.AdamW(model.parameters(), lr=1e-3)
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
            # inputs:  (B, 101, Hc, Wc)
            # targets: (B, 101, Hf, Wf)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Normalize all 101 coarse channels
            norm_in  = (inputs - data_mean.to(device)) / data_std.to(device)
            outputs  = model(norm_in)  # (B,101,Hf,Wf)
            # Denormalize just as before (coarse & fine share scales)
            outputs  = outputs * data_std.to(device) + data_mean.to(device)

            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            loss.backward()
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
            }, f'/pscratch/sd/h/hbassi/models/2d_heat_FUnet_ckpt_causal_vec_{epoch:04d}.pth')

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           '/pscratch/sd/h/hbassi/models/2d_heat_FUnet_best_causal_vec.pth')
                print(f'  ↳ new best loss ({best_val:.8f}) saved')

            np.save('./logs/2d_heat_FUnet_train_causal_vec.npy', np.asarray(tr_hist))
            np.save('./logs/2d_heat_FUnet_val_causal_vec.npy',   np.asarray(va_hist))


# ----------------------------------------------------------------
if __name__ == '__main__':
    train_model()
