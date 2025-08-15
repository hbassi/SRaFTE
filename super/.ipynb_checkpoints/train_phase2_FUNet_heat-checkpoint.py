import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import logging
import math
import os


# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    filename=f'training_heat_32to128_smooth2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# Heat equation parameters (must match data‐generation)
Nf = 128
Nc = 32
nu = 0.1    
dt = 0.01
f_coefficient = 0.01     
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Paths to saved datasets (as produced by 2Dheateqn.py)
COARSE_DATA_PATH = '/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_coarse_all_smooth_gauss_1k.npy'
FINE_DATA_PATH   = '/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k.npy'

# Where to save phase‐2 checkpoints and logs
CHECKPOINT_DIR = './pscratch/sd/h/hbassi/checkpoints_phase2_heat_smooth'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Define coarse‐grid one‐step propagator (integrating‐factor, spectral)
# ------------------------------------------------------------
# Precompute Fourier wavenumbers and forcing on coarse grid as torch tensors
kx_c_1d = torch.fft.fftfreq(Nc, d=1.0 / Nc) * (2.0 * math.pi)        # (Nc,)
ky_c_1d = kx_c_1d.clone()                                            # same for y
kx_c, ky_c = torch.meshgrid(kx_c_1d, ky_c_1d, indexing='ij')         # (Nc, Nc)
k2_c = kx_c ** 2 + ky_c ** 2                                         # (Nc, Nc)
# Avoid division by zero on zero mode
k2_c[0, 0] = 1e-14

# Integrating factor for one time step
exp_c = torch.exp(-nu * k2_c * dt).to(device)                        # (Nc, Nc)

# Build forcing f(x,y) = sin(2πx) sin(2πy) on [0,1]^2 for coarse grid
xs = (torch.arange(Nc, dtype=torch.float64) + 0.5) / Nc             # (Nc,)
ys = (torch.arange(Nc, dtype=torch.float64) + 0.5) / Nc             # (Nc,)
Xc, Yc = torch.meshgrid(xs, ys, indexing='ij')                       # (Nc, Nc)
f_xy_c = torch.sin(2 * math.pi * Xc) * torch.sin(2 * math.pi * Yc)   # (Nc, Nc)
f_hat_c = torch.fft.fft2(f_xy_c).to(device)                           # (Nc, Nc), complex64

# Move k2_c to device
k2_c = k2_c.to(device)


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
    def compl_mul2d(inp, w):                                 # (B,IC,H,W) × (IC,OC,H,W)
        return torch.einsum('bixy,ioxy->boxy', inp, w)

    def forward(self, x):                                    # (B,C,H,W)  real
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.size(-1))                 # W_freq = W//2+1

        out_ft = torch.zeros(
            B, self.weight.size(1), H, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])

# ---------------------------------------------------------------
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
    def forward(self, x): return self.block(x)

# ================================================================
# ↓↓↓ PixelShuffle-based up-sample block ↓↓↓
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
# U-Net with Fourier bottleneck + PixelShuffle up-sampling
# NOTE: -- No pooling inside the bottleneck (only in encoder) --
# ================================================================
class SuperResUNet(nn.Module):
    def __init__(
        self,
        in_channels=101,
        lift_dim=128,
        mapping_size=64,
        mapping_scale=5.0,
        final_scale=2        # ← auto-detected from data
    ):
        super().__init__()

        # -------- lift ---------------
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        # -------- encoder ------------
        self.enc1 = ConvBlock(lift_dim,        lift_dim)        # keep  (Hc)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)    # pool → (Hc/2)
        self.pool = nn.MaxPool2d(2)

        # -------- bottleneck ---------
        # ⚠️ Removed extra pooling here
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=64, modes2=64),
            nn.GELU()
        )

        # -------- decoder ------------
        # up1 keeps spatial dims (upscale=1) so it matches e2
        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)                    # cat(up1,e2)

        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)             # ×2  (Hc/2 → Hc)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)               # cat(up2,e1)

        self.dec0 = nn.Sequential(                                       # Hc → Hc×final_scale
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )

        # -------- output head --------
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)                      # 11-channel output
        )

    # -----------------------------------------------------------
    def forward(self, x):                                 # (B,11,Hc,Wc) normalised
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)   # lift
        x = self.lift(x)

        e1 = self.enc1(x)               # Hc
        e2 = self.enc2(self.pool(e1))   # Hc/2

        # ---- bottleneck (no extra pooling) ----
        b  = self.bottleneck(e2)        # Hc/2

        # ---- decoder ----
        d2 = self.up1(b)                             # Hc/2  (spatially matches e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up2(d2)                            # Hc
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.dec0(d1)                           # Hf
        return self.out_head(d0)                     # (B,11,Hf,Wf)  normalised

# ----------------------------
# Projection Operator: Fine -> Coarse (applied channel–wise)
# ----------------------------
def projection_operator(fine_data, factor=2):
    """
    Downsamples the input spatially by the given factor.
    fine_data: Tensor of shape [B, 10, H, W]
    """
    return fine_data[..., ::factor, ::factor]

def coarse_time_step_heat(coarse_seq: torch.Tensor) -> torch.Tensor:
    """
    Advance one time step for every frame in a batch of coarse 2D heat‐equation sequences.

    Input:
        coarse_seq: Tensor of shape [B, T, Nc, Nc] (float32 or float64),
                    representing B sequences, each with T time steps on an Nc×Nc grid.
    Output:
        next_coarse: Tensor of shape [B, T, Nc, Nc] (float32),
                     where next_coarse[b, t] = solver applied to coarse_seq[b, t].
    """
    B, T, Nc, _ = coarse_seq.shape

    # Flatten batch and time dims → shape [B*T, Nc, Nc], convert to float64 for FFT accuracy
    u_flat = coarse_seq.reshape(-1, Nc, Nc).to(torch.float64)

    # Add channel dimension for FFT solver → [B*T, 1, Nc, Nc]
    u_flat = u_flat.unsqueeze(1)

    # Apply one‐step integrating‐factor heat solver on each [1, Nc, Nc]
    # (returns [B*T, 1, Nc, Nc], float32)
    with torch.no_grad():
        u_next_flat = coarse_time_step_heat_single(u_flat)

    # Remove channel dimension → [B*T, Nc, Nc]
    u_next_flat = u_next_flat.squeeze(1)

    # Reshape back to [B, T, Nc, Nc] and return as float32
    return u_next_flat.reshape(B, T, Nc, Nc).to(torch.float32)


def coarse_time_step_heat_single(coarse_field: torch.Tensor) -> torch.Tensor:
    """
    Advance one time step for a batch of coarse 2D heat‐equation frames.

    Input:
        coarse_field: Tensor of shape [M, 1, Nc, Nc] (float64 or float32),
                      where M = B*T collapsed, on an Nc×Nc grid.
    Output:
        next_coarse: Tensor of shape [M, 1, Nc, Nc] (float32),
                     the coarse field at the next time step for each frame.
    """
    # Remove channel dim for FFT → [M, Nc, Nc], ensure float64
    u_c = coarse_field.squeeze(1).to(torch.float64)

    # FFT in batch → [M, Nc, Nc], complex128
    u_hat = torch.fft.fft2(u_c)

    # Multiply by integrating factor exp(-nu * k^2 * dt), broadcasted
    u_hat = u_hat.to(exp_c.device) * exp_c.unsqueeze(0)

    # Add forcing correction: (f_hat_c * (1 - exp_c)) / (nu * k2_c) * f_coefficient
    numerator = f_hat_c * (1.0 - exp_c)                     # [Nc, Nc], complex128
    denom = (nu * k2_c)                                    # [Nc, Nc], float64
    forcing_term = (numerator / denom) * f_coefficient     # [Nc, Nc], complex128
    u_hat = u_hat + forcing_term.unsqueeze(0)              # broadcast to [M, Nc, Nc]

    # Inverse FFT back to real space → [M, Nc, Nc]
    u_next = torch.real(torch.fft.ifft2(u_hat))

    # Return with channel dim → [M, 1, Nc, Nc] as float32
    return u_next.unsqueeze(1).to(torch.float32)

# ----------------------------
# Data Loader: Create input-target pairs for NS (10-channel version)
# ----------------------------
def load_fine_data_all():
    """
    Loads fine-scale NS trajectories.
    Assumes the data is stored in a NumPy array of shape (num_samples, T, 10, H, W).
    For each trajectory, the first 10 time steps are used as input and the next 10 (shifted by one) as target.
    """
    # Replace the file path with your actual NS fine data path.
    fine_data = np.load(f'/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k.npy')
    inputs, targets = [], []
    for traj in fine_data:
        inputs.append(traj[:100])   
        targets.append(traj[1:101]) 
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets = torch.tensor(np.array(targets), dtype=torch.float32)
    return inputs, targets

# ----------------------------
# Fine-Tuning Training Loop (using NS time-stepping, 10->10 mapping)
# ----------------------------
def train_finetune():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load the fine-scale trajectories (each sample: [T, 10, H, W])
    print('loading data')
    inputs, targets = load_fine_data_all()
    print('data loaded')
    print(inputs.shape, targets.shape)
    # Normalize stats
    data_mean = inputs.mean(dim=(0,2,3), keepdim=True)
    data_std  = inputs.std(dim=(0,2,3), keepdim=True).clamp_min(1e-8)
    #print(data_mean, data_std)
    torch.save({
    'data_mean': data_mean,
    'data_std':  data_std
    }, f'./data/2d_heat_phase2_funet_stats_nu={nu}.pt')
    dataset = TensorDataset(inputs, targets)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    # Initialize the network for 100->100 mapping
    model = SuperResUNet(in_channels=100, final_scale=4).to(device)
    # Optionally, load pretrained weights if available.
    #model.load_state_dict(torch.load(f"/pscratch/sd/h/hbassi/models/2d_heat_FUnet_best_PS_FT_32to128_1k.pth"))
    model.load_state_dict(torch.load(f"/pscratch/sd/h/hbassi/models/best_funet_heat_nuNA_gauss1k.pth"))
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    criterion = nn.L1Loss()
    
    # Coarse grid parameters (assume domain [0,1]x[0,1])
    downsample_factor = 4
    Lx = 1.0
    Nx_fine = inputs.shape[-1]  # assuming square grid (e.g., 128)
    Nx_coarse = Nx_fine // downsample_factor
    dx_coarse = Lx / Nx_coarse
    dy_coarse = dx_coarse
    dt = 0.01      # time step for coarse evolution
    
    best_val_loss = float('inf')
    num_epochs = 2500
    
    for epoch in trange(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            fine_seq, target_seq = [b.to(device) for b in batch]  # each: [B, 10, 10, H, W]
            optimizer.zero_grad()
            # Use the first frame of the input sequence (10 channels) as current state.
            fine_frame = fine_seq[:, :, :, :]  # shape: [B, 10, H, W]
            # Project the fine field to the coarse grid.
            coarse_field = projection_operator(fine_frame, factor=downsample_factor)
            # Advance the coarse field one time step using NS time-stepping.
            coarse_field_tp = coarse_time_step_heat(coarse_field)
            # Superresolve the evolved coarse field.
            norm_in = (coarse_field_tp - data_mean.to(device)) / data_std.to(device)
            outputs = model(norm_in)
            pred_tp = outputs * data_std.to(device) + data_mean.to(device)
            #pred_tp = model(coarse_field_tp)
            # Use the corresponding target frame (first frame of target sequence).
            target_frame = target_seq[:, :, :, :]
            loss = criterion(pred_tp, target_frame)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                fine_seq, target_seq = [b.to(device) for b in batch]
                fine_frame = fine_seq[:, :, :, :]
                coarse_field = projection_operator(fine_frame, factor=downsample_factor)
                coarse_field_tp = coarse_time_step_heat(coarse_field)
                norm_in = (coarse_field_tp - data_mean.to(device)) / data_std.to(device)
                outputs = model(norm_in)
                pred_tp = outputs * data_std.to(device) + data_mean.to(device)
               # pred_tp = model(coarse_field_tp)
                target_frame = target_seq[:, :, :, :]
                loss = criterion(pred_tp, target_frame)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        if epoch % 10 == 0:
            log_message = f"Epoch {epoch} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}"
            print(log_message)
            logging.info(log_message)
        if epoch % 50 == 0:
         # Save periodic checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss.item(),
            }, f"/pscratch/sd/h/hbassi/models/fine_tuning_2d_heat_FUnet_epoch_{epoch}_smooth2.pth")
            
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"/pscratch/sd/h/hbassi/models/fine_tuning_best_FUnet_2d_heat_smooth2.pth")
            logging.info(f"Epoch {epoch} | New best validation loss: {avg_val_loss:.8f}. Model saved.")
    
if __name__ == "__main__":
    train_finetune()
