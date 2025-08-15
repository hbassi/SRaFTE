import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import logging
import math

nu = 1e-4
k_cutoff_coarse = 7.5
k_cutoff_fine = 7.5

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    filename=f'training_NS_32to128_complex_nu={nu}_mode={k_cutoff_coarse}_with_forcing_no_norm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#
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
# ----------------------------
# Projection Operator: Fine -> Coarse (applied channel–wise)
# ----------------------------
def projection_operator(fine_data, factor=2):
    """
    Downsamples the input spatially by the given factor.
    fine_data: Tensor of shape [B, 10, H, W]
    """
    return fine_data[..., ::factor, ::factor]

# ----------------------------
# NS Time-Stepping Function (channel-wise NS update)
# ----------------------------
def coarse_time_step_NS(coarse_field, dt, dx, dy, nu):
    """
    Advances the coarse 10-channel field one time step using the NS vorticity update.
    The update is applied independently on each channel.
    
    coarse_field: Tensor of shape [B, 10, H, W]
    """
    B, C, H, W = coarse_field.shape
    N = H  # assume square grid
    L = 1.0
    # Create wave number grid (using spacing dx)
    k = torch.fft.fftfreq(N, d=dx) * 2 * math.pi  # shape: (N,)
    KX, KY = torch.meshgrid(k, k, indexing='ij')
    KX = KX.to(coarse_field.device)
    KY = KY.to(coarse_field.device)
    ksq = KX**2 + KY**2
    ksq[0, 0] = 1e-10  # avoid division by zero
    # Expand dimensions for broadcasting: shape [1, 1, N, N]
    ksq_ = ksq.unsqueeze(0).unsqueeze(0)
    x = torch.linspace(0, L, N)
    y = torch.linspace(0, L, N)
    X, Y = torch.meshgrid(x, y)
    X = X.to('cuda:1')
    Y = Y.to('cuda:1')
    forcing = 0.025 * (torch.sin(2 * torch.pi * (X + Y)) + torch.cos(2 * torch.pi * (X + Y)))
    # Compute Fourier transform for each channel
    field_hat = torch.fft.fft2(coarse_field)  # shape [B, 10, H, W]
    # Compute streamfunction for each channel: psi_hat = -field_hat/ksq
    psi_hat = -field_hat / ksq_
    psi = torch.fft.ifft2(psi_hat).real  # streamfunction, shape [B, 10, H, W]
    
    # Compute velocity components via spectral differentiation for each channel:
    KX_ = KX.unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
    KY_ = KY.unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
    u = torch.fft.ifft2(1j * KY_ * psi_hat).real
    v = -torch.fft.ifft2(1j * KX_ * psi_hat).real
    
    # Compute spatial derivatives of the field
    dfield_dx = torch.fft.ifft2(1j * KX_ * field_hat).real
    dfield_dy = torch.fft.ifft2(1j * KY_ * field_hat).real
    
    # Nonlinear advection term (computed channel-wise)
    nonlinear = u * dfield_dx + v * dfield_dy
    
    # Compute Laplacian in Fourier space then invert FFT
    lap_field = torch.fft.ifft2(-ksq_ * field_hat).real
    
    # Explicit Euler update
    field_new = coarse_field + dt * (-nonlinear + nu * lap_field + forcing)
    
    # # De-aliasing using the 2/3 rule
    # cutoff = N // 3
    # mask = ((KX.abs() < cutoff) & (KY.abs() < cutoff)).to(coarse_field.device)
    # mask = mask.unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
    # field_new_hat = torch.fft.fft2(field_new)
    # field_new_hat = field_new_hat * mask
    # field_new = torch.fft.ifft2(field_new_hat).real
    
    return field_new

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
    fine_data = np.load(f'/pscratch/sd/h/hbassi/NavierStokes_fine_128_nu={nu}_kcutoff={k_cutoff_fine}_with_forcing_no_norm.npy')
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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Load the fine-scale trajectories (each sample: [T, 10, H, W])
    print('loading data')
    inputs, targets = load_fine_data_all()
    print('data loaded')
    print(inputs.shape, targets.shape)
    # Normalize stats
    data_mean = inputs.mean(dim=(0,2,3), keepdim=True)
    data_std  = inputs.std(dim=(0,2,3), keepdim=True).clamp_min(1e-8)
    print(data_mean, data_std)
    torch.save({
    'data_mean': data_mean,
    'data_std':  data_std
    }, f'phase2_stats_nu={nu}.pt')
    dataset = TensorDataset(inputs, targets)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Initialize the network for 100->100 mapping
    model = SuperResUNet(in_channels=100).to(device)
    # Optionally, load pretrained weights if available.
    model.load_state_dict(torch.load(f"/pscratch/sd/h/hbassi/models/best_PSFUnet_NS_nu={nu}.pth"))
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
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
            coarse_field_tp = coarse_time_step_NS(coarse_field, dt, dx_coarse, dy_coarse, nu)
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
                coarse_field_tp = coarse_time_step_NS(coarse_field, dt, dx_coarse, dy_coarse, nu)
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
        if epoch % 100 == 0:
         # Save periodic checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss.item(),
            }, f"/pscratch/sd/h/hbassi/models/fine_tuning_PSFUnet_NS_nu={nu}_epoch_{epoch}.pth")
            
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"/pscratch/sd/h/hbassi/models/fine_tuning_best_PSFUnet_NS_nu={nu}.pth")
            logging.info(f"Epoch {epoch} | New best validation loss: {avg_val_loss:.8f}. Model saved.")
    
if __name__ == "__main__":
    train_finetune()
