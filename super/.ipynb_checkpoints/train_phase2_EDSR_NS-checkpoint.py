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
    filename=f'phase2_EDSR_training_NS_64to128_complex_nu={nu}_mode={k_cutoff_coarse}_with_forcing_no_norm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ShiftMean(nn.Module):
    # data: [t,c,h,w]
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        len_c = mean.shape[0]
        self.mean = torch.Tensor(mean).view(1, len_c, 1, 1)
        self.std = torch.Tensor(std).view(1, len_c, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif mode == 'add':
            return x * self.std.to(x.device) + self.mean.to(x.device)
        else:
            raise NotImplementedError

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class EDSR(nn.Module):
    def __init__(self, in_feats, n_feats, n_res_blocks, upscale_factor, mean, std, conv=default_conv):

        super(EDSR, self).__init__()

        n_resblocks = n_res_blocks # 16
        n_feats = n_feats # 64
        kernel_size = 3 
        scale = upscale_factor
        act = nn.ReLU(True)
        

        self.shift_mean = ShiftMean(torch.Tensor(mean), torch.Tensor(std)) 

        # define head module
        m_head = [conv(in_feats, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, in_feats, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.shift_mean(x, mode='sub')
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.shift_mean(x, mode='add')

        return x 
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
    dataset = TensorDataset(inputs, targets)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    #Initialize model
    data_mean = inputs.mean(dim=(0, 2, 3))  
    data_std  = inputs.std(dim=(0, 2, 3))   
    #import pdb; pdb.set_trace()
    model = EDSR(100, 64, 4, 4, data_mean, data_std).to(device)
    # Optionally, load pretrained weights if available.
    model.load_state_dict(torch.load(f'/pscratch/sd/h/hbassi/models/EDSR_NS_new_multi_traj_best_model_spectral_solver_32to128_nu={nu}_mode={k_cutoff_coarse}_forcing_no_norm.pth'))#['model_state_dict'])
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
            pred_tp = model(coarse_field_tp)
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
                pred_tp = model(coarse_field_tp)
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
            }, f"/pscratch/sd/h/hbassi/models/EDSR_fine_tuning_NS_multi_traj_checkpoint_epoch_{epoch}_spectral_solver_32to128_nu={nu}_mode={k_cutoff_fine}_with_forcing_no_norm.pth")
            
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"/pscratch/sd/h/hbassi/models/EDSR_fine_tuning_NS_32to128_nu={nu}_mode={k_cutoff_fine}_with_forcing_no_norm_best_model.pth")
            logging.info(f"Epoch {epoch} | New best validation loss: {avg_val_loss:.8f}. Model saved.")
    
if __name__ == "__main__":
    train_finetune()
