import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import logging

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    filename='fno_training_wave_32to128_phase2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width):
        """
        in_channels: number of time steps (features) per spatial location
        out_channels: number of output channels 
        modes1, modes2: number of Fourier modes to keep
        """
        super(FNO2d, self).__init__()
        self.width = width
        # Lift the input (here, in_channels = T) to a higher-dimensional feature space.
        self.fc0 = nn.Linear(in_channels, self.width)

        # Fourier layers and pointwise convolutions 
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)

        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)

        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        """
        x: input of shape [B, T, H, W]
        """
        # Permute to [B, H, W, T] so each spatial location has a feature vector of length T
        x = x.permute(0, 2, 3, 1)
        # Lift to higher-dimensional space
        x = self.fc0(x)
        # Permute to [B, width, H, W] for convolutional operations
        x = x.permute(0, 3, 1, 2)

        # Apply Fourier layers with local convolution
        x = self.conv0(x) + self.w0(x)
        x = nn.GELU()(x)
        #x = self.conv1(x) + self.w1(x)
        #x = nn.GELU()(x)
        #x = self.conv2(x) + self.w2(x)
        #x = nn.GELU()(x)
        x = self.conv3(x) + self.w3(x)

        # Permute back and project to output space
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

# Spectral convolution layer remains unchanged
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
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
# ----------------------------
# Data Loader: Create input-target pairs for U and V (each with 10 channels)
# ----------------------------
def load_fine_data_both():
    fine_data = np.load(f'/pscratch/sd/h/hbassi/wave_dataset_multi_sf_modes=4_kmax=4/u_fine.npy')
    inputs, targets = [], []
    for traj in fine_data:
        inputs.append(traj[:101])   
        targets.append(traj[1:102]) 
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets = torch.tensor(np.array(targets), dtype=torch.float32)
    return inputs, targets

# ----------------------------
# Fine-Tuning Training Loop (with coupled coarse evolution and two networks)
# ----------------------------
def train_finetune():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Load the fine-scale data for U and V (each sample: [10, H, W])
    inputs_u, targets_u = load_fine_data_both()
    # Create a combined dataset: each sample is (U_t, U_{t+Δt}, V_t, V_{t+Δt})
    dataset = TensorDataset(inputs_u, targets_u)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Initialize the two networks (for U and V) using the same pretrained architecture
    model_u = FNO2d(101, 101, 16, 16, 128).to(device)
    
    optimizer = optim.AdamW(list(model_u.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    num_epochs = 5000
    
    for epoch in trange(num_epochs):
        model_u.train()
        #model_v.train()
        train_loss = 0.0
        
        for batch in train_loader:
            fine_u_t, fine_u_tp = [b.to(device) for b in batch]
            #import pdb; pdb.set_trace()
            optimizer.zero_grad()
        
            pred_u_tp = model_u(fine_u_t)
            #pred_v_tp = model_v(coarse_v_tp)
            
            loss_u = criterion(pred_u_tp, fine_u_tp)
            #loss_v = criterion(pred_v_tp, fine_v_tp)
            loss = loss_u #+ loss_v
            
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model_u.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation loop
        model_u.eval()
        #model_v.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                fine_u_t, fine_u_tp, = [b.to(device) for b in batch]
                
                pred_u_tp = model_u(fine_u_t)
                #pred_v_tp = model_v(coarse_v_tp)
                loss_u = criterion(pred_u_tp, fine_u_tp)
                #loss_v = criterion(pred_v_tp, fine_v_tp)
                val_loss += (loss_u ).item()
        avg_val_loss = val_loss / len(val_loader)
        
        if epoch % 10 == 0:
            log_message = f"Epoch {epoch} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}"
            print(log_message)
            logging.info(log_message)
        
        # Save the best models based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model_u.state_dict(), f"/pscratch/sd/h/hbassi/models/2d_wave_FNO_best_v1_phase2_medium_sf=4.pth")
            #torch.save(model_v.state_dict(), "/pscratch/sd/h/hbassi/fine_tuning_GS_64to128_random_IC_tmax=5_sigma=5_numterms=20_FD_best_model_V.pth")
            logging.info(f"Epoch {epoch} | New best validation loss: {avg_val_loss:.8f}. Models saved.")
    
if __name__ == "__main__":
    train_finetune()
