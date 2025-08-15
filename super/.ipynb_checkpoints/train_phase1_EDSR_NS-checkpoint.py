import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
from tqdm import trange
torch.set_float32_matmul_precision('high')
#torch.manual_seed(seed=999)
import math
# ----------------------------
# Data Loading and Preparation
# ----------------------------
nu = 1e-4
k_cutoff_coarse = 7.5
k_cutoff_fine = 7.5
def load_data():
   
    input_CG = np.load(f'/pscratch/sd/h/hbassi/NavierStokes_coarse_32_nu={nu}_kcutoff={k_cutoff_coarse}_with_forcing_no_norm.npy')[:, :100]
    target_FG = np.load(f'/pscratch/sd/h/hbassi/NavierStokes_fine_128_nu={nu}_kcutoff={k_cutoff_fine}_with_forcing_no_norm.npy')[:, :100]
    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_CG, dtype=torch.float32)
    target_tensor = torch.tensor(target_FG, dtype=torch.float32)
    return input_tensor, target_tensor

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
# Training Setup
# ----------------------------
def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    # Load and prepare data
    print('Loading data')
    input_tensor, target_tensor = load_data()
    print('Data loaded')
    print(f'Shape of inputs: {input_tensor.shape}')
    print(f'Shape of targets: {target_tensor.shape}')
    dataset = TensorDataset(input_tensor, target_tensor)
    train_ds, val_ds = random_split(dataset, [950, 50])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")
    
    # Initialize model
    data_mean = input_tensor.mean(dim=(0, 2, 3))  # This will give you a tensor of shape (T,)
    data_std  = input_tensor.std(dim=(0, 2, 3))   # Also a tensor of shape (T,)
    #import pdb; pdb.set_trace()
    model = EDSR(100, 64, 4, 4, data_mean, data_std).to(device)
    print('Compiling model')
    #model = torch.compile(model)
    print('Model compiled')
    num_epochs = 5500
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.L1Loss()
    lambda_spectral = 0.1 
    training_losses = []
    validation_losses = []
    # Training loop
    for epoch in trange(num_epochs + 1):
        model.train()
        training_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            #import pdb; pdb.set_trace()
            # Compute the standard L1 loss
            loss_l1 = criterion(outputs, targets)
            # Compute the spectral loss
            #loss_spec = spectral_loss(outputs, targets)
            # Combine the losses
            loss = loss_l1 #+ lambda_spectral * loss_spec
            training_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Validation
        if epoch % 100 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            avg_val_loss = val_loss/len(val_loader)
            print(f"Epoch {epoch} | Train Loss: {training_loss/len(train_loader):.8f} | Val Loss: {avg_val_loss:.8f}")
            training_losses.append(training_loss/len(train_loader))
            validation_losses.append(avg_val_loss)
       
            # Save periodic checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss.item(),
                'val_loss': avg_val_loss,
            }, f"/pscratch/sd/h/hbassi/models/EDSR_NS_multi_traj_checkpoint_epoch_{epoch}_spectral_solver_32to128_nu={nu}_mode={k_cutoff_coarse}_forcing_no_norm.pth")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"/pscratch/sd/h/hbassi/models/EDSR_NS_new_multi_traj_best_model_spectral_solver_32to128_nu={nu}_mode={k_cutoff_coarse}_forcing_no_norm.pth")
                print(f"New best model saved with val loss {avg_val_loss:.8f}")
        with open(f'./logs/EDSR_NS_new_multi_traj_training_spectral_solver_32to128_nu={nu}_mode={k_cutoff_coarse}_forcing_no_norm.npy', 'wb') as f:
            np.save(f, training_losses)
        f.close()
        with open(f'./logs/EDSR_NS_new_multi_traj_validation_spectral_solver_32to128_nu={nu}_mode={k_cutoff_coarse}_forcing_no_norm.npy', 'wb') as f:
            np.save(f, validation_losses)
        f.close()

# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    train_model()