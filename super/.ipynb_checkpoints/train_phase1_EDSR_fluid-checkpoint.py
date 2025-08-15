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
    data = np.load('/pscratch/sd/h/hbassi/fluid_data/fluid_dynamics_datasets.npz')
    input_CG  = data['coarse']
    target_FG = data['fine']
    #import pdb; pdb.set_trace()
    input_tensor  = torch.tensor(input_CG[:, :100, :-1, :-1],  dtype=torch.float32)      
    target_tensor = torch.tensor(target_FG[:, :100, :-1, :-1], dtype=torch.float32)      
    return input_tensor, target_tensor

# ===============================================================
# 3 ▸ EDSR components
# ===============================================================
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias)

class ShiftMean(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        c = mean.shape[1]
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
    data_mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True).to(device)   # (1,11,1,1)
    #import pdb; pdb.set_trace()
    data_std  = input_tensor.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8).to(device)
    torch.save({'data_mean': data_mean, 'data_std': data_std},
               './data/fluids_EDSR_phase1_stats.pt')

    dataset = TensorDataset(input_tensor, target_tensor)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    print(f'Train={len(train_ds)}  Val={len(val_ds)}  |  upscale ×{upscale_factor}')

    # ---------------- model ----------------
    model = EDSR(in_ch=100, n_feats=128, n_res_blocks=16,
                              upscale_factor=4,
                              mean=data_mean.to(device), std=data_std.to(device)).to(device)
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
            }, f'/pscratch/sd/h/hbassi/models/fluids_EDSR_ckpt_{epoch:04d}_32to128.pth')

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           '/pscratch/sd/h/hbassi/models/fluids_EDSR_best_PS_FT_32to128.pth')
                print(f'  ↳ new best loss ({best_val:.8f}) saved')

            np.save('./logs/fluids_EDSR_train_PS_FT_32to128.npy', np.asarray(tr_hist))
            np.save('./logs/fluids_EDSR_val_PS_FT_32to128.npy',   np.asarray(va_hist))

# ----------------------------------------------------------------
if __name__ == '__main__':
    train_model()
