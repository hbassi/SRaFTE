import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import trange
import math
import torch.fft                          as fft
import torch.nn.functional as F
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
# 3 ▸ 2‑D Spectral Convolution layer (FNO core)
# ===============================================================
class SpectralConv2d(nn.Module):
    """Fourier layer that keeps only the lowest modes coefficients."""
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
    """
    Plain FNO backbone followed by a bilinear upsample to the fine
    resolution.  No coordinate embedding is used to keep parity
    with the minimal baseline UNet.
    """
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
               './data/fluids_FNO_phase1_stats.pt')

    dataset = TensorDataset(input_tensor, target_tensor)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    print(f'Train={len(train_ds)}  Val={len(val_ds)}  |  upscale ×{upscale_factor}')

    # ---------------- model ----------------
    model = FNO2dSR(in_ch=100, modes1=16, modes2=16,
                              upscale_factor=4).to(device)
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
            }, f'/pscratch/sd/h/hbassi/models/fluids_FNO_ckpt_{epoch:04d}_32to128.pth')

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           '/pscratch/sd/h/hbassi/models/fluids_FNO_best_PS_FT_32to128.pth')
                print(f'  ↳ new best loss ({best_val:.8f}) saved')

            np.save('./logs/fluids_FNO_train_PS_FT_32to128.npy', np.asarray(tr_hist))
            np.save('./logs/fluids_FNO_val_PS_FT_32to128.npy',   np.asarray(va_hist))

# ----------------------------------------------------------------
if __name__ == '__main__':
    train_model()
