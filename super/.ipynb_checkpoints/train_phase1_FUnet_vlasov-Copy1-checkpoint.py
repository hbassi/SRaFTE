#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Super-resolution training with a wavelet-based importance-weight mask.

• A single-level 2-D Haar DWT is applied to each target frame.
• The high-frequency energy map F = LH² + HL² + HH² is up-sampled to
  the full resolution and converted to a mask a ∈ [α, β] on pixels
  whose energy exceeds the θ-quantile; elsewhere a = 1.
• The pixel-wise L1 loss is multiplied by this mask.
"""

import torch, math, pywt, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------------------
# ---------------------------  DATA  ----------------------------------
# ---------------------------------------------------------------------
def load_data():
    input_CG  = np.load('/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_coarse_32_data.npy')
    target_FG = np.load('/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_fine_128_data.npy')
    return (torch.tensor(input_CG , dtype=torch.float32),   # (N,11,Hc,Wc)
            torch.tensor(target_FG, dtype=torch.float32))   # (N,11,Hf,Wf)

# ---------------------------------------------------------------------
# -------------------  FOURIER FEATURE + FNO BLOCKS  ------------------
# ---------------------------------------------------------------------
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):                     # (B,H,W,2)
        proj = 2 * math.pi * torch.matmul(coords, self.B)   # (B,H,W,mapping_size)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)                        # (B,2*map,H,W)

def get_coord_grid(batch,h,w,device):
    xs,ys = torch.linspace(0,1,w,device=device), torch.linspace(0,1,h,device=device)
    gy,gx = torch.meshgrid(ys,xs,indexing='ij')
    return torch.stack((gx,gy),dim=-1).unsqueeze(0).repeat(batch,1,1,1)

class FourierLayer(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weight = nn.Parameter(torch.randn(in_ch,out_ch,modes1,modes2,dtype=torch.cfloat)
                                   /(in_ch*out_ch))

    def compl_mul2d(self, x, w):                   # (B,IC,H,W)×(IC,OC,H,W)
        return torch.einsum('bixy,ioxy->boxy', x, w)

    def forward(self,x):
        B,_,H,W = x.shape
        x_ft = torch.fft.rfft2(x)
        m1,m2 = min(self.modes1,H), min(self.modes2,x_ft.size(-1))
        out_ft = torch.zeros(B,self.weight.size(1),H,x_ft.size(-1),
                             dtype=torch.cfloat,device=x.device)
        out_ft[:,:,:m1,:m2] = self.compl_mul2d(x_ft[:,:,:m1,:m2], self.weight[:,:,:m1,:m2])
        return torch.fft.irfft2(out_ft,s=x.shape[-2:])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.GELU(),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.GELU())
    def forward(self,x): return self.block(x)

class PixelShuffleUpsample(nn.Module):
    def __init__(self,in_ch,out_ch,upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch*(upscale**2),3,padding=1)
        self.pix  = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self,x): return self.act(self.pix(self.conv(x)))

# -------------------------  U-NET  -----------------------------------
class SuperResUNet(nn.Module):
    def __init__(self,in_channels=11,lift_dim=64,mapping_size=64,
                 mapping_scale=5.0,final_scale=2):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2,mapping_size,mapping_scale)
        lifted = in_channels#+2*mapping_size
        self.lift = nn.Conv2d(lifted,lift_dim,1)

        self.enc1 = ConvBlock(lift_dim, lift_dim)
        self.enc2 = ConvBlock(lift_dim, lift_dim*2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim*2,lift_dim*2),
            FourierLayer(lift_dim*2,lift_dim*2,32,32),
            nn.GELU())

        self.up1  = PixelShuffleUpsample(lift_dim*2,lift_dim*2,upscale=1)
        self.dec2 = nn.Sequential(ConvBlock(lift_dim*4,lift_dim), nn.Dropout2d(0.15))
        self.up2  = PixelShuffleUpsample(lift_dim,lift_dim)
        self.dec1 = nn.Sequential(ConvBlock(lift_dim*2,lift_dim//2), nn.Dropout2d(0.15))
        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim//2,lift_dim//4,upscale=final_scale),
            ConvBlock(lift_dim//4,lift_dim//4))
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim//4,32,3,padding=1), nn.GELU(),
            nn.Conv2d(32,in_channels,3,padding=1))

    def forward(self,x):
        B,_,H,W = x.shape
        #coords = get_coord_grid(B,H,W,x.device)
        #x = torch.cat([x,self.fourier_mapping(coords)],dim=1)
        x = self.lift(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(e2)

        d2 = self.up1(b)
        d2 = self.dec2(torch.cat([d2,e2],1))
        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1,e1],1))
        d0 = self.dec0(d1)
        return self.out_head(d0)

# ---------------------------------------------------------------------
# ------------------  IMPORTANCE WEIGHT  MASK  ------------------------
# ---------------------------------------------------------------------
def build_importance_mask(target_batch, theta=0.8, alpha=0.5, beta=3.0):
    """
    target_batch : (B, C, H, W) torch tensor on *any* device
    returns       : (B, 1, H, W) tensor on the same device
    """
    B,C,H,W = target_batch.shape
    device  = target_batch.device
    masks   = []

    for b in range(B):
        # accumulate energy over channels in numpy for pywt speed
        energy = 0.0
        tb = target_batch[b].detach().cpu().numpy()  # (C,H,W)
        for c in range(C):
            _, (LH, HL, HH) = pywt.dwt2(tb[c], 'haar')
            energy += LH**2 + HL**2 + HH**2          # (H/2,W/2)

        # upsample to H×W
        energy = torch.tensor(energy, dtype=torch.float32)
        energy = torch.nn.functional.interpolate(
            energy.unsqueeze(0).unsqueeze(0), size=(H,W),
            mode='bilinear', align_corners=False).squeeze()

        # linear weights on high-energy pixels
        thresh = torch.quantile(energy, theta).item()
        mask   = torch.ones_like(energy)
        high   = energy > thresh
        if high.any():
            mask[high] = alpha + (beta-alpha)*(energy[high]-thresh) / (energy[high].max()-thresh + 1e-8)

        # normalise so ⟨a⟩ = 1
        mask /= mask.mean()
        masks.append(mask)

    masks = torch.stack(masks,0).unsqueeze(1).to(device)     # (B,1,H,W)
    return masks
# ================================================================
# Spectral loss  (magnitude difference in Fourier space)
# ================================================================
def spectral_loss(output, target):
    fft_o = torch.fft.rfft2(output)     # (B,C,H,W/2+1)
    fft_t = torch.fft.rfft2(target)
    return torch.mean(torch.abs(torch.abs(fft_o) - torch.abs(fft_t)))

# ---------------------------------------------------------------------
# --------------------------  TRAINING  -------------------------------
# ---------------------------------------------------------------------
def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -------- data --------
    x_cg, y_fg = load_data()
    print('inputs ', x_cg.shape,' targets ', y_fg.shape)
    upscale_factor = y_fg.shape[2] // x_cg.shape[2]
    assert y_fg.shape[3]//x_cg.shape[3]==upscale_factor

    data_mean = x_cg.mean((0,2,3),keepdim=True)
    data_std  = x_cg.std((0,2,3),keepdim=True).clamp_min(1e-8)
    torch.save({'data_mean':data_mean, 'data_std':data_std},
               './data/2d_vlasov_funet_phase1_stats.pt')

    ds = TensorDataset(x_cg, y_fg)
    ntr = int(0.9*len(ds))
    tr_ds, va_ds = random_split(ds,[ntr,len(ds)-ntr])
    tr_loader = DataLoader(tr_ds,batch_size=8,shuffle=True)
    va_loader = DataLoader(va_ds,batch_size=8)

    # -------- model / opt --------
    model = SuperResUNet(final_scale=upscale_factor).to(device)
    opt   = optim.AdamW(model.parameters(), lr=1e-3)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5000,eta_min=1e-6)
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=10, verbose=True)
    num_epochs = 5000

    L1  = nn.L1Loss(reduction='none')   # we will weight manually
    best_val = float('inf')

    for epoch in trange(num_epochs+1):
        model.train();  tr_loss = 0.0
        for x,y in tr_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()

            # normalise → predict → denormalise
            x_n = (x-data_mean.to(device))/data_std.to(device)
            y_hat = model(x_n)
            y_hat = y_hat*data_std.to(device)+data_mean.to(device)

            # -------- importance-weighted loss --------
            w_mask = build_importance_mask(y, theta=0.75, alpha=0.9, beta=1.2) # (B,1,H,W)
            #loss = (w*L1(y_hat,y)).mean()
            iw_loss  = (w_mask * (y_hat - y).abs()).mean()
            spec_loss = spectral_loss(y_hat, y) 
            λ_iw, λ_spec = 0.7, 0.3                                    
            loss = λ_iw * iw_loss + λ_spec * spec_loss
            tr_loss += loss.item()

            loss.backward()
            opt.step(); sch.step()

        # ---------------- validation every 100 epochs -------------
        if epoch % 100 == 0:
            model.eval();  val_loss = 0.0
            with torch.no_grad():
                for x,y in va_loader:
                    x,y = x.to(device), y.to(device)
                    x_n = (x-data_mean.to(device))/data_std.to(device)
                    y_hat = model(x_n)
                    y_hat = y_hat*data_std.to(device)+data_mean.to(device)
                    w_mask = build_importance_mask(y, theta=0.8, alpha=0.5, beta=3.0)
                    #val_loss += (w*L1(y_hat,y)).mean().item()
                    iw_loss  = (w_mask * (y_hat - y).abs()).mean()
                    spec_loss = spectral_loss(y_hat, y)
                    val_loss      += (0.7* iw_loss + 0.3 * spec_loss).item()
            

            val_loss /= len(va_loader)
            tr_loss  /= len(tr_loader)
            plateau.step(val_loss)
            print(f'Epoch {epoch:4d} | train {tr_loss:.8f} | val {val_loss:.8f}')

            # checkpointing (unchanged paths)
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optim_state_dict':opt.state_dict(),
                        'sched_state_dict':sch.state_dict(),
                        'train_loss':tr_loss,'val_loss':val_loss},
                       f'/pscratch/sd/h/hbassi/models/2d_vlasov_FUnet_ckpt_{epoch:04d}_IW.pth')

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           '/pscratch/sd/h/hbassi/models/2d_vlasov_FUnet_best_IW.pth')
                print(f'  ↳ new best loss ({best_val:.8f}) saved')

# ---------------------------------------------------------------------
if __name__ == '__main__':
    train_model()
