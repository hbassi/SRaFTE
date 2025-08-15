#!/usr/bin/env python3
"""
evaluate_all_models.py
──────────────────────
Visualise Phase-1 predictions of the three trained super-resolution
models (FNO, EDSR, FUnet) on a *single* Vlasov two-stream test case.
For every 10th frame it shows

    • coarse input (32×32)
    • FNO, EDSR, FUnet predictions (128×128)
    • true fine-grid field  (128×128)

and insets the relative L2 error of each prediction.

Change the two CSV paths below if you want to test a different case.
"""

import os, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch

# ───────────────────────────────────────────────────────────────
# 1 ▸ helpers
# ───────────────────────────────────────────────────────────────
def csv_to_tensor(path, t_shift=200, dtype=np.float32):
    """
    CSV rows are [t, j, k, value].  Returns a (T,J,K) numpy tensor.
    """
    data   = np.loadtxt(path, delimiter=',', dtype=dtype)
    idx    = data[:, :3].astype(int)
    idx[:, 0] -= t_shift                       # shift time axis
    vals   = data[:, 3]

    shape  = tuple(idx.max(axis=0) + 1)        # (T,J,K)
    tensor = np.zeros(shape, dtype=dtype)
    tensor[tuple(idx.T)] = vals
    return tensor


def rel_L2(a, b):
    """‖a − b‖₂ / ‖b‖₂, both numpy arrays."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def load_data():
    data = np.load('/pscratch/sd/h/hbassi/fluid_data/fluid_dynamics_datasets.npz')
    input_CG  = data['coarse']
    target_FG = data['fine']
    #import pdb; pdb.set_trace()
    input_tensor  = torch.tensor(input_CG[42, :100, :-1, :-1],  dtype=torch.float32)      
    target_tensor = torch.tensor(target_FG[42, :100, :-1, :-1], dtype=torch.float32)      
    return input_tensor, target_tensor
# ───────────────────────────────────────────────────────────────
# 2 ▸ load the single test trajectory
# ───────────────────────────────────────────────────────────────
#FG_PATH = '/pscratch/sd/h/hbassi/fluid_data/vorticity_LV=7_Re=90.000000_ratio=1.500000_angle=0.616850_CB.dat'
#CG_PATH = '/pscratch/sd/h/hbassi/fluid_data/vorticity_LV=5_Re=90.000000_ratio=1.500000_angle=0.616850_CB.dat'

#rho_fine_all = csv_to_tensor(FG_PATH)           # (T_full, 512, 512) originally
#rho_coarse_all = csv_to_tensor(CG_PATH)         # (T_full, 128, 128)
rho_coarse_all, rho_fine_all = load_data()
# crop to the training sizes used earlier
T, Hc, Wc = 100, 32, 32
_, Hf, Wf = 100, 128, 128
rho_cg   = rho_coarse_all[:T, :Hc, :Wc]         # (100,32,32)
rho_fg   = rho_fine_all  [:T, :Hf, :Wf]         # (100,128,128)
scale    = Hf // Hc                             # 4

# torch tensors for one-shot prediction
input_tensor  = rho_cg.unsqueeze(0)#torch.from_numpy(rho_cg[None])  # (1,100,32,32)
#rho_fg = torch.from_numpy(rho_fg)
device        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ───────────────────────────────────────────────────────────────
# 3 ▸ instantiate and load the three models
#     (FNO2d, EDSR, SuperResUNet must already be in your PYTHONPATH)
# ───────────────────────────────────────────────────────────────
def load_model(model_name, in_ch, upscale, device):
    if model_name == 'FNO':
        from models import FNO2dSR                    
        model = FNO2dSR(in_ch, modes1=16, modes2=16, upscale_factor=4)
    elif model_name == 'EDSR':
        from models import EDSR                      
        stats = torch.load(f'./data/fluids_EDSR_phase1_stats.pt', map_location=device)
        model = EDSR(in_ch, n_feats=128, n_res_blocks=16,
                     upscale_factor=upscale,
                     mean=stats['data_mean'], std=stats['data_std'])
    elif model_name == 'FUnet':
        from models import SuperResUNet
        model = SuperResUNet(in_channels=in_ch, final_scale=upscale)
    else:
        raise ValueError(f'Unknown model {model_name}')

    wt_path = f'/pscratch/sd/h/hbassi/models/fluids_{model_name}_best_PS_FT_32to128.pth'
    state   = torch.load(wt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


models = {}
for name in ('FNO', 'EDSR', 'FUnet'):
    models[name] = load_model(name, in_ch=100, upscale=scale, device=device)

with torch.no_grad():
    pred_results = {}
    inp = input_tensor.to(device) 
    for name, net in models.items():
        if name == 'FUnet':
            lname = name.lower()
        else:
            lname = name
        stats = torch.load(f'./data/fluids_{lname}_phase1_stats.pt', map_location=device)
        
        out = net((inp - stats['data_mean'])/stats['data_std']).cpu()[0]            # (100,128,128)
        pred_results[name] = (out*stats['data_std'].cpu() + stats['data_mean'].cpu()).numpy().squeeze(0)


# ───────────────────────────────────────────────────────────────
# 4 ▸ plotting loop
# ───────────────────────────────────────────────────────────────
case_tag = "Re=90, r=1.5, θ=0.617"
Nt = 100

for t in range(0, Nt, 10):
    cg       = rho_cg[t]                           # 32×32
    cg_up    = zoom(cg, scale, order=3)            # 128×128
    fg       = rho_fg[0].numpy()                           # 128×128

    rel_errors = {name: rel_L2(pred_results[name][t], fg)
                  for name in models}
    up_error   = rel_L2(cg_up, fg)

    # ----------------------------------------------------------
    cols = ['CG (32×32)', 'CG↑ (bicubic)', 'FNO', 'EDSR', 'FUnet', 'FG (128×128)']
    #import pdb; pdb.set_trace()
    panels = [cg, cg_up,
              pred_results['FNO'][t],
              pred_results['EDSR'][t],
              pred_results['FUnet'][t],
              fg]

    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3))
    vmin, vmax = fg.min(), fg.max()
    #import pdb; pdb.set_trace()
    for ax, im, title in zip(axes, panels, cols):
        #import pdb; pdb.set_trace()
        h = ax.imshow(im, origin='lower', cmap='viridis')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # inset relative L2 for each prediction
    axes[1].text(0.02, 0.95, f"rel $L_2$={up_error:.2e}", transform=axes[1].transAxes,
                 fontsize=7, va='top', ha='left', color='w', bbox=dict(fc='k', alpha=0.5))
    for i, name in enumerate(('FNO', 'EDSR', 'FUnet'), start=2):
        axes[i].text(0.02, 0.95, f"rel $L_2$={rel_errors[name]:.2e}",
                     transform=axes[i].transAxes,
                     fontsize=7, va='top', ha='left', color='w',
                     bbox=dict(fc='k', alpha=0.5))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(h, cax=cbar_ax, label=r'$\omega_z$')

    fig.suptitle(f"{case_tag}   •   t = {t*0.01:.2f}", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f'fluids_test_traj_t={t}.pdf')
    plt.show()

    # console log
    print(f"[t={t:03d}]  CG↑  {up_error:.3e} | "
          f"FNO  {rel_errors['FNO']:.3e} | "
          f"EDSR {rel_errors['EDSR']:.3e} | "
          f"FUnet {rel_errors['FUnet']:.3e}")
