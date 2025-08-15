# ──────────────────────────────────────────────────────────────
#  Phase‑2 autoregressive roll‑out 
# ──────────────────────────────────────────────────────────────
# ───────────────────────── imports ────────────────────────────
import os, re, math, argparse, numpy as np, matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from numpy.fft import fftfreq, fft2, ifft2, rfft2
from sklearn.decomposition import PCA
import torch, torch.nn.functional as F, models                       
from tqdm import trange
torch.set_float32_matmul_precision("high")

# ---------- runtime / data paths ----------
#PHASE2_CKPT = "/pscratch/sd/h/hbassi/models/NS_best_{model}_phase2.pth"
PHASE2_CKPT = "/pscratch/sd/h/hbassi/models/NS_best_{model}_nu1e-4_k7.5_phase2.pth"
MODELS      = ["funet", "unet", "edsr", "fno"]      # evaluate all 4
AR_NAME     = "fnoar" 
UPSCALE, NU, DT = 4, 1e-4, 0.01
WINDOW = 100                                          # sliding context
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------- load fine test data  (shape B,101,H,W) ----------
#fine_np = np.load(f"/pscratch/sd/h/hbassi/NavierStokes_test_traj_fine_nu={1e-4}_mode={7.5}_no_dealias_32to128_without_forcing_new.npy")[:]   
fine_np = np.load('/pscratch/sd/h/hbassi/NavierStokes_fine_256_nu0.0001_k7.5_test_data.npy')[7]
# ---------- load the matching coarse test data (shape B,101,Hc,Wc) ----------
coarse_np = np.load('/pscratch/sd/h/hbassi/'
                    'NavierStokes_coarse_64_nu0.0001_k7.5_test_data.npy')[7]
coarse_t  = torch.tensor(coarse_np, dtype=torch.float32, device=device).unsqueeze(0)

fine_t  = torch.tensor(fine_np, dtype=torch.float32, device=device).unsqueeze(0)
B,Tp1,H,W = fine_t.shape
# ---------- mean / std from first 100 frames ----------
#mean = fine_t[:,:100].mean((0,2,3),keepdim=True)
#std  = fine_t[:,:100].std ((0,2,3),keepdim=True).clamp_min(1e-8)
stats = torch.load('./data/phase2_funet_stats_nu1e-4_k7.5.pt')
mean = stats['mean']
std = stats['std']
Hc = H // UPSCALE                      # coarse‑grid resolution
dx_coarse = dy_coarse = 1.0 / Hc       # same spacing used in training
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def coarse_time_step_NS(u, dt=DT, dx=dx_coarse, dy=dy_coarse, nu=NU):
    B, C, Hc, Wc = u.shape
    k = torch.fft.fftfreq(Hc, d=dx, device=u.device) * 2 * math.pi
    KX, KY = torch.meshgrid(k, k, indexing='ij')
    k2 = KX**2 + KY**2
    k2[0, 0] = 1e-10
    KX, KY, k2 = [t.unsqueeze(0).unsqueeze(0) for t in (KX, KY, k2)]

    w_hat  = torch.fft.fft2(u)               # vorticity in Fourier space
    psi_hat = -w_hat / k2                    # stream‑function
    u_x =  torch.fft.ifft2(1j * KY * psi_hat).real
    u_y = -torch.fft.ifft2(1j * KX * psi_hat).real
    dw_dx =  torch.fft.ifft2(1j * KX * w_hat).real
    dw_dy =  torch.fft.ifft2(1j * KY * w_hat).real
    adv   = u_x * dw_dx + u_y * dw_dy        
    lap_w = torch.fft.ifft2(-k2 * w_hat).real

    # diagonal sinusoidal forcing
    x = torch.linspace(0., 1., Hc, device=u.device)
    y = torch.linspace(0., 1., Wc, device=u.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    forcing = 0.025 * (torch.sin(2 * math.pi * (X + Y))
                       + torch.cos(2 * math.pi * (X + Y)))
    forcing = forcing.unsqueeze(0).unsqueeze(0)            # (1,1,Hc,Wc)

    return u + dt * (-adv + nu * lap_w + forcing)
# --------------------------------------------------------------------------

# ---------- projection utility ----------
def project(fine):           # fine (H,W) → coarse (H/4,W/4)
    return fine[..., ::UPSCALE, ::UPSCALE]

# ---------- build & load all Phase‑2 nets ----------
def build(name,in_ch=WINDOW):
    if   name=="funet": return models.SuperResUNet(in_channels=in_ch,final_scale=UPSCALE)
    elif name=="unet":  return models.UNetSR(in_ch,upscale_factor=UPSCALE)
    elif name=="edsr":  return models.EDSR(in_ch,128,16,UPSCALE,
                                           mean=np.zeros(in_ch,np.float32),
                                           std =np.ones(in_ch,np.float32))
    elif name=="fno":   return models.FNO2dSR(in_ch,modes1=16,modes2=16,upscale_factor=UPSCALE)
    else: raise ValueError
nets={}
for m in MODELS:
    net=build(m).to(device)
    net.load_state_dict(torch.load(PHASE2_CKPT.format(model=m),
                                   map_location="cpu"))
    net.eval(); nets[m]=net
    print(f"Loaded Phase‑2 {m.upper()}.")
print("Loading autoregressive FNO ...")
ar_net = models.FNO2dAR(100, 100, 16, 16, 64).to(device)
ar_ckpt = "/pscratch/sd/h/hbassi/FNO_fine_tuning_NS_64to256_nu=0.0001_mode=7.5_best_model.pth"
ar_net.load_state_dict(torch.load(ar_ckpt, map_location=device))
ar_net.eval()
print(f"Loaded AR-FNO weights from {ar_ckpt}")
#nets["fnoar"] = ar_net
# --------------------------------------------------------------

# ---------- allocate roll‑out arrays (B,101,H,W) ----------
roll  = {m: torch.zeros_like(fine_t) for m in MODELS}
roll[AR_NAME] = torch.zeros_like(fine_t)        # ← NEW

# ---------- autoregressive sliding‑window loop -------------
for b in trange(B, desc="Trajectories"):
    # seed windows (shape (10,Hc,Wc) )
    fine_window = fine_t[b, :WINDOW].clone()                 # (10,H,W)
    coarse_window = torch.stack([project(fr) for fr in fine_window]).clone()

    # copy first WINDOW fine frames into outputs
    for m in MODELS: roll[m][b, :WINDOW] = fine_window
    #ups[b,:WINDOW] = fine_window

    for step in range(WINDOW, 1000):
        # ------ coarse Euler predictor on 10‑channel window ------
        coarse_input = project(fine_window).unsqueeze(0)#coarse_window.unsqueeze(0)            # (1,10,Hc,Wc)
        updated_coarse = coarse_time_step_NS(coarse_input).squeeze(0)
        new_coarse_state = updated_coarse[-1]                # last channel
        coarse_window = torch.cat([coarse_window[1:], new_coarse_state.unsqueeze(0)],0)

        

        #------ SR with every network --------------------------
        # ------ AR-FNO inference on the fine grid -----------------
        with torch.no_grad():
            # window_u shape: (1,100,256,256)
            if step == WINDOW:                   # first call → seed window
                window_u = fine_window.unsqueeze(0)        # fine_window is (100,H,W)
            outputs_u = ar_net(window_u)                   # (1,100,H,W)
            final_time = outputs_u.squeeze(0)[-1]          # (H,W)
            roll[AR_NAME][b, step] = final_time
            # slide the window
            window_u = torch.cat([window_u[:,1:], final_time.unsqueeze(0).unsqueeze(0)], dim=1)
        coarse_net_in = coarse_window.unsqueeze(0)           # (1,10,Hc,Wc)
        norm_in = (coarse_net_in - mean)/std
        for m,net in nets.items():
            with torch.no_grad():
                out = net(norm_in.float())*std + mean        # (1,10,H,W)
            next_fine = out[:, -1]                           # immediate next
            roll[m][b,step]=next_fine.squeeze()
  
        

        # shift fine_window as context 
        fine_window = torch.cat([fine_window[1:], 
                                 roll[m][b,step].unsqueeze(0)],0)
# ---------- build bicubic baseline from ground‑truth coarse -------------
with torch.no_grad():
    ups_t = F.interpolate(coarse_t, scale_factor=UPSCALE,
                          mode="bicubic", align_corners=False)  # (1,101,H,W)

# ---------- convert to numpy for downstream metric / plots --------------
cutoff = 701
fine_np  = fine_t.cpu().numpy()[:, :cutoff]
nets['fnoar'] = ar_net
pred_np  = {m: roll[m].cpu().numpy()[:, :cutoff] for m in MODELS + ['fnoar']}
ups_np   = ups_t.cpu().numpy()[:, :cutoff]           # ← new baseline

# ────────────────── metric helpers ──────────────────
def tex_sci(x, prec=2):
    base, exp = f"{x:.{prec}e}".split("e")
    return rf"$ {base} \times 10^{{{int(exp):d}}} $"

def spectral_mse_2d(a, b):
    return np.mean((np.abs(rfft2(a)) - np.abs(rfft2(b))) ** 2)

def ms_ssim_2d(a, b):
    return ssim(a, b, multiscale=True, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False,
                data_range=b.max() - b.min())

def psnr(pred, ref):
    mse = np.mean((pred - ref) ** 2)
    return np.inf if mse == 0 else 10.0 * math.log10(
        (ref.max() - ref.min()) ** 2 / mse)

def div_mse_from_vorticity(w):
    H, W = w.shape
    ky, kx = np.meshgrid(fftfreq(H)*2*math.pi, fftfreq(W)*2*math.pi)
    ksq = kx**2 + ky**2; ksq[0,0] = 1.0
    psi = np.real(ifft2(fft2(w) / -ksq)); psi -= psi.mean()
    dy, dx = 2*math.pi/H, 2*math.pi/W
    u = -(np.roll(psi,-1,0) - np.roll(psi,1,0)) / (2*dy)
    v =  (np.roll(psi,-1,1) - np.roll(psi,1,1)) / (2*dx)
    div = (np.roll(u,-1,1) - np.roll(u,1,1)) / (2*dx) + \
          (np.roll(v,-1,0) - np.roll(v,1,0)) / (2*dy)
    return np.mean(div**2)

# ─────────────────── per‑snapshot figures ─────────────────────
fig_dir = "./figures/ns_phase2_rollout_test123"
os.makedirs(fig_dir, exist_ok=True)
plot_steps = range(0, cutoff, 100)          # 0,100,200,…

#COLS      = ["GT", "FUNET", "FNO", "FNOAR", "UNET", "EDSR", "UPS"]
COLS      = ["GT", "UPS", "FNOAR", "FUNET", "EDSR", "FNO", "UNET"]
label_map = {"GT":"Ground truth", "FUNET":"FUnet", "FNO":"FNO-SR",
             "FNOAR":"FNO-AR", "UNET":"U-Net", "EDSR":"EDSR", "UPS":"Upsampled"}


for t in plot_steps:
    vmin, vmax = fine_np[0,t].min(), fine_np[0,t].max()
    fig, axes = plt.subplots(1, len(COLS), figsize=(4*len(COLS), 4))
    for ax, col in zip(axes, COLS):
        if col == "GT":        img = fine_np[0,t]
        elif col == "UPS":     img = ups_np [0,t]
        elif col == "FNOAR":  img = pred_np["fnoar"][0, t]

        else:                  img = pred_np[col.lower()][0,t]

        ax.imshow(img, cmap="jet")
        ax.set_title(label_map[col], fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # error inset (skip GT)
        if col != "GT":
            err = np.linalg.norm(img - fine_np[0,t]) / \
                  np.linalg.norm(fine_np[0,t])
            ax.text(0.02, 0.94, rf"$\bf L_2={err:.2e}$",
                    transform=ax.transAxes, fontsize=9,
                    color="white", ha="left", va="top",
                    bbox=dict(fc="black", alpha=0.6, pad=0.25))

    # fig.colorbar(axes[0].images[0], ax=axes, fraction=0.02, pad=0.03)\
    #    .set_label("$w$", fontsize=12)
    fig.savefig(os.path.join(fig_dir, f"phase2_t{t:04d}.pdf"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"Saved snapshot figures to {fig_dir}")

# ─────────────────── global metric averages ───────────────────
metrics = {m: dict(L2=0, SSIM=0, MSS=0, SPEC=0, CORR=0, PSNR=0, PHY=0)
           for m in ["FUNET", "FNO", "UNET", "EDSR", "UPS", "FNOAR"]}

total_frames = fine_np.shape[0] * fine_np.shape[1]

# ─────────────────── global metric averages ───────────────────
METRICS = ["L2","SSIM","MSS","SPEC","CORR","PSNR","PHY"]
metrics = {m: {k:0. for k in METRICS} for m in ["FUNET","FNO","FNOAR","UNET","EDSR","UPS"]}
sq      = {m: {k:0. for k in METRICS} for m in ["FUNET","FNO","FNOAR", "UNET","EDSR","UPS"]}

total = fine_np.shape[0] * fine_np.shape[1] - 2 * fine_np.shape[0]  # skip edges

for b in range(fine_np.shape[0]):
    for t in range(1, fine_np.shape[1]-1):
        ref_prev, ref_curr, ref_next = fine_np[b,t-1], fine_np[b,t], fine_np[b,t+1]
        for tag, arr in [("UPS",ups_np),("FUNET",pred_np["funet"]),
                         ("FNO",pred_np["fno"]),("FNOAR",pred_np["fnoar"]),("UNET",pred_np["unet"]),
                         ("EDSR",pred_np["edsr"])]:
            pred_prev, pred_curr, pred_next = arr[b,t-1], arr[b,t], arr[b,t+1]

            vals = dict(
                L2   = np.linalg.norm(pred_curr-ref_curr) /
                       (np.linalg.norm(ref_curr)+1e-8),
                SSIM = ssim(pred_curr, ref_curr, data_range=ref_curr.ptp()),
                MSS  = ms_ssim_2d(pred_curr, ref_curr),
                SPEC = spectral_mse_2d(pred_curr, ref_curr),
                CORR = pearsonr(ref_curr.ravel(), pred_curr.ravel())[0],
                PSNR = psnr(pred_curr, ref_curr)
            )
            for k,v in vals.items():
                metrics[tag][k] += v
                sq[tag][k]      += v**2

# ─────────────────────── print mean ± std ─────────────────────
print("\n─── Dataset-wide mean ± std ─────────────────────────────")
for tag in ["FUNET","FNO","FNOAR", "UNET","EDSR","UPS"]:
    out = []
    for k in METRICS:
        mu  = metrics[tag][k] / total
        var = max(sq[tag][k] / total - mu**2, 0.0)
        sd  = math.sqrt(var)
        if k=="PSNR":
            out.append(f"{mu:5.2f}±{sd:4.2f}dB")
        elif k in ("L2","SPEC","PHY"):
            out.append(f"{mu:.2e}±{sd:.1e}")
        else:
            out.append(f"{mu:.4f}±{sd:.4f}")
    print(f"{tag:6s} | " + " | ".join(out))
# ─────────────────── time-resolved rel-L2 curves ───────────────────
print("Building time-resolved relative L2 curves …")
def rel_l2_series(pred, ref):
    num = np.linalg.norm(pred - ref, axis=(2, 3))
    den = np.linalg.norm(ref, axis=(2, 3)) + 1e-8
    return (num / den).mean(axis=0)   # average across batch

# Collect series for all models
series = {}
series["UPS"]    = rel_l2_series(ups_np,              fine_np)
series["FUNET"]  = rel_l2_series(pred_np["funet"],    fine_np)
series["FNO-SR"] = rel_l2_series(pred_np["fno"],      fine_np)
series["FNO-AR"] = rel_l2_series(pred_np["fnoar"],    fine_np)
series["UNET"]   = rel_l2_series(pred_np["unet"],     fine_np)
series["EDSR"]   = rel_l2_series(pred_np["edsr"],     fine_np)

from matplotlib.lines import Line2D

# Consistent display names (match earlier figure)
display_name = {
    "UPS":    "Upsampled",
    "FNO-AR": "FNO-AR",
    "FUNET":  "FUnet",
    "EDSR":   "EDSR",
    "FNO-SR": "FNO-SR",
    "UNET":   "U-Net",
}

# Style map (SRaFTE solid + color; baselines dashed + gray)
srafte    = ["FUNET", "FNO-SR", "UNET", "EDSR"]
baselines = ["UPS", "FNO-AR"]

cycle_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map  = {tag: cycle_cols[i % len(cycle_cols)] for i, tag in enumerate(srafte)}
style = {
    "FUNET":  dict(color=color_map["FUNET"],  lw=2.8, ls="-"),
    "FNO-SR": dict(color=color_map["FNO-SR"], lw=2.8, ls="-"),
    "UNET":   dict(color=color_map["UNET"],   lw=2.8, ls="-"),
    "EDSR":   dict(color=color_map["EDSR"],   lw=2.8, ls="-"),
    "UPS":    dict(color="0.55", lw=1.8, ls="--", alpha=0.9),
    "FNO-AR": dict(color="0.75", lw=1.8, ls="--", alpha=0.9),
}

# Plot (SRaFTE first so they render on top)
t = np.arange(fine_np.shape[1])
plt.figure(figsize=(8, 4.5))
for tag in srafte + baselines:
    y = series[tag][100:]
    plt.semilogy(t[100:], y, label=display_name[tag],
             zorder=3 if tag in srafte else 2, **style[tag])

plt.xlabel("timestep", fontweight="bold")
plt.ylabel(r"relative $L_2$ error", fontweight="bold")
plt.title("Phase 2 rollout error accumulation", fontweight="bold")
plt.grid(True, alpha=0.3)
plt.xticks(fontweight="bold"); plt.yticks(fontweight="bold")
plt.ylim([1e-2, 7e1])

# Group legend (bold) + per-model legend (bold) with matching labels
ax = plt.gca()
group_handles = [
    Line2D([0],[0], color="k",   lw=2.8, ls="-",  label="SRaFTE models"),
    Line2D([0],[0], color="0.6", lw=1.8, ls="--", label="Baselines (non-SRaFTE)"),
]
leg_group  = ax.legend(handles=group_handles, loc="upper left",
                       frameon=True, prop={'weight':'bold', 'size':14})
ax.add_artist(leg_group)

# Per-model legend (ordered to mirror earlier figure names)
legend_order = ["Upsampled", "FNO-AR", "FUnet", "EDSR", "FNO-SR", "U-Net"]
handles, labels = ax.get_legend_handles_labels()
lbl2hdl = {lbl: h for h, lbl in zip(handles, labels) if lbl in legend_order}
ordered = [lbl2hdl[lbl] for lbl in legend_order]
ax.legend(ordered, legend_order, ncol=3, frameon=True, loc="lower right",
          prop={'weight':'bold', 'size':14})

plt.tight_layout()
out_path = os.path.join(fig_dir, "relL2_vs_time.pdf")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
