#!/usr/bin/env python
"""
Evaluate Phase-1 Navier–Stokes models on a held-out test set.

Outputs:
  out_dir/vis_t000.pdf, vis_t010.pdf, …   (per-frame comparisons)
  out_dir/energy_spectra.pdf              (radial spectra at final frame)
"""

# ───────────────────────── imports ─────────────────────────────
import argparse, os
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
import models

torch.set_float32_matmul_precision("high")

# ───────── checkpoint / stats templates ───────────────────────
CKPT_TPL  = "/pscratch/sd/h/hbassi/models/best_{model}_{tag}.pth"
STATS_TPL = "/pscratch/sd/h/hbassi/models/stats_{model}_{tag}.pt"
TAG       = "2d-wave_high_freq_sf=8"        # adjust if needed

# ───────── radial energy spectrum util ────────────────────────
def radial_energy_spectrum(u):
    N      = u.shape[0]
    u_hat  = np.fft.fftshift(np.fft.fft2(u))
    energy = np.abs(u_hat)**2 / N**2
    kx     = np.fft.fftshift(np.fft.fftfreq(N)) * N
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    kr     = np.sqrt(KX**2 + KY**2).astype(int)
    E      = np.bincount(kr.ravel(), energy.ravel(),
                         minlength=kr.max()+1)
    return np.arange(len(E)), E

# ───────────────────────── main ───────────────────────────────
def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- load test data ----------
    fine   = np.load(args.fine)                 # (B,100,Hf,Wf)
    coarse = np.load(args.coarse)               # (B,100,Hc,Wc)
    idx    = args.index
    frames = np.arange(0, 100, 10)              # 0,10,…,90

    # ------------------------------------------------------------------
    # 🔄  Compute per-channel mean / std *from the coarse data itself*  🔄
    # ------------------------------------------------------------------
    coarse_torch = torch.tensor(coarse[:, :100],   # (B,100,Hc,Wc)
                                dtype=torch.float32, device=dev)
    mean = coarse_torch.mean(dim=(0, 2, 3), keepdim=False).view(1, 100, 1, 1)
    std  = coarse_torch.std (dim=(0, 2, 3), keepdim=False).view(1, 100, 1, 1)
    std  = std.clamp_min(1e-6)  # avoid division by zero

    # ---------- input tensor (fp32) ----------
    x_in = torch.tensor(coarse[idx, :100], dtype=torch.float32,
                        device=dev).unsqueeze(0)              # (1,100,Hc,Wc)
    normed = (x_in - mean) / std                              # fp32

    # ---------- load models (device only, keep dtype) ----------
    MODELS = ["funet", "unet", "edsr", "fno"]
    nets   = {}
    for m in MODELS:
        ckpt = torch.load(CKPT_TPL.format(model=m, tag=TAG), map_location="cpu")
        net = (
            models.FNO2dSR(100, modes1=8, modes2=8, upscale_factor=8)   if m == "fno"  else
            models.UNetSR(100, upscale_factor=8)                        if m == "unet" else
            models.EDSR(100, 128, 16, 8, np.zeros(100, np.float32), np.ones(100, np.float32)) if m == "edsr" else
            models.SuperResUNet(in_channels=100, final_scale=8)         # FUNet
        )
        net.load_state_dict(ckpt)
        net.to(dev).eval()
        nets[m] = net

    # ---------- run each model once ----------
    pred_bank = {}
    with torch.no_grad():
        for m, net in nets.items():
            out = net(normed.float()) * std + mean
            pred_bank[m] = out.cpu().squeeze().numpy()   # (100,Hf,Wf)

    # ---------- make output directory ----------
    os.makedirs(args.out_dir, exist_ok=True)
    import matplotlib.ticker as mtick
    from matplotlib import gridspec
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold"
    })

    # helper ───────────────────────────────────────────────────────
    def add_ticks(ax, N=128, show_y=False):
        xt = [0, N//2, N-1]
        ax.set_xticks(xt)
        ax.set_xticklabels([str(v) for v in xt], fontsize=8)
        ax.set_xlabel("x", fontsize=10)
        if show_y:
            ax.set_yticks(xt)
            ax.set_yticklabels([str(v) for v in xt], fontsize=8)
            ax.set_ylabel("y", fontsize=10)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

    # ---------- loop over selected frames ----------
    DISPLAY = {
        "GT_COARSE": "Ground truth (coarse)",
        "GT_FINE":   "Ground truth (fine)",
        "UPS":       "Upsampled",
        "FUNET":     "FUnet",
        "EDSR":      "EDSR",
        "FNO":       "FNO-SR",
        "UNET":      "U-Net",
    }

    # Column order (exactly as requested)
    COLS = ["GT_COARSE", "GT_FINE", "UPS", "FUNET", "EDSR", "FNO", "UNET"]
    n_img = len(COLS)

    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    # infer sizes for ticks
    Hc = coarse.shape[-1]
    Hf = fine.shape[-1]

    for t in frames:
        fig = plt.figure(figsize=(3 * n_img + 1, 3))
        gs  = gridspec.GridSpec(1, n_img + 1,  # +1 for shared colorbar
                                width_ratios=[1] * n_img + [0.05],
                                wspace=0.20)

        gt_fine   = fine[idx, t]               # (Hf,Wf)
        gt_coarse = coarse[idx, t]             # (Hc,Wc)

        # upsample coarse → fine (bicubic) for the UPS panel
        up = F.interpolate(
            torch.tensor(gt_coarse[None, None], dtype=torch.float32),
            scale_factor=8, mode="bicubic", align_corners=False
        ).squeeze().numpy()

        # shared vmin/vmax from GT-Fine (models will use this)
        vmin, vmax = gt_fine.min(), gt_fine.max()

        # build per-column image + scale policy
        images = {
            "GT_COARSE": dict(img=gt_coarse, vmin=None, vmax=None, N=Hc, show_y=True),
            "GT_FINE":   dict(img=gt_fine,   vmin=vmin, vmax=vmax, N=Hf, show_y=False),
            "UPS":       dict(img=up,        vmin=None, vmax=None, N=Hf, show_y=False),
            "FUNET":     dict(img=pred_bank["funet"][t], vmin=vmin, vmax=vmax, N=Hf, show_y=False),
            "EDSR":      dict(img=pred_bank["edsr"][t],  vmin=vmin, vmax=vmax, N=Hf, show_y=False),
            "FNO":       dict(img=pred_bank["fno"][t],   vmin=vmin, vmax=vmax, N=Hf, show_y=False),
            "UNET":      dict(img=pred_bank["unet"][t],  vmin=vmin, vmax=vmax, N=Hf, show_y=False),
        }

        im_ref = None  # keep the imshow for GT-Fine to anchor the colorbar

        for j, key in enumerate(COLS):
            ax = fig.add_subplot(gs[0, j])
            spec = images[key]
            im = ax.imshow(spec["img"], cmap="turbo",
                           vmin=spec["vmin"], vmax=spec["vmax"])
            ax.set_title(DISPLAY[key])
            add_ticks(ax, N=spec["N"], show_y=spec["show_y"])
            # relative L2 w.r.t. GT-Fine (skip GT-Coarse and GT-Fine)
            if key not in ("GT_COARSE", "GT_FINE"):
                err = rel_l2(spec["img"], gt_fine)
                ax.text(0.02, 0.93, rf"$\bf L_2={err:.4f}$",
                        color="black", transform=ax.transAxes,
                        fontsize=13, va="top", ha="left")
            if key == "GT_FINE":
                im_ref = im

        # shared colour-bar (far right) for GT-Fine scale
        #cax = fig.add_subplot(gs[0, -1])
        #cb  = fig.colorbar(im_ref, cax=cax)
        #cb.ax.tick_params(labelsize=8)

        fig.savefig(Path(args.out_dir) / f"2d-wave_vis_t{t:03d}.pdf",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- energy spectra at final frame ----------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    t_last = frames[-1]
    fig, ax = plt.subplots(figsize=(6, 4))

    # helper to plot and record each line
    def add_line(x, y, *args, **kwargs):
        (ln,) = ax.loglog(x, y, *args, **kwargs)
        return ln

    lines = []

    # 1) Upsampled (bicubic) and neural models first (lower z-order)
    k, E = radial_energy_spectrum(up)  # 'up' from last frame drawn above
    ln_up = add_line(k, E + 1e-20, color="gray", label="Upsampled", zorder=3)
    lines.append(ln_up)

    for tag, label in [("funet", "FUnet"),
                       ("edsr",  "EDSR"),
                       ("fno",   "FNO-SR"),
                       ("unet",  "U-Net")]:
        k, E = radial_energy_spectrum(pred_bank[tag][t_last])
        ln = add_line(k, E + 1e-20, label=label, zorder=3)
        lines.append(ln)

    # 2) GT-Fine on top (highest z-order)
    k, E = radial_energy_spectrum(fine[idx, t_last])
    ln_gt = add_line(k, E + 1e-20, "k--", label="Ground truth", zorder=10, linewidth=2.0)
    lines.append(ln_gt)

    ax.set_xlabel(r"$k$", fontweight="bold")
    ax.set_ylabel(r"$E(k)$", fontweight="bold")

    # bold legend
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0, prop={"weight": "bold", "size": 9})

    # inset (k ≤ 8) — EXACT logic matched
    axins = inset_axes(ax, width="48%", height="40%", loc="lower left", borderpad=2.75)
    for ln in lines:  # GT drawn last → on top
        kdata, Edata = ln.get_xdata(), ln.get_ydata()
        m = kdata <= 8
        axins.loglog(kdata[m], Edata[m],
                     linestyle=ln.get_linestyle(),
                     color=ln.get_color(),
                     linewidth=ln.get_linewidth())

    ticks = [1, 2, 3, 4, 5, 6, 7, 8]
    axins.set_xlim(1, 8)
    axins.set_ylim(5e0, 7e1)
    axins.set_xticks(ticks)
    axins.xaxis.set_major_locator(mtick.FixedLocator(ticks))
    axins.xaxis.set_major_formatter(mtick.FixedFormatter([str(t) for t in ticks]))
    axins.tick_params(labelsize=8)

    fig.savefig(Path(args.out_dir) / "2d-wave_energy_spectra.pdf",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # reload arrays (kept to mirror your exact logic flow)
    fine   = np.load(args.fine)
    coarse = np.load(args.coarse)

    # ───────────────────────── global L² errors ───────────────────
    print("\n▶  Computing dataset-wide relative L² …")
    B, T = fine.shape[:2]  # B trajectories, 100 frames each

    # collect every frame’s error so we can get mean ± std
    err_vals = {k: [] for k in ["upsampled", "funet", "edsr", "fno", "unet"]}

    with torch.no_grad():
        for b in range(B):
            xin    = torch.tensor(coarse[b, :100], dtype=torch.float32, device=dev).unsqueeze(0)
            normed = (xin - mean) / std

            # predictions
            preds = {}
            for m, net in nets.items():
                out      = net(normed) * std + mean  # (1,100,Hf,Wf)
                preds[m] = out.cpu().squeeze().numpy()

            gt_frames = fine[b, :100]  # (100,Hf,Wf)

            # bicubic once
            up_frames = F.interpolate(
                torch.tensor(coarse[b, :100, None, :, :], dtype=torch.float32),
                scale_factor=8, mode="bicubic", align_corners=False
            ).squeeze().numpy()  # (100,Hf,Wf)

            # accumulate per-frame rel-L2
            err_vals["upsampled"].extend(
                np.linalg.norm(up_frames - gt_frames, axis=(1, 2)) /
                (np.linalg.norm(gt_frames, axis=(1, 2)) + 1e-12)
            )
            for m in ["funet", "edsr", "fno", "unet"]:
                err_vals[m].extend(
                    np.linalg.norm(preds[m] - gt_frames, axis=(1, 2)) /
                    (np.linalg.norm(gt_frames, axis=(1, 2)) + 1e-12)
                )

    # ───────────────────────── report  ────────────────────────────
    name_map = {
        "upsampled": "Upsampled",
        "funet":     "FUnet",
        "edsr":      "EDSR",
        "fno":       "FNO-SR",
        "unet":      "U-Net",
    }
    print("\n●  Relative L² error over the entire test set (mean ± std)")
    for key in ["upsampled", "funet", "edsr", "fno", "unet"]:
        vals = np.asarray(err_vals[key], dtype=np.float64)
        mu   = vals.mean()
        sig  = vals.std(ddof=1)  # sample std-dev
        print(f"   {name_map[key]:12s}: {mu:.4e}  ±  {sig:.4e}")

# ───────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate Phase-1 models on wave test set")
    ap.add_argument("--coarse", default="/pscratch/sd/h/hbassi/"
                    f"wave_dataset_multi_sf_modes=10_kmax=7/u_coarse_sf={8}_test.npy")
    ap.add_argument("--fine",   default="/pscratch/sd/h/hbassi/"
                    "wave_dataset_multi_sf_modes=10_kmax=7/u_fine_test.npy")
    ap.add_argument("--index",  type=int, default=0,
                    help="which test trajectory to visualise")
    ap.add_argument("--out_dir", default="./eval_figs_2d-wave_sf=8", type=str)
    main(ap.parse_args())
