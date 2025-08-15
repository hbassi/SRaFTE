
"""
aggregate_buneman_prune.py
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
• Keep only runs whose pickled arrays have the expected shapes:
      coarse (Ls6,6,6): (1252, 32, 32)
      fine   (Ls8,8,8): (1252,128,128)
• Retain the window 36 s ≤ t < 40 s  (900:1000 for dt = 0.04 s).
• Append the 100-frame float32 slices to lists, delete full arrays
  immediately to keep memory down.
• After all runs are read, stack each list → 4-D tensor and save each
  tensor as a separate .npy file.
• Generates a diagnostic plot (run 0 electrons, first & last frame).
"""
import argparse, pickle, re, gc, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
EXPECTED_C_SHAPE = (5002, 64, 64)
EXPECTED_F_SHAPE = (5002, 256, 256)

def _numeric_key(path: Path) -> int:
    m = re.match(r"run(\d+)", path.name)
    return int(m.group(1)) if m else sys.maxsize

def _load(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)

def _dt_from_name(name: str, default=0.04):
    m = re.search(r"_dt([0-9.]+)", name)
    return float(m.group(1)) if m else default


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir",
        default="/pscratch/sd/h/hbassi/data250506/VP1-buneman-SystemA_high-res_test/")
    ap.add_argument("--out_dir",
        default="/pscratch/sd/h/hbassi/")
    ap.add_argument("--stem", default="2d_vlasov_multi_traj")
    args = ap.parse_args()

    base   = Path(args.base_dir).expanduser()
    outdir = Path(args.out_dir ).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(base.glob("run*"), key=_numeric_key)
    if not run_dirs:
        raise RuntimeError("No run* folders found")

    # ------------- storage lists (will hold 100-frame slices) ----------
    fe_c_L, fi_c_L, fe_f_L, fi_f_L = [], [], [], []

    kept = 0
    for rd in run_dirs:
        try:
            fe_c_p = next(rd.glob("fe_*Ls6,6,6*.pkl"))
            fi_c_p = next(rd.glob("fi_*Ls6,6,6*.pkl"))
            fe_f_p = next(rd.glob("fe_*Ls8,8,8*.pkl"))
            fi_f_p = next(rd.glob("fi_*Ls8,8,8*.pkl"))
        except StopIteration:
            print(f"[warn] {rd.name}: missing file(s) – skipped")
            continue

        # ---------- shape filter ---------------------------------------
        fe_c_full = _load(fe_c_p)
        if fe_c_full.shape != EXPECTED_C_SHAPE:
            print(f"[warn] {rd.name}: coarse shape {fe_c_full.shape} ≠ {EXPECTED_C_SHAPE} – skipped")
            continue
        fi_c_full = _load(fi_c_p)
        fe_f_full = _load(fe_f_p)
        if fe_f_full.shape != EXPECTED_F_SHAPE:
            print(f"[warn] {rd.name}: fine shape {fe_f_full.shape} ≠ {EXPECTED_F_SHAPE} – skipped")
            continue
        fi_f_full = _load(fi_f_p)

        dt = _dt_from_name(fe_c_p.name, 0.01)
        start, end = int(round(22/dt)), int(round(24/dt)) + 1  
        slice_c = slice(start, end)                        

        # ---------- slice, cast to float32, append, free ---------------
        fe_c_L.append(fe_c_full[slice_c].astype(np.float32, copy=False))
        fi_c_L.append(fi_c_full[slice_c].astype(np.float32, copy=False))
        fe_f_L.append(fe_f_full[slice_c].astype(np.float32, copy=False))
        fi_f_L.append(fi_f_full[slice_c].astype(np.float32, copy=False))

        # free 1252-frame arrays ASAP
        del fe_c_full, fi_c_full, fe_f_full, fi_f_full
        gc.collect()

        kept += 1
        if kept % 100 == 0 or kept == 1:
            print(f"kept {kept} runs so far")

    if kept == 0:
        raise RuntimeError("No runs passed the shape filter!")

    # ------------- stack & save ---------------------------------------
    def save(arr_list, suffix):
        arr = np.stack(arr_list, axis=0)
        path = outdir / f"{args.stem}_{suffix}.npy"
        np.save(path, arr)
        print(f"saved {path.name} – shape {arr.shape}, dtype {arr.dtype}")
        return arr, path

    fe_c, _ = save(fe_c_L, "electron_coarse_64_fixed_timestep_buneman_phase1_test_data_no_ion")
    fi_c, _ = save(fi_c_L, "ion_coarse_64_fixed_timestep_buneman_phase1_test_data_no_ion")
    fe_f, _ = save(fe_f_L, "electron_fine_256_fixed_timestep_buneman_phase1_test_data_no_ion")
    fi_f, _ = save(fi_f_L, "ion_fine_256_fixed_timestep_buneman_phase1_test_data_no_ion")
    
    # ------------- quick diagnostic plot (run 0 electrons) -------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    im0 = ax[0,0].imshow(fe_c[0,0], aspect="auto"); ax[0,0].set_title("Coarse e⁻  t=36 s")
    im1 = ax[0,1].imshow(fe_f[0,0], aspect="auto"); ax[0,1].set_title("Fine   e⁻  t=36 s")
    im2 = ax[1,0].imshow(fe_c[0,-1], aspect="auto"); ax[1,0].set_title("Coarse e⁻  t≈40 s")
    im3 = ax[1,1].imshow(fe_f[0,-1], aspect="auto"); ax[1,1].set_title("Fine   e⁻  t≈40 s")
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    for im, a in zip((im0, im1, im2, im3), ax.flat):
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Run 0 electron density – coarse vs. fine")
    plot_path = outdir / "test_sample_trajectory_electron_coarse_fine.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print("\nDiagnostic plot →", plot_path)
    print(f"Total runs kept: {kept}")

if __name__ == "__main__":
    main()
