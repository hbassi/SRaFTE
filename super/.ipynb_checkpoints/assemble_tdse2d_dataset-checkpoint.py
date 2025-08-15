#!/usr/bin/env python3
# ===============================================================
#  stack_tdse_datasets.py      (v2 – skips missing trajectories)
# ===============================================================

import os
import sys
import numpy as np
from pathlib import Path


def discover_trajectories(grid_dir: Path):
    """
    Return a list of traj* subfolders that actually contain trajectory.npy,
    sorted lexicographically.  Missing ones (e.g. traj0213) are silently skipped.
    """
    traj_dirs = []
    for d in sorted(grid_dir.iterdir()):
        if d.is_dir() and d.name.startswith("traj"):
            if (d / "trajectory.npy").is_file():
                traj_dirs.append(d)
            else:
                print(f"⚠  {d.name} has no trajectory.npy – skipped", flush=True)
    return traj_dirs


def stack_trajectories(grid_dir: Path, out_path: Path) -> None:
    traj_dirs = discover_trajectories(grid_dir)
    if not traj_dirs:
        raise RuntimeError(f"No valid traj* folders found in {grid_dir}")

    # discover shape / dtype from the first GOOD trajectory
    sample = np.load(traj_dirs[0] / "trajectory.npy", mmap_mode="r")
    traj_shape = sample.shape          # (Nt, Nx, Ny)
    dtype      = sample.dtype
    n_traj     = len(traj_dirs)

    print(f"{grid_dir.name}: {n_traj} trajectories • each {traj_shape} • dtype {dtype}")

    # prepare output mem-mapped array
    out_array = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(n_traj, *traj_shape)
    )

    # stream-copy trajectories
    for idx, tdir in enumerate(traj_dirs):
        data = np.load(tdir / "trajectory.npy", mmap_mode="r")

        if data.shape != traj_shape:
            raise ValueError(f"Shape mismatch in {tdir}: {data.shape} vs {traj_shape}")

        out_array[idx] = data
        if (idx + 1) % 50 == 0 or idx == n_traj - 1:
            print(f"  • processed {idx + 1:4d}/{n_traj}", flush=True)

    out_array.flush()
    print(f"✓ saved → {out_path}\n")


def main() -> None:
    base = Path("/pscratch/sd/h/hbassi/tdse2d_test_64to128/")
    for grid in ["grid256", "grid64"]:
        stack_trajectories(base / grid, base / f"{grid}_test_dataset.npy")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
