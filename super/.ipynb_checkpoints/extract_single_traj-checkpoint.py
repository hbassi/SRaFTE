#!/usr/bin/env python
# ===============================================================
#  extract_single_traj.py
#  ---------------------------------------------------------------
#  Extract the same trajectory index from both the *fine* (256×256)
#  and *coarse* (128×128) TDSE datasets and save them to separate
#  .npy files for faster re-use.
# ===============================================================
import argparse, numpy as np, os, sys

parser = argparse.ArgumentParser(
    description='Extract matching trajectories from 256×256 and 128×128 TDSE datasets.'
)
parser.add_argument('--src256', required=True,
                    help='Path to grid256_dataset.npy (big fine-grid file)')
parser.add_argument('--src128', required=True,
                    help='Path to grid128_dataset.npy (big coarse-grid file)')
parser.add_argument('--traj',   type=int, default=0,
                    help='Trajectory index to extract (default 0)')
parser.add_argument('--dst256', required=True,
                    help='Output .npy filename for the extracted fine-grid trajectory')
parser.add_argument('--dst128', required=True,
                    help='Output .npy filename for the extracted coarse-grid trajectory')
args = parser.parse_args()

# --------------------------------------------------------------- fine grid
print('[256] Memory-mapping fine-grid file header …')
fine_big = np.load(args.src256, mmap_mode='r')
n_traj_fine = fine_big.shape[0]
print(f'[256] File contains {n_traj_fine} trajectories.')

# --------------------------------------------------------------- coarse grid
print('[128] Memory-mapping coarse-grid file header …')
coarse_big = np.load(args.src128, mmap_mode='r')
n_traj_coarse = coarse_big.shape[0]
print(f'[128] File contains {n_traj_coarse} trajectories.')

# consistency check --------------------------------------------------------
if n_traj_fine != n_traj_coarse:
    print('ERROR: Fine and coarse files have different numbers of trajectories!',
          file=sys.stderr)
    sys.exit(1)

if not (0 <= args.traj < n_traj_fine):
    raise IndexError('Requested trajectory index out of range.')

# ---------------------------------------------------------------- extraction
print(f'Extracting trajectory index {args.traj} …')
traj_fine   = np.array(fine_big[args.traj])    # (T,256,256) complex64
traj_coarse = np.array(coarse_big[args.traj])  # (T,128,128) complex64

print(f'[256] Extracted shape {traj_fine.shape}, dtype {traj_fine.dtype}')
print(f'[128] Extracted shape {traj_coarse.shape}, dtype {traj_coarse.dtype}')

# ---------------------------------------------------------------- saving
for out_path, traj in ((args.dst256, traj_fine), (args.dst128, traj_coarse)):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, traj)
    print(f'Saved to {out_path}')

print('Done.')
