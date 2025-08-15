#!/bin/bash
#SBATCH -A m4577
#SBATCH -C gpu             # request GPU nodes on Perlmutter
#SBATCH -q regular
#SBATCH -t 06:00:00         # six‑hour wall‑clock limit
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1   # one GPU
#SBATCH --cpus-per-task=32  # plenty of CPU threads for I/O & FFTs
#SBATCH -o %x-%j.out

# ── environment ────────────────────────────────────────────────
module load python          # loads system Python; swap for your stack

# ── launch training ────────────────────────────────────────────
# Pick the model you want:  funet | unet | edsr | fno
python train_phase2_2d-heat_all.py --model unet
