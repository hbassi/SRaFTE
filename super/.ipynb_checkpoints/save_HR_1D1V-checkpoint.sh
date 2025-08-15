#!/bin/bash
#SBATCH -A m4577
#SBATCH -C cpu             
#SBATCH -q regular
#SBATCH -t 01:00:00         # six‑hour wall‑clock limit
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32  # plenty of CPU threads for I/O & FFTs
#SBATCH -o %x-%j.out

# ── environment ────────────────────────────────────────────────
module load python          # loads system Python; swap for your stack

# save the data
python untitled.py
