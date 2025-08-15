#!/bin/bash
#SBATCH -A m1027
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:10:00      # less wait time than 48 h if you can
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32  # speed up FFTs; adjust to taste
#SBATCH -o %x-%j.out

module load python
python NS_data_generator.py
