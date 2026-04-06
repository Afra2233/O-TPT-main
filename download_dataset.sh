#!/bin/bash
#SBATCH --job-name=download_datatset
#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
module add anaconda3/2022.05
source activate otpt

srun python download_dataset.py