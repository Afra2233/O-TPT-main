#!/bin/bash
#SBATCH --job-name=draw_robust
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8


# robust
srun python analyze_calibration_plots.py \
  --baseline_npz /scratch/hpc/07/zhang303/O-TPT-main/analysis_npz/DTD_baseline_vitb32_tecoa_eps_1_pgd_eps0.00392156862745.npz \
  --otpt_npz /scratch/hpc/07/zhang303/O-TPT-main/analysis_npz/DTD_tpt_otpt_vitb32_tecoa_eps_1_pgd_eps0.00392156862745.npz \
  --output_dir /scratch/hpc/07/zhang303/O-TPT-main/analysis_plots/DTD_tecoa \
  --mode robust

# clean
# srun  python analyze_calibration_plots.py \
#   --baseline_npz /scratch/hpc/07/zhang303/O-TPT-main/analysis_npz/DTD_baseline_vitb32_tecoa_eps_1_pgd_eps0.00392156862745.npz \
#   --otpt_npz /scratch/hpc/07/zhang303/O-TPT-main/analysis_npz/DTD_tpt_otpt_vitb32_tecoa_eps_1_pgd_eps0.00392156862745.npz \
#   --output_dir /scratch/hpc/07/zhang303/O-TPT-main/analysis_plots/DTD_tecoa \
#   --mode clean