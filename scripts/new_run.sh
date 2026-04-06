#!/bin/bash
#SBATCH --job-name=otpt_tecoa
#SBATCH -p gpu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/hpc/07/zhang303/O-TPT-main/%x-%j.out
#SBATCH -e /scratch/hpc/07/zhang303/O-TPT-main/%x-%j.err

module add anaconda3/2022.05
source activate otpt

data_root='/scratch/hpc/07/zhang303/O-TPT-main/data'

cd /scratch/hpc/07/zhang303/O-TPT-main || exit 1

testsets=$1
if [ -z "${testsets}" ]; then
  echo "Usage: sbatch $0 <testset>"
  echo "Example: sbatch $0 Food101"
  exit 1
fi

arch='ViT-B/32'
bs=64
ctx_init='a_photo_of_a'

run_type='tpt_otpt'
lambda_term=18

clip_ckpt='/scratch/hpc/07/zhang303/O-TPT-main/checkpoints/vitb32_tecoa_eps_1.pt'
csv_loc="/scratch/hpc/07/zhang303/O-TPT-main/log/test_otpt_tecoa_${testsets}_pgd.csv"

attack='pgd'
attack_eps=0.00392156862745
attack_alpha=0.00098039215686
attack_steps=10
attack_restarts=1
eval_mode='both'

gpu_id=0
workers=8

mkdir -p /scratch/hpc/07/zhang303/O-TPT-main/log

echo "=================================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Dataset root: ${data_root}"
echo "Test set: ${testsets}"
echo "Arch: ${arch}"
echo "Checkpoint: ${clip_ckpt}"
echo "Attack: ${attack}"
echo "Eps: ${attack_eps}"
echo "Alpha: ${attack_alpha}"
echo "Steps: ${attack_steps}"
echo "=================================================="

python ./new_otpt_classification.py ${data_root} \
  --test_sets ${testsets} \
  --csv_log ${csv_loc} \
  --dataset_mode test \
  -a ${arch} \
  -j ${workers} \
  -b ${bs} \
  --gpu ${gpu_id} \
  --ctx_init ${ctx_init} \
  --run_type ${run_type} \
  --clip_ckpt ${clip_ckpt} \
  --attack ${attack} \
  --attack_eps ${attack_eps} \
  --attack_alpha ${attack_alpha} \
  --attack_steps ${attack_steps} \
  --attack_restarts ${attack_restarts} \
  --eval_mode ${eval_mode}\
  --tpt \
  --lambda_term ${lambda_term} 

exit_code=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "Exit code: ${exit_code}"
echo "=================================================="

exit ${exit_code}