#!/bin/bash
#SBATCH --job-name=rtpt
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

# Usage:
#   sbatch run_rtpt_compare.sh Food101 rtpt
#   sbatch run_rtpt_compare.sh Food101 rtpt_dir

testsets=$1
method=$2

if [ -z "${testsets}" ]; then
  echo "Usage: sbatch $0 <testset> <method>"
  echo "Example: sbatch $0 Food101 rtpt"
  echo "Example: sbatch $0 Food101 rtpt_dir"
  exit 1
fi

if [ -z "${method}" ]; then
  echo "Usage: sbatch $0 <testset> <method>"
  echo "method must be one of: rtpt, rtpt_dir"
  exit 1
fi

if [ "${method}" != "rtpt" ] && [ "${method}" != "rtpt_dir" ]; then
  echo "Unknown method: ${method}"
  echo "method must be one of: rtpt, rtpt_dir"
  exit 1
fi

arch='ViT-B/32'
bs=64
ctx_init='a_photo_of_a'
if [ "${method}" = "rtpt" ]; then
  run_type='rtpt'
else
  run_type='rtpt_dir'
fi


attack='pgd'
attack_eps=0.00392156862745
attack_alpha=0.00098039215686
attack_steps=10
attack_restarts=1
eval_mode='both'

gpu_id=0
workers=8

rtpt_neighbor_k=20
rtpt_tau=0.01

lambda_tpt=1.0
lambda_dir=1.0
dir_temp=1.0
alpha_offset=1.0

log_dir="/scratch/hpc/07/zhang303/O-TPT-main/log"
npz_dir="/scratch/hpc/07/zhang303/O-TPT-main/analysis_npz"

mkdir -p ${log_dir}
mkdir -p ${npz_dir}

if [ "${method}" = "rtpt" ]; then
  csv_loc="${log_dir}/test_rtpt_openai_${testsets}_pgd.csv"
else
  csv_loc="${log_dir}/test_rtpt_dir_openai_${testsets}_pgd.csv"
fi

echo "=================================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Dataset root: ${data_root}"
echo "Python file: rptp_dirichlet.py"
echo "Test set: ${testsets}"
echo "Method: ${method}"
echo "Run type: ${run_type}"
echo "Arch: ${arch}"
echo "Checkpoint: original OpenAI CLIP"
echo "Batch size: ${bs}"
echo "Attack: ${attack}"
echo "Eps: ${attack_eps}"
echo "Alpha: ${attack_alpha}"
echo "Steps: ${attack_steps}"
echo "Restarts: ${attack_restarts}"
echo "Eval mode: ${eval_mode}"
echo "R-TPT neighbor k: ${rtpt_neighbor_k}"
echo "R-TPT tau: ${rtpt_tau}"
echo "CSV log: ${csv_loc}"
echo "NPZ dir: ${npz_dir}"

if [ "${method}" = "rtpt_dir" ]; then
  echo "lambda_tpt: ${lambda_tpt}"
  echo "lambda_dir: ${lambda_dir}"
  echo "dir_temp: ${dir_temp}"
  echo "alpha_offset: ${alpha_offset}"
fi

echo "=================================================="

if [ "${method}" = "rtpt" ]; then

  python ./rptp_dirichlet.py ${data_root} \
    --test_sets ${testsets} \
    --csv_log ${csv_loc} \
    --dataset_mode test \
    -a ${arch} \
    -j ${workers} \
    -b ${bs} \
    --gpu ${gpu_id} \
    --ctx_init ${ctx_init} \
    --run_type ${run_type} \
    --attack ${attack} \
    --attack_eps ${attack_eps} \
    --attack_alpha ${attack_alpha} \
    --attack_steps ${attack_steps} \
    --attack_restarts ${attack_restarts} \
    --eval_mode ${eval_mode} \
    --save_npz \
    --npz_dir ${npz_dir} \
    --tpt \
    --rtpt_neighbor_k ${rtpt_neighbor_k} \
    --rtpt_tau ${rtpt_tau}

else

  python ./rptp_dirichlet.py ${data_root} \
    --test_sets ${testsets} \
    --csv_log ${csv_loc} \
    --dataset_mode test \
    -a ${arch} \
    -j ${workers} \
    -b ${bs} \
    --gpu ${gpu_id} \
    --ctx_init ${ctx_init} \
    --run_type ${run_type} \
    --attack ${attack} \
    --attack_eps ${attack_eps} \
    --attack_alpha ${attack_alpha} \
    --attack_steps ${attack_steps} \
    --attack_restarts ${attack_restarts} \
    --eval_mode ${eval_mode} \
    --save_npz \
    --npz_dir ${npz_dir} \
    --tpt \
    --rtpt_neighbor_k ${rtpt_neighbor_k} \
    --rtpt_tau ${rtpt_tau} \
    --dirichlet_consistency \
    --lambda_tpt ${lambda_tpt} \
    --lambda_dir ${lambda_dir} \
    --dir_temp ${dir_temp} \
    --alpha_offset ${alpha_offset}

fi

exit_code=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "Exit code: ${exit_code}"
echo "=================================================="

exit ${exit_code}