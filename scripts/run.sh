#!/bin/bash
#SBATCH --job-name=adv_otpt_exp
#SBATCH -p gpu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/hpc/07/zhang303/O-TPT-main/%x-%j.out
#SBATCH -e /scratch/hpc/07/zhang303/O-TPT-main/%x-%j.err

module add anaconda3/2022.05
source activate ctpt

# =========================
# Basic paths
# =========================
data_root='/scratch/hpc/07/zhang303/O-TPT-main/data'
testsets=$1
csv_log='/scratch/hpc/07/zhang303/O-TPT-main/log/adv_otpt_results.csv'

# =========================
# Experiment switches
# =========================
# method: baseline / tpt / otpt / ts
method='baseline'

# backbone: openai / fare2 / openclip
backbone='fare2'

# attack: true / false
use_attack='true'

# optional checkpoint init: true / false
use_load_ckpt='false'
load_ckpt=''

# =========================
# Model settings
# =========================
arch='ViT-B/32'
bs=64
gpu_id=0
download_root='~/.cache/clip'

# prompt/tuning params
ctx_init='a_photo_of_a'
tta_steps=1
selection_p=0.1
n_ctx=4
lambda_term=18

# attack params
pgd_eps=0.01568627
pgd_alpha=0.00392157
pgd_steps=5

# openclip/fare2 params
# for backbone='fare2', this will be used automatically
# for backbone='openclip', replace with another pretrained id if needed
openclip_pretrained='hf-hub:chs20/fare2-clip'
# hf-hub:chs20/tecoa4-clip
# hf-hub:chs20/fare4-clip
# hf-hub:chs20/fare2-clip
# hf-hub:chs20/tecoa4-clip

# optional extras
seed=0
dataset_mode='test'
use_i_augmix='false'
use_two_step='false'

# =========================
# Resolve method -> run_type
# =========================
if [ "${method}" = "baseline" ]; then
  run_type='baseline'
elif [ "${method}" = "tpt" ]; then
  run_type='tpt'
elif [ "${method}" = "otpt" ]; then
  run_type='tpt_otpt'
elif [ "${method}" = "ts" ]; then
  run_type='tpt_ts'
else
  echo "Unknown method: ${method}"
  exit 1
fi

# =========================
# Resolve backbone
# =========================
if [ "${backbone}" = "openai" ]; then
  clip_impl='openai'
elif [ "${backbone}" = "fare2" ]; then
  clip_impl='open_clip'
  openclip_pretrained='hf-hub:chs20/fare2-clip'
elif [ "${backbone}" = "openclip" ]; then
  clip_impl='open_clip'
else
  echo "Unknown backbone: ${backbone}"
  exit 1
fi

# =========================
# Build base command
# =========================
CMD="python ./adv_otpt_classification.py ${data_root} \
  --test_sets ${testsets} \
  --csv_log ${csv_log} \
  -a ${arch} \
  -b ${bs} \
  --gpu ${gpu_id} \
  --run_type ${run_type} \
  --clip_impl ${clip_impl} \
  --download_root ${download_root} \
  --seed ${seed} \
  --dataset_mode ${dataset_mode}"

# =========================
# Backbone-specific args
# =========================
if [ "${clip_impl}" = "open_clip" ]; then
  CMD="${CMD} --openclip_pretrained ${openclip_pretrained}"
fi

# =========================
# Method-specific args
# =========================
if [ "${method}" = "tpt" ] || [ "${method}" = "otpt" ] || [ "${method}" = "ts" ]; then
  CMD="${CMD} --tpt"
  CMD="${CMD} --ctx_init ${ctx_init}"
  CMD="${CMD} --tta_steps ${tta_steps}"
  CMD="${CMD} --selection_p ${selection_p}"
  CMD="${CMD} --n_ctx ${n_ctx}"
fi

if [ "${method}" = "otpt" ]; then
  CMD="${CMD} --lambda_term ${lambda_term}"
fi

# =========================
# Attack-specific args
# =========================
if [ "${use_attack}" = "true" ]; then
  CMD="${CMD} --eval_attack"
  CMD="${CMD} --pgd_eps ${pgd_eps}"
  CMD="${CMD} --pgd_alpha ${pgd_alpha}"
  CMD="${CMD} --pgd_steps ${pgd_steps}"
  CMD="${CMD} --pgd_random_start"
fi

# =========================
# Optional extras
# =========================
if [ "${use_i_augmix}" = "true" ]; then
  CMD="${CMD} --I_augmix"
fi

if [ "${use_two_step}" = "true" ]; then
  CMD="${CMD} --two_step"
fi

if [ "${use_load_ckpt}" = "true" ]; then
  CMD="${CMD} --load ${load_ckpt}"
fi

echo "${CMD}"
srun ${CMD}