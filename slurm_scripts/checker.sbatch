#!/bin/bash
#SBATCH --job-name=celeba_nll
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --exclude=babel-l5-24
#SBATCH --output=/home/xinyueai/Experiments/flow-maps/log/check_%A_%a.out
#SBATCH --error=/home/xinyueai/Experiments/flow-maps/log/check_%A_%a.err


# Hook conda/mamba into the shell and activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowmaps

mkdir -p /data/user_data/xinyueai/lsd_check

python /home/xinyueai/Experiments/iF2D2/py/launchers/learn.py \
    --cfg_path configs.checker --slurm_id 0 --dataset_location ""  \
    --output_folder /data/user_data/xinyueai/lsd_check \


echo "=========================================="
echo "Job completed."
echo "=========================================="
