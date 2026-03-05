# F2D2-Jax

JAX implementation of **F2D2:Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models** 
https://arxiv.org/abs/2512.02636


Modified from **[Official repository](https://github.com/nmboffi/flow-maps) for "How to build a consistency model: Learning flow maps via self-distillation" (NeurIPS 2025).** https://arxiv.org/abs/2505.18825 by Nicholas M. Boffi (CMU), Michael Albergo (Harvard), and Eric Vanden-Eijnden (Courant Institute of Mathematical Sciences, Capital Fund Management)

We implement **F2D2** with **Lagrangian Self-Distillation (LSD)** on CelebA-64 and checkerboard.

The Pytorch version can be found [here](https://github.com/Keely-Ai/F2D2).

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ or 12.0+

### Setup

**1. Clone and create environment**
```bash
git clone https://github.com/nmboffi/flow-maps.git
cd flow-maps
conda create -n flowmaps python=3.9
conda activate flowmaps
```

**2. Install JAX** for your CUDA version:
```bash
# CUDA 12.x
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11.8
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CPU only
pip install --upgrade jax
```

**3. Install dependencies**
```bash
pip install \
    flax==0.8.2 \
    optax==0.2.2 \
    ml_collections==0.1.1 \
    tensorflow==2.15.0 \
    tensorflow-datasets==4.9.4 \
    wandb==0.16.5 \
    matplotlib==3.7.0 \
    seaborn==0.12.2 \
    scipy==1.10.1 \
    click==8.1.7 \
    requests==2.31.0 \
    tqdm==4.65.0
```

**4. Verify**
```bash
python -c "import jax; print(f'JAX {jax.__version__} | Devices: {jax.devices()}')"
```


## Quick start

### Training

- Download the pretrained vanilla LSD model on CelebA-64 from huggingface: https://huggingface.co/Keely-Ai/lsd/tree/main
- The pretrained vanilla LSD model on checkerboard is at iF2D2/checkerboard/checker_paper_lsd.pkl

```bash
# CelebA-64
python py/launchers/learn.py \
    --cfg_path configs.celeba64 \
    --slurm_id 0 \
    --dataset_location /path/to/datasets \
    --output_folder /path/to/outputs

# Checkerboard
python py/launchers/learn.py \
    --cfg_path configs.checker \
    --slurm_id 0 \
    --dataset_location "" \
    --output_folder /path/to/outputs
```

**Important**: You will need to modify for F2D2 training :
- Visualization, save, BPD calculation, FID calculation frequency
- Modify `config.network.load_path` to the model path you previously downloaded from huggingface or from this repo
- Modify `config.teacher.load_path` to the same as `config.network.load_path` (use the same pretrained model for teacher to calculate teacher velocity and divergence)
- **We only calculate BPD on a single batch on-the-fly.**

### Evaluation

```bash
# Compute FID
python py/launchers/calc_dataset_fid_stats.py --dataset celeb_a --out celeba_stats.npz
python py/launchers/sample_and_calc_fid.py \
    --cfg_path configs.celeba64 \
    --checkpoint checkpoints/model.pkl \
    --stats celeba_stats.npz \
    --n_steps 1
```

```bash
# Compute BPD
python py/launchers/calc_celeba_nll.py \
    --data_dir /path/to/data_dir \
    --checkpoint /path/to/F2D2_trained_ckpt \
    --batch_size {batch_size} \
    --n_steps {number_of_steps_for_BPD} \
```

## Datasets

- **CelebA-64**: Auto-downloaded via TensorFlow Datasets; pre-processed via cropping in included code.
- **Checkerboard**


## Multi-GPU training
This codebase is written for single-node, multi-GPU training. JAX automatically uses all visible GPUs:

```bash
# Use all GPUs
python py/launchers/learn.py --cfg_path configs.celeba64 --slurm_id 0

# Restrict to specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python py/launchers/learn.py --cfg_path configs.celeba64 --slurm_id 0
```


## SLURM cluster deployment

For large-scale experiments, SLURM batch scripts are provided in `slurm_scripts/`:

```bash
sbatch slurm_scripts/celeba.sbatch

# FID computation for trained models
sbatch slurm_scripts/celeba_fid.sbatch
```

**Important**: These scripts are configured for our specific cluster. You will need to modify:
- Account/partition names (`#SBATCH --account`, `#SBATCH --partition`)
- Module loading commands (`module load`)
- Conda environment paths and activation
- Dataset and output directory paths
- Time limits and memory requirements based on your hardware


## Weights & Biases logging

This codebase uses [Weights & Biases](https://wandb.ai) for experiment tracking and visualization.

### Setup

1. **Create a WandB account** at [wandb.ai](https://wandb.ai)

2. **Login** on your machine:
```bash
wandb login
```

3. **Configure your entity**: Set an environment variable with your WandB username:
```bash
export WANDB_ENTITY="your-username"
```

### Disabling WandB

To train without WandB logging:
```bash
export WANDB_MODE=offline
python py/launchers/learn.py --cfg_path configs.celeba64 --slurm_id 0
```

Or disable completely:
```bash
export WANDB_DISABLED=true
python py/launchers/learn.py --cfg_path configs.celeba64 --slurm_id 0
```

### Logging structure

- **Project**: Experiments log to the project specified in config (default: `self-distill-flow-maps`)
- **Run names**: Automatically generated from dataset, loss type, and hyperparameters
- **Metrics logged**:
  - Training losses
  - FID scores at multiple sampling steps (1, 2, 4, 8, 16)
  - BPD score on one batch at multiple sampling steps (1, 2, 4, 8)
  - Learning rate, gradient norms
  - Sample visualizations



## License

This code is distributed under the MIT License.
