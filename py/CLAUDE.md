# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **public release** research codebase for **Lagrangian Map Matching** and **self-distillation** for generative models, implementing stochastic interpolants and flow-based generative models using JAX/Flax with EDM2-style architectures.

**This is a cleaned, production-ready version** containing only the code necessary to reproduce the paper's final experiments. All experimental features and unused loss variants have been removed.

## Key Commands

### Training Models
```bash
# Basic training with config module
python py/launchers/learn.py --cfg_path configs.default_cifar10 --slurm_id 0

# Training with external paths (HPC environments)
python py/launchers/learn.py \
    --cfg_path configs.cifar10 \
    --slurm_id 0 \
    --dataset_location /path/to/datasets \
    --output_folder /path/to/outputs
```

### Sampling and Evaluation
```bash
# Generate samples from checkpoint
python py/launchers/sample_model.py \
    --config_path /path/to/config.py \
    --checkpoint_path /path/to/checkpoint \
    --n_samples 10000 \
    --output_path samples.npy

# Calculate FID from samples
python py/launchers/calc_fid.py \
    --image_file samples.npy \
    --cifar_stats /path/to/cifar_stats.npz \
    --batch_size 100

# Combined sampling and FID (uses WandB runs)
python py/launchers/sample_and_calc_fid.py \
    --config_name config_name \
    --slurm_id 0
```

### Dataset Preprocessing
```bash
# Generate FID reference statistics
python py/launchers/calc_dataset_fid_stats.py \
    --dataset cifar10 \
    --dataset_location /path/to/datasets \
    --out cifar10_fid_stats.npz
```

### Code Formatting
```bash
# Format Python code with Black
black py/ --line-length 100
```

### Code Search and Refactoring
**Important**: When searching for code patterns or performing refactoring tasks, prefer syntax-aware tools over text-based search:

```bash
# Use ast-grep for syntax-aware Python searches
ast-grep --lang python -p 'def $FUNC($$$): $$$'

# ast-grep is available and should be the default for:
# - Finding function definitions and call sites
# - Structural pattern matching
# - Syntax-aware refactoring
# - Any search requiring semantic understanding

# Only fall back to ripgrep/grep for:
# - Plain text search (comments, strings, documentation)
# - Multi-language searches
# - Simple string matching
```

**Rationale**: Syntax-aware search prevents false positives from comments, strings, and similar-looking code, making refactoring safer and more reliable.

## High-Level Architecture

### Core Training Loop (`launchers/learn.py`)
The main training script orchestrates:
1. **Single-node multi-GPU training** via JAX `pmap` for data parallelism
2. **State management** with EMA (Exponential Moving Average) for stable generation
3. **Loss computation** selecting between diagonal/interpolant terms, PSD, LSD, or ESD losses
4. **Dynamic loss switching** based on training stage (velocity vs full loss)
5. **Online FID computation** during training for monitoring quality

### Loss Functions Architecture (`common/losses.py`)
Multiple loss types with configurable gradient stopping strategies:
- **Diagonal/Interpolant**: Basic velocity matching loss
- **PSD** (Progressive Self-Distillation): Two-step distillation with stopgrad options:
  - `uniform`: Uniform weighting of intermediate steps
  - `midpoint`: Midpoint weighting variant
- **LSD** (Lagrangian Self-Distillation): Time derivative matching with stopgrad options:
  - `convex`: Stops gradients on teacher evaluations
  - `none`: Full gradient flow
- **ESD** (Eulerian Self-Distillation): Spatial derivative matching with stopgrad options:
  - `full`: Stop all gradients through spatial Jacobian
  - `convex`: Stop gradients on teacher evaluations only
  - `none`: Full gradient flow

### Configuration System (`configs/`)
Configurations use `ml_collections.ConfigDict` with key sections:
- `config.training`: Loss types, stopgrad strategies, EMA factors, training modes
- `config.problem`: Dataset, interpolant type, dimensions
- `config.network`: EDM2 architecture parameters
- `config.optimization`: Learning rates, schedules, batch sizes
- `config.logging`: WandB settings, FID computation, visualization

The `slurm_id` parameter in configs enables sweep experiments by indexing into parameter lists.

### Network Architecture (`common/edm2_net.py`, `common/flow_map.py`)
- **EDM2 UNet**: Enhanced architecture with:
  - MPConv layers using sphere projections for weight normalization
  - Positional embeddings for time conditioning
  - Self-attention at multiple resolutions
  - Dropout and label conditioning
- **Flow Map**: Wraps the UNet to compute transformations X(s,t,x) and potentials φ(s,t,x)
- **Velocity Field**: Alternative parameterization for integration-based sampling

### State Management (`common/state_utils.py`)
- **EMATrainState**: Extends Flax's TrainState with multiple EMA parameter copies
- **StaticArgs**: Immutable configuration and function references
- **Dynamic switching**: Between velocity and full loss training based on step count

### Single-Node Multi-GPU Training (`common/dist_utils.py`)
- Automatic multi-GPU data parallelism using JAX's `pmap`
- Efficient batch sharding across local devices
- Parameter replication with automatic synchronization across local GPUs
- Online statistics computation using Welford's algorithm
- Note: This codebase supports single-node only (no multi-node/distributed training)

### FID Computation (`FID_INTEGRATION_README.md`)
- **On-the-fly evaluation** during training with configurable frequency
- **Distributed sampling** across all GPUs for efficiency
- **Inception network** initialized once and reused
- **Memory-efficient** online statistics computation

## Key Implementation Details

### Training Dynamics
- **Two-phase training**: Initial velocity matching followed by full loss optimization
- **Teacher-student setup**: EMA parameters serve as teacher for self-distillation
- **Constant triangle sampling**: Full triangle sampling with configurable time ranges [tmin, tmax]
- **Gradient clipping**: Prevents training instabilities

### Memory Management
- Batch sizes automatically distributed across available GPUs
- Compilation forced upfront to detect memory issues early
- Online statistics avoid storing all samples for FID
- Configurable gradient accumulation for large models

### GPU Requirements
- Designed for single-node multi-GPU training
- Automatically detects and uses all available GPUs on the node via `jax.device_count()`
- Batch sizes are automatically sharded across available devices
- Works on single GPU or multi-GPU workstations/nodes

## Algorithm Concepts

### Stochastic Interpolants
Define smooth paths between noise x₀ and data x₁ distributions via:
- α(t), β(t): Time-dependent interpolation functions
- I_t = α(t)x₀ + β(t)x₁: Interpolated state
- İ_t: Time derivative (velocity) of interpolant

### Self-Distillation Loss Types
- **Progressive Self-Distillation (PSD)**: Two-step teacher distillation with composition
- **Lagrangian Self-Distillation (LSD)**: Matches time derivatives ∂_t X(s,t,x)
- **Eulerian Self-Distillation (ESD)**: Matches spatial derivatives ∂_s X(s,t,x)

### Gradient Stopping Strategies
Control information flow in teacher-student setup:
- `none`: Full gradients through all computations
- `convex`: Stop gradients on teacher network evaluations (most common)
- `full`: Stop all gradients including spatial Jacobian (ESD only)

## Dependencies

Core stack (managed externally via conda/pip):
- JAX/JAXlib 0.4.26+
- Flax 0.8.2+
- Optax 0.2.2+
- ml_collections 0.1.1+
- TensorFlow 2.15+ (data loading only)
- WandB 0.16.5+
- Black 24.3.0 (formatting)

## Public Release Simplifications

This codebase has been cleaned for public release. The following experimental features have been removed:

### Removed Features
- **Dual weight functions**: Single unified weight function `calc_weight(s, t)` for both diagonal and off-diagonal terms
- **Complex annealing schedules**: Constant full triangle sampling throughout training
- **Unused loss variants**: PFMM and mean_flow losses (keeping only LSD, PSD, ESD)
- **Velocity field training**: Removed velocity-specific training modes

### Final Experiment Structure
Each dataset has **4 experiments** (indexed by `slurm_id`):
1. **LSD** with `convex` stopgrad
2. **PSD uniform** with `convex` stopgrad
3. **PSD midpoint** with `convex` stopgrad
4. **ESD full** with `full` stopgrad (for ESD) or `convex` (for PSD/LSD)

### Key Simplifications
- **Weight functions**: All code uses single `calc_weight(s, t)` method
- **Time sampling**: Constant `delta = tmax - tmin` throughout training (no annealing)
- **Configs**: Cleaned to remove unused parameters (`use_dual_weight_functions`, annealing parameters)
- **Code formatting**: Applied Black with 100 character line length throughout