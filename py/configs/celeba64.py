"""
Nicholas M. Boffi
10/5/25

Algorithmic comparison on the CelebA-64 dataset.
"""

import os
import ml_collections

experiments = [
    ("lsd", None, "convex"),
]


def get_config(
    slurm_id: int, dataset_location: str, output_folder: str
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (weird issue with importlib)
    import jax

    # setup overall config
    loss_type, psd_type, stopgrad_type = experiments[slurm_id % len(experiments)]
    config = ml_collections.ConfigDict()

    # training config
    config.training = ml_collections.ConfigDict()
    config.training.shuffle = True
    config.training.conditional = False
    config.training.class_dropout = 0.0
    config.training.stopgrad_type = stopgrad_type
    config.training.psd_type = psd_type
    config.training.loss_type = loss_type
    config.training.tmin = 0.0
    config.training.tmax = 1.0
    config.training.seed = 42
    config.training.ema_facs = [0.9999]
    config.training.ndevices = jax.device_count()
    config.training.diag_teacher_source = "external"
    config.training.offdiag_teacher_source = "self"

    # problem config
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 202_599  # CelebA dataset size
    config.problem.image_dims = (3, 64, 64)
    config.problem.d = 12288  # 3 * 64 * 64
    config.problem.num_classes = 0  # No classes for CelebA
    config.problem.target = "celeb_a"
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.gaussian_scale = "adaptive"

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 256
    config.optimization.diag_fraction = 0.5
    config.optimization.learning_rate = 1e-4  # Initial learning rate
    config.optimization.clip = 1.0
    config.optimization.total_samples = 204_800_000
    config.optimization.total_steps = int(
        config.optimization.total_samples // config.optimization.bs
    )
    config.optimization.decay_steps = 35000
    config.optimization.schedule_type = "sqrt"  # Square root schedule

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 25
    
    # visualization frequency
    config.logging.visual_freq = 1000
    
    # save frequency
    config.logging.save_freq = 5000  # Save every 5k steps
    config.logging.wandb_project = "self-distill-flow-maps"
    
    # BPD computation settings
    # Compute BPD for the first batch to monitor
    # BPD computation frequency
    config.logging.bpd_freq = 1000
    config.logging.bpd_n_steps = [1, 2, 4, 8]
    config.logging.bpd_batch_size = 128
    config.logging.bpd_ema_factor = 0.9999

    # Create systematic name for the experiment
    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type

    config.logging.wandb_name = f"celeba_paper_{method_str}"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    # FID computation settings
    # FID computation frequency
    config.logging.fid_freq = 1000  # Compute FID every 10k steps
    config.logging.fid_stats_path = f"{dataset_location}/celeb_a/celeba_stats.npz"
    
    # Compute 10k-FID to monitor
    config.logging.fid_n_samples = 10000
    config.logging.fid_batch_size = 256
    config.logging.fid_n_steps_flow = [1, 2, 4, 8, 16]
    config.logging.fid_ema_factor = 0.9999
    config.logging.visual_ema_factor = 0.9999

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "edm2"
    
    # download pretrained model from huggingface and change it to local path
    config.network.load_path = "/data/user_data/xinyueai/flow-maps/celeba-lsd/celeba_paper_lsd_64.pkl"
    config.network.img_resolution = config.problem.image_dims[1]
    config.network.img_channels = config.problem.image_dims[0]
    config.network.input_dims = config.problem.image_dims
    config.network.label_dim = 0  # No class conditioning for CelebA
    config.network.use_cfg = False
    config.network.reset_optimizer = True
    config.network.logvar_channels = 128
    config.network.use_bfloat16 = True
    config.network.use_weight = True
    config.network.rescale = 0.5
    config.network.divergence_rescale = 10000.0
    config.network.init_from_ema_factor = 0.9999

    # CelebA-specific UNet architecture
    config.network.unet_kwargs = {
        "model_channels": 128,
        "channel_mult": [1, 2, 3, 4],
        "num_blocks": 3,
        "attn_resolutions": [16, 8],
        "predict_divergence": True,
        "block_kwargs": {
            "dropout": 0.0,  # No dropout for CelebA
        },
    }

    # teacher checkpoint for calculating teacher velocity and divergence
    config.teacher = ml_collections.ConfigDict()
    # use the same pretrained model for teacher as initialization
    config.teacher.load_path = "/data/user_data/xinyueai/flow-maps/celeba-lsd/celeba_paper_lsd_64.pkl"
    config.teacher.ema_fac = 0.9999

    return config