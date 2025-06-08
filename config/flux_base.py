# Configuration file for Flux-GRPO training

import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    ###################
    # Experiment
    ###################
    config.seed = 42
    config.run_name = ""
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.num_checkpoint_limit = 5
    config.save_freq = 10  # Save every N epochs
    config.eval_freq = 5   # Evaluate every N epochs

    ###################
    # Logging
    ###################
    config.logdir = "./logs"
    config.save_dir = "./checkpoints"

    ###################
    # Model
    ###################
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"  # or FLUX.1-schnell
    config.use_lora = True
    config.resolution = 1024  # Flux native resolution

    ###################
    # Training
    ###################
    config.num_epochs = 100
    
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 1  # Per device training batch size
    config.train.learning_rate = 1e-4
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.use_8bit_adam = False
    config.train.max_grad_norm = 1.0
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1.0  # Fraction of timesteps to train on
    config.train.lora_path = ""  # Path to pretrained LoRA weights

    # GRPO-specific training parameters
    config.train.clip_range = 0.2
    config.train.adv_clip_max = 5.0
    config.train.beta = 0.001  # KL regularization weight
    config.train.true_cfg_scale = 1.0  # True classifier-free guidance scale

    # EMA
    config.train.ema = True

    ###################
    # Sampling
    ###################
    config.sample = ml_collections.ConfigDict()
    config.sample.train_batch_size = 4  # Per device sampling batch size
    config.sample.test_batch_size = 8   # Per device evaluation batch size
    config.sample.num_steps = 28  # Number of sampling steps (Flux default)
    config.sample.eval_num_steps = 28  # Evaluation sampling steps
    config.sample.num_batches_per_epoch = 64
    config.sample.guidance_scale = 3.5  # Flux default guidance scale
    config.sample.num_image_per_prompt = 4  # Number of images per prompt for GRPO grouping
    config.sample.kl_reward = 0.1  # Weight for KL divergence in reward
    config.sample.global_std = False  # Use global std for advantage normalization

    ###################
    # Prompt and Reward
    ###################
    config.prompt_fn = "general_ocr"  # Type of prompts to use
    config.dataset = "./datasets/ocr_prompts"  # Path to prompt dataset
    
    # Reward function configuration
    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.ocr_accuracy = ml_collections.ConfigDict()
    config.reward_fn.ocr_accuracy.weight = 1.0
    
    # Per-prompt statistics tracking
    config.per_prompt_stat_tracking = True

    return config