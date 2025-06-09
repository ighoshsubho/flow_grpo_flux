import ml_collections
import imp
import os

base = imp.load_source("flux_base", os.path.join(os.path.dirname(__file__), "flux_base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.wandb_project = "flux-grpo-run"  # Set your wandb project name

    config.num_epochs = 100
    config.use_lora = True

    config.sample.train_batch_size = 4  # Smaller batch for Flux due to memory
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 2  # Smaller batch for Flux
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_flux():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # FLUX.1-dev
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 10  # Reduced steps for training
    config.sample.eval_num_steps = 28  # Full steps for evaluation
    config.sample.guidance_scale = 3.5
    config.sample.true_cfg_scale = 1.0  # No true CFG by default

    # Optimized for 2 A6000 Pro GPUs
    # Following empirical rule: train_batch_size * num_gpu / num_image_per_prompt * num_batches_per_epoch = 48
    # With group_size = 24 (num_image_per_prompt)
    config.resolution = 1024  # Flux native resolution
    config.sample.train_batch_size = 12  # 12 * 2 GPUs = 24 total batch size (divisible by 24)
    config.sample.num_image_per_prompt = 24  # group_size = 24
    config.sample.num_batches_per_epoch = 2  # (12 * 2 / 24) * 2 = 2, so group_number = 4
    config.sample.test_batch_size = 12

    # Alternative if memory is tight:
    # config.sample.train_batch_size = 6
    # config.sample.num_image_per_prompt = 12  # Reduced group size
    # config.sample.num_batches_per_epoch = 4  # (6 * 2 / 12) * 4 = 4

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2  # Following empirical rule
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.learning_rate = 5e-5  # Lower LR for Flux
    
    # KL regularization
    config.train.beta = 0.004  # Following SD3 empirical values
    # KL reward (alternative to KL loss)
    config.sample.kl_reward = 0
    
    # SFT mixing (optional)
    config.train.sft = 0.0
    config.train.sft_batch_size = 2
    
    # Stats and EMA
    config.sample.global_std = True
    config.train.ema = True
    
    # Training duration
    config.num_epochs = 100000
    config.save_freq = 60  # epochs
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/flux'
    
    # Reward configuration
    config.reward_fn = {
        "ocr": 1.0,
        # "unifiedreward": 0.7,  # Can add additional rewards
    }
    
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True
    return config

def geneval_flux():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # FLUX.1-dev
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 10  # Reduced for training
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.true_cfg_scale = 1.0

    # Optimized for 2 A6000 Pro GPUs - GenEval configuration
    # Following empirical rule: train_batch_size * num_gpu / num_image_per_prompt * num_batches_per_epoch = 48
    config.resolution = 1024
    config.sample.train_batch_size = 4  # 4 * 2 GPUs = 8 total
    config.sample.num_image_per_prompt = 24  # group_size = 24
    config.sample.num_batches_per_epoch = 6  # (4 * 2 / 24) * 6 = 2, so group_number = 12
    config.sample.test_batch_size = 6  # Adjusted for test set size

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.learning_rate = 5e-5
    config.train.beta = 0.004  # Following SD3 empirical values
    config.sample.kl_reward = 0
    config.sample.global_std = True
    config.train.ema = True
    
    config.num_epochs = 100000
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/geneval/flux'
    
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config

def pickscore_flux():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # FLUX.1-dev
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 8  # Even fewer steps for PickScore
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.true_cfg_scale = 1.0

    # Optimized for 2 A6000 Pro GPUs - PickScore configuration
    config.resolution = 1024
    config.sample.train_batch_size = 6  # 6 * 2 GPUs = 12 total
    config.sample.num_image_per_prompt = 12  # group_size = 24
    config.sample.num_batches_per_epoch = 4  # (6 * 2 / 24) * 4 = 2, so group_number = 8
    config.sample.test_batch_size = 8

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.learning_rate = 5e-5
    config.train.beta = 0.001  # Lower beta for PickScore following SD3 pattern
    config.train.sft = 0.0
    config.train.sft_batch_size = 2
    config.sample.kl_reward = 0
    config.sample.global_std = True
    config.train.ema = True
    
    config.num_epochs = 100000
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/flux'
    
    config.reward_fn = {
        "pickscore": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True
    return config

def visual_text_flux():
    """Configuration for visual text rendering task with Flux"""
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/visual_text")

    # FLUX.1-dev - good for text rendering
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 12
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    # Optimized for text rendering
    config.resolution = 1024
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 6
    config.sample.num_batches_per_epoch = 10
    config.sample.test_batch_size = 8

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.learning_rate = 3e-5  # Lower LR for text precision
    config.train.beta = 0.002  # Higher KL for text quality
    config.sample.global_std = True
    config.train.ema = True

    config.save_dir = 'logs/visual_text/flux'
    config.reward_fn = {
        "text_accuracy": 1.0,
        "ocr": 1.0,
    }
    
    config.prompt_fn = "visual_text"
    return config

def flux_schnell_ocr():
    """Configuration using FLUX.1-schnell for faster training"""
    config = general_ocr_flux()
    
    # Use FLUX.1-schnell for faster inference
    config.pretrained.model = "black-forest-labs/FLUX.1-schnell"
    config.sample.num_steps = 4  # Schnell requires fewer steps
    config.sample.eval_num_steps = 4
    config.sample.guidance_scale = 0.0  # Schnell doesn't use guidance
    
    # Can use larger batches due to faster inference
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12
    config.sample.num_batches_per_epoch = 10
    
    config.save_dir = 'logs/ocr/flux_schnell'
    return config

def flux_controlnet_composition():
    """Configuration for compositional generation with potential ControlNet support"""
    config = geneval_flux()
    
    # Enhanced composition settings
    config.sample.guidance_scale = 4.0  # Higher guidance for composition
    config.train.learning_rate = 3e-5   # Lower LR for careful composition learning
    config.train.beta = 0.002           # Higher KL to maintain quality
    
    # Composition-specific rewards
    config.reward_fn = {
        "geneval": 1.0,
        "composition_score": 0.5,  # Custom composition reward
        "spatial_accuracy": 0.3,
    }
    
    config.save_dir = 'logs/composition/flux'
    return config

def get_config(name):
    return globals()[name]()