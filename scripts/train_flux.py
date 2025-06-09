# Example training script for Flux with Flow-GRPO
# Adapted from the SD3 Flow-GRPO training script

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.flux_sde_with_logprob import sde_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
import itertools
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from peft.utils import get_peft_model_state_dict
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/flux_base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每卡的batch大小
        self.k = k                    # 每个样本重复的次数
        self.num_replicas = num_replicas  # 总卡数
        self.rank = rank              # 当前卡编号
        self.seed = seed              # 随机种子，用于同步
        
        # 计算每个迭代需要的不同样本数
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # 不同样本数
        self.epoch=0

    def __iter__(self):
        while True:
            # 生成确定性的随机序列，确保所有卡同步
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # print('epoch', self.epoch)
            # 随机选择m个不同的样本
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            # print(self.rank, 'indices', indices)
            # 每个样本重复k次，生成总样本数n*b
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # 打乱顺序确保均匀分配
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            # print(self.rank, 'shuffled_samples', shuffled_samples)
            # 将样本分割到各个卡
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            # print(self.rank, 'per_card_samples', per_card_samples[self.rank])
            # 返回当前卡的样本索引
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于同步不同 epoch 的随机状态

def compute_text_embeddings(prompt, pipeline, max_sequence_length, device):
    """Compute text embeddings for Flux model"""
    with torch.no_grad():
        # For Flux, we need both CLIP and T5 embeddings
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,  # Use same prompt for both encoders
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids

def compute_log_prob(transformer, pipeline, sample, j, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, config):
    """Compute log probability for a single denoising step"""
    
    # Get guidance if model supports it
    guidance = None
    if transformer.config.guidance_embeds:
        guidance = torch.full([1], config.sample.guidance_scale, 
                            device=prompt_embeds.device, dtype=torch.float32)
        guidance = guidance.expand(sample["latents"][:, j].shape[0])
    
    # Handle true CFG if needed
    if config.train.true_cfg_scale > 1.0:
        # Get negative prompt embeddings (should be precomputed)
        neg_prompt_embeds = sample.get("neg_prompt_embeds")
        neg_pooled_prompt_embeds = sample.get("neg_pooled_prompt_embeds") 
        neg_text_ids = sample.get("neg_text_ids")
        
        # Positive prediction
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j] / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        
        # Negative prediction
        neg_noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j] / 1000,
            guidance=guidance,
            pooled_projections=neg_pooled_prompt_embeds,
            encoder_hidden_states=neg_prompt_embeds,
            txt_ids=neg_text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        
        # Apply CFG
        noise_pred = neg_noise_pred + config.train.true_cfg_scale * (noise_pred - neg_noise_pred)
    else:
        # Standard prediction
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j] / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
    
    # Compute log probability using SDE step
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def main(_):
    # Basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # Number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    
    if accelerator.is_main_process:
        # Initialize wandb with project name from config
        wandb_project = getattr(config, 'wandb_project', 'flux-grpo-run')
        accelerator.init_trackers(
            project_name=wandb_project,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # Set seed
    set_seed(config.seed, device_specific=True)

    # Load Flux pipeline
    pipeline = FluxPipeline.from_pretrained(config.pretrained.model)
    
    # Freeze non-trainable components
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    # Disable safety checker
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Handle mixed precision
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move models to device and cast to inference dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.transformer.to(accelerator.device)

    # Setup LoRA if enabled
    if config.use_lora:
        # Flux transformer LoRA target modules
        target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_mlp", "proj_out",
            "norm.linear", "norm_out.linear",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.train.lora_path
            )
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(
                pipeline.transformer, transformer_lora_config
            )
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    
    # EMA setup
    ema = EMAModuleWrapper(
        transformer_trainable_parameters, 
        decay=0.9, 
        update_step_interval=8, 
        device=accelerator.device
    )
    
    # Enable TF32 for faster training
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. "
                "You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Setup reward function
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(
        accelerator.device, config.reward_fn
    )
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(
        accelerator.device, config.reward_fn
    )

    # Setup datasets and dataloaders
    if config.prompt_fn == "general_ocr":
        
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError(f"Dataset {config.prompt_fn} not implemented for Flux")

    # Prepare negative prompt embeddings
    neg_prompt_embed, neg_pooled_prompt_embed, neg_text_ids = compute_text_embeddings(
        [""], pipeline, max_sequence_length=512, device=accelerator.device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)
    sample_neg_text_ids = neg_text_ids.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_text_ids = neg_text_ids.repeat(config.train.batch_size, 1, 1)

    # Initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # Setup autocast
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, test_dataloader
    )

    # Executor for async reward computation
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Training info
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running Flux-GRPO training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    global_step = 0
    train_iter = iter(train_dataloader)

    for epoch in range(config.num_epochs):
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            # Compute text embeddings for Flux
            prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                prompts, pipeline, max_sequence_length=512, device=accelerator.device
            )
            
            # For tokenizer compatibility
            prompt_ids = pipeline.tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # Skip first epochs for stability
            if epoch < 2:
                continue
            
            # Sample images with log probabilities
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs, kls = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        true_cfg_scale=config.sample.get('true_cfg_scale', 1.0),
                        output_type="pt",
                        return_dict=False,
                        height=config.resolution,
                        width=config.resolution, 
                        kl_reward=config.sample.kl_reward,
                        determistic=False,
                    )

            # Prepare trajectory data
            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
            kls = torch.stack(kls, dim=1) 
            kl = kls.detach()

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )

            # Compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)  # Yield to start reward computation

            samples.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "text_ids": text_ids,
                "timesteps": timesteps,
                "latents": latents[:, :-1],  # States before timestep t
                "next_latents": latents[:, 1:],  # States after timestep t
                "log_probs": log_probs,
                "kl": kl,
                "rewards": rewards,
            })

        if epoch < 2:
            continue

        # Wait for rewards and process samples
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # Collate samples
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # Log sample images periodically
        if epoch % 10 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                accelerator.log({
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                        )
                        for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                }, step=global_step)

        # Apply KL penalty to rewards
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = (
            samples["rewards"]["avg"].unsqueeze(-1) - 
            config.sample.kl_reward * samples["kl"]
        )

        # Gather rewards across processes
        gathered_rewards = {
            key: accelerator.gather(value) for key, value in samples["rewards"].items()
        }
        gathered_rewards = {
            key: value.cpu().numpy() for key, value in gathered_rewards.items()
        }

        # Log rewards
        accelerator.log({
            "epoch": epoch,
            **{
                f"reward_{key}": value.mean() 
                for key, value in gathered_rewards.items() 
                if '_strict_accuracy' not in key and '_accuracy' not in key
            },
            "kl": samples["kl"].mean().cpu().numpy(),
            "kl_abs": samples["kl"].abs().mean().cpu().numpy()
        }, step=global_step)

        # Compute advantages using GRPO
        if config.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            
            group_size, trained_prompt_num = stat_tracker.get_stats()
            accelerator.log({
                "group_size": group_size,
                "trained_prompt_num": trained_prompt_num,
            }, step=global_step)
            stat_tracker.clear()
        else:
            advantages = (
                (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / 
                (gathered_rewards['avg'].std() + 1e-4)
            )

        # Distribute advantages back to processes
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )

        # Filter out zero advantage samples
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True

        accelerator.log({
            "actual_batch_size": mask.sum().item() // config.sample.num_batches_per_epoch,
        }, step=global_step)

        # Apply mask
        samples = {k: v[mask] for k, v in samples.items()}
        
        total_batch_size, num_timesteps = samples["timesteps"].shape

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # Shuffle samples
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # Shuffle timesteps independently
            perms = torch.stack([
                torch.arange(num_timesteps, device=accelerator.device)
                for _ in range(total_batch_size)
            ])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # Rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # Convert to list of dicts
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # Train
            pipeline.transformer.train()
            info = defaultdict(list)
            
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # Prepare embeddings for CFG
                if config.train.true_cfg_scale > 1.0:
                    embeds = torch.cat([
                        train_neg_prompt_embeds[:len(sample["prompt_embeds"])], 
                        sample["prompt_embeds"]
                    ])
                    pooled_embeds = torch.cat([
                        train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], 
                        sample["pooled_prompt_embeds"]
                    ])
                    text_ids = torch.cat([
                        train_neg_text_ids[:len(sample["text_ids"])], 
                        sample["text_ids"]
                    ])
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]
                    text_ids = sample["text_ids"]

                # Prepare latent image IDs for Flux
                batch_size = sample["latents"].shape[0]
                height = width = int(np.sqrt(sample["latents"].shape[2]))  # Assuming square latents
                latent_image_ids = pipeline._prepare_latent_image_ids(
                    batch_size, height, width, accelerator.device, sample["latents"].dtype
                )

                train_timesteps = list(range(num_train_timesteps))
                
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            # Compute current policy log probability
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                transformer, pipeline, sample, j, embeds, pooled_embeds, 
                                text_ids, latent_image_ids, config
                            )
                            
                            # Compute reference policy log probability (for KL regularization)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref = compute_log_prob(
                                            transformer, pipeline, sample, j, embeds, pooled_embeds,
                                            text_ids, latent_image_ids, config
                                        )

                        # GRPO loss computation
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        # KL regularization
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        # Track metrics
                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float())
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                        # Backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Update global step and log
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                
                # Update EMA
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

if __name__ == "__main__":
    app.run(main)