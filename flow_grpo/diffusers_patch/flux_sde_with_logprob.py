# Adapted from the SD3 SDE implementation for Flux
# Converts Flux's deterministic flow ODE into stochastic SDE for RL training

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity) using an SDE formulation
    that enables log probability computation for RL training.

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model (velocity prediction).
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        prev_sample (`torch.FloatTensor`, *optional*):
            Optionally provide the previous sample for KL computation.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        determistic (bool):
            If True, performs deterministic sampling (no noise injection).

    Returns:
        Tuple containing:
        - prev_sample: The computed previous sample
        - log_prob: Log probability of the step
        - prev_sample_mean: Mean of the step distribution
        - std_dev: Standard deviation used for the step
    """
    
    # Get timestep information
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    
    # Handle sigma values for Flux scheduler
    if hasattr(self, 'sigmas'):
        sigma = self.sigmas[step_index].view(-1, 1, 1, 1)
        sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1)
        sigma_max = self.sigmas[1].item()
    else:
        # For Flux, we need to construct sigma from timestep
        # Flux uses a different parameterization, so we adapt it
        sigma = timestep.view(-1, 1, 1, 1)
        sigma_prev = (timestep + 1.0/len(self.timesteps)).view(-1, 1, 1, 1)
        sigma_max = 1.0
    
    dt = sigma_prev - sigma
    
    # Compute noise level for SDE
    # For Flux, we adjust the noise schedule to match its flow formulation
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * 0.7
    
    # SDE mean computation adapted for Flux
    # Flux uses a different velocity parameterization, so we adapt the SDE accordingly
    prev_sample_mean = sample * (1 + std_dev_t**2 / (2*sigma) * dt) + model_output * (1 + std_dev_t**2 * (1-sigma) / (2*sigma)) * dt
    
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    # Generate stochastic component
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        # Add noise for stochastic sampling
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

    # Deterministic sampling (for evaluation)
    if determistic:
        prev_sample = sample + dt * model_output

    # Compute log probability of the step
    # This is crucial for policy gradient computation in RL
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # Mean along all but batch dimension for proper gradient computation
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)