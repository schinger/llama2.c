from dataclasses import dataclass, field
from typing import Literal, Optional, List
import tyro
import math
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class PPOConfig:
    """
    Configuration class for PPO
    """
    seed: int = 0
    """Seed value for random generations"""
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    """Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"""

    # hyperparameters
    steps: int = 20000
    """Number of training steps"""
    learning_rate: float = 1e-5
    """Adam learning rate"""
    adap_kl_ctrl: bool = True
    """Use adaptive KL control, otherwise linear"""
    init_kl_coef: Optional[float] = 0.2
    """Initial KL penalty coefficient (used for adaptive and linear control)"""
    kl_penalty: Literal["kl", "abs", "mse", "full"] = "kl"
    """kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"""
    target: Optional[float] = 6
    """Target KL value for adaptive KL control"""
    horizon: Optional[float] = 10000
    """Horizon for adaptive KL control"""
    gamma: float = 1
    """Gamma parameter for advantage calculation"""
    lam: float = 0.95
    """Lambda parameter for advantage calculation"""
    cliprange: float = 0.2
    """Range for clipping in PPO policy gradient loss"""
    cliprange_value: float = 0.2
    """Range for clipping values in loss calculation"""
    vf_coef: float = 0.1
    """Scaling factor for value loss"""
    batch_size: int = 256
    """Number of samples per optimisation step"""
    mini_batch_size: int = 1
    """Number of samples optimized in each mini batch"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    world_size: int = None
    """The world size for distributed training"""
    ppo_epochs: int = 4
    """Number of optimisation epochs per batch of samples"""
    max_grad_norm: Optional[float] = None
    """Maximum gradient norm for gradient clipping"""
    optimize_device_cache: Optional[bool] = False
    """Optimize device cache for slightly more memory-efficient training"""
    early_stopping: bool = False
    """Whether to stop the PPO optimization loop early is the KL too high"""
    target_kl: float = 1
    """Stop early if we exceed this value by over 50%"""
    compare_steps: int = 1
    """Number of steps between comparison of the current reward with the best seen so far"""
    ratio_threshold: float = 10.0
    """Skip mini-batches with high PPO ratios that can cause loss spikes"""
    whiten_rewards: bool = False
    """Whiten the rewards before compute advantages"""

    # computed hyperparameters at runtime; we use `tyro.conf.Suppress` to hide them from the help text
    backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: Number of samples optimized in an `optimizer.step()` call"""
    global_backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `backward_batch_size` across all processes"""
    global_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `batch_size` across all processes"""


class PPOTrainer():
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback
    and the HuggingFace implementation trl here:
    """
    def __init__(
        self,
        config: PPOConfig = None,
        model: nn.Module = None,
        ref_model: Optional[nn.Module] = None,
        tokenizer = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        pass
    # @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        pass


class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value=0, max_value=10):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)

class LengthReward:
    """
    Calculates the reward based on the length of the response
    """

    def __init__(self, target_length=50):
        self.target_length = target_length
    
    def __call__(self, response_length):
        return -abs(self.target_length - response_length)