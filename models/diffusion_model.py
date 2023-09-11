"""Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch"""
from argparse import Namespace
import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from einops import reduce, rearrange
from torch import nn, Tensor

from models.unet_model import Unet
from trainers.utils import default, get_index_from_list, normalize_to_neg_one_to_one


def linear_beta_schedule(
    timesteps: int,
    start: float = 0.0001,
    end: float = 0.02
) -> Tensor:
    """
    :param timesteps: Number of time steps

    :return schedule: betas at every timestep, (timesteps,)
    """
    scale = 1000 / timesteps
    beta_start = scale * start
    beta_end = scale * end
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ

    :param timesteps: Number of time steps
    :param s: scaling factor

    :return schedule: betas at every timestep, (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionModel(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()

        # Default parameters
        self.config = config
        dim: int = self.default('dim', 64)
        dim_mults: List[int] = self.default('dim_mults', [1, 2, 4, 8])
        channels: int = self.default('channels', 1)
        timesteps: int = self.default('timesteps', 1000)
        beta_schedule: str = self.default('beta_schedule', 'cosine')
        objective: str = self.default('objective', 'pred_noise')  # 'pred_noise' or 'pred_x_0'
        p2_loss_weight_gamma: float = self.default('p2_loss_weight_gamma', 0.)  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k: float = self.default('p2_loss_weight_k', 1.)
        dynamic_threshold_percentile: float = self.default('dynamic_threshold_percentile', 0.995)

        self.timesteps = timesteps
        self.objective = objective
        self.dynamic_threshold_percentile = dynamic_threshold_percentile
        self.model = Unet(
            dim,
            dim_mults=dim_mults,
            channels=channels
        )

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20))
        )
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        )

        # p2 reweighting
        p2_loss_weight = ((p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
                          ** (-p2_loss_weight_gamma))
        self.register_buffer('p2_loss_weight', p2_loss_weight)

    def default(self, val, d):
        return vars(self.config)[val] if val in self.config else d

    def train_step(self, x_0: Tensor, cond: Optional[Tensor] = None, t:Optional[Tensor] = None) -> Tensor:
        N, device = x_0.shape[0], x_0.device
        
        # If t is not none, use it, otherwise sample from uniform
        if t is not None:
            t = t.long().to(device)
        else:
            t = torch.randint(0, self.timesteps, (N,), device=device).long()  # (N)

        model_out, noise = self(x_0, t, cond=cond)

        if self.objective == 'pred_noise':
            target = noise  # (N, C, H, W)
        elif self.objective == 'pred_x_0':
            target = x_0  # (N, C, H, W)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.l1_loss(model_out, target, reduction='none')  # (N, C, H, W)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')  # (N, (C x H x W))

        # p2 reweighting
        loss = loss * get_index_from_list(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def val_step(self, x_0: Tensor, cond: Optional[Tensor] = None, t_steps:Optional[int] = None) -> Tensor:
        if not t_steps:
            t_steps = self.timesteps
        step_size = self.timesteps // t_steps
        N, device = x_0.shape[0], x_0.device
        losses = []
        for t in range(0, self.timesteps, step_size):  
            t = torch.ones((N,)) * t
            t = t.long().to(device)
            losses.append(self.train_step(x_0, cond, t))

        return torch.stack(losses).mean()

    def forward(self, x_0: Tensor, t: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """
        Noise x_0 for t timestep and get the model prediction.

        :param x_0: Clean image, (N, C, H, W)
        :param t: Timestep, (N,)
        :param cond: element to condition the reconstruction on - eg image when x_0 is a segmentation (N, C', H, W)

        :return pred: Model output, predicted noise or image, (N, C, H, W)
        :return noise: Added noise, (N, C, H, W)
        """
        if self.config.normalize:
            x_0 = normalize_to_neg_one_to_one(x_0)
        if cond is not None and self.config.normalize:
            cond = normalize_to_neg_one_to_one(cond)
        x_t, noise = self.forward_diffusion_model(x_0, t)
        return self.model(x_t, t, cond), noise

    def forward_diffusion_model(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Takes an image and a timestep as input and returns the noisy version
        of it.

        :param x_0: Image at timestep 0, (N, C, H, W)
        :param t: Timestep, (N)
        :param cond: element to condition the reconstruction on - eg image when x_0 is a segmentation (N, C', H, W)

        :return x_t: Noisy image at timestep t, (N, C, H, W)
        :return noise: Noise added to the image, (N, C, H, W)
        """
        noise = default(noise, lambda: torch.randn_like(x_0))

        sqrt_alphas_cumprod_t = get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # mean + variance
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    @torch.no_grad()
    def sample_timestep(self, x_t: Tensor, t: int, cond=Optional[Tensor]) -> Tensor:
        """
        Sample from the model.
        :param x_t: Image noised t times, (N, C, H, W)
        :param t: Timestep
        :return: Sampled image, (N, C, H, W)
        """
        N = x_t.shape[0]
        device = x_t.device
        batched_t = torch.full((N,), t, device=device, dtype=torch.long)  # (N)
        model_mean, model_log_variance, _ = self.p_mean_variance(x_t, batched_t, cond=cond)
        noise = torch.randn_like(x_t) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img

    def p_mean_variance(self, x_t: Tensor, t: Tensor, clip_denoised: bool = True, cond:Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        _, pred_x_0 = self.model_predictions(x_t, t, cond=cond)

        if clip_denoised:
            # pred_x_0.clamp_(-1., 1.)
            # Dynamic thrsholding
            s = torch.quantile(rearrange(pred_x_0, 'b ... -> b (...)').abs(),
                               self.dynamic_threshold_percentile,
                               dim=1)
            s = torch.max(s, torch.tensor(1.0))[:, None, None, None]
            pred_x_0 = torch.clip(pred_x_0, -s, s) / s

        (model_mean,
         posterior_log_variance) = self.q_posterior(pred_x_0, x_t, t)
        return model_mean, posterior_log_variance, pred_x_0

    def model_predictions(self, x_t: Tensor, t: Tensor, cond:Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:
        """
        Return the predicted noise and x_0 for a given x_t and t.

        :param x_t: Noised image at timestep t, (N, C, H, W)
        :param t: Timestep, (N,)
        :return pred_noise: Predicted noise, (N, C, H, W)
        :return pred_x_0: Predicted x_0, (N, C, H, W)
        """
        model_output = self.model(x_t, t, cond)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            pred_x_0 = self.predict_x_0_from_noise(x_t, t, model_output)

        elif self.objective == 'pred_x_start':
            pred_noise = self.predict_noise_from_x_0(x_t, t, model_output)
            pred_x_0 = model_output

        return pred_noise, pred_x_0

    def q_posterior(self, x_start: Tensor, x_t: Tensor, t: Tensor) \
            -> Tuple[Tensor, Tensor]:
        posterior_mean = (
            get_index_from_list(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + get_index_from_list(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = get_index_from_list(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def predict_x_0_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) \
            -> Tensor:
        """
        Get x_0 given x_t, t, and the known or predicted noise.

        :param x_t: Noised image at timestep t, (N, C, H, W)
        :param t: Timestep, (N,)
        :param noise: Noise, (N, C, H, W)
        :return: Predicted x_0, (N, C, H, W)
        """
        return (
            get_index_from_list(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_t
            - get_index_from_list(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def predict_noise_from_x_0(self, x_t: Tensor, t: Tensor, x_0: Tensor) \
            -> Tensor:
        """
        Get noise given the known or predicted x_0, x_t, and t

        :param x_t: Noised image at timestep t, (N, C, H, W)
        :param t: Timestep, (N,)
        :param noise: Noise, (N, C, H, W)
        :return: Predicted noise, (N, C, H, W)
        """
        return (
            (get_index_from_list(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0)
            / get_index_from_list(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
