import os
import torch
from torch import nn, Tensor
from typing import Dict, Tuple, Optional
from argparse import Namespace
from einops import repeat
from einops.layers.torch import Rearrange
from functools import partial
from models.diffusion_model import DiffusionModel
from trainers.utils import compare_configs



# Hooks code inspired by https://www.lyndonduong.com/saving-activations/
# Accessed on 13Feb23
def save_activations(
        activations: Dict,
        name: str,
        module: nn.Module,
        inp: Tuple,
        out: torch.Tensor
        ) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    #activations[name].append(out.detach().cpu())
    activations[name] = out.detach().cpu()


class DatasetDM(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        # Load the model
        if not os.path.isfile(args.saved_diffusion_model):
            self.diffusion_model = DiffusionModel(args)
            if args.verbose:
                print(f'No model found at {args.saved_diffusion_model}. Please load model!')
        else:
            checkpoint = torch.load(args.saved_diffusion_model, map_location=torch.device(args.device))
            old_config = checkpoint['config']
            compare_configs(old_config, args)
            self.diffusion_model = DiffusionModel(old_config)
            self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.diffusion_model.eval()

        # storage for saved activations
        self._features = {}

        # Note that this only works for the model in model.py
        for i, (block1, block2, attn, upsample) in enumerate(self.diffusion_model.model.ups):
            attn.register_forward_hook(
                partial(save_activations, self._features, i)
            )

        self.steps = args.t_steps_to_save

        self.classifier = nn.Sequential(
            nn.Conv2d(960 * len(self.steps), 128, 1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 1))
        

    @torch.no_grad()
    def extract_features(self, x_0: Tensor, noise: Optional[Tensor] = None) -> Dict[int, Tensor]:
        if noise is not None:
            assert(x_0.shape == noise.shape)
        activations=[]
        for t_step in self.steps:
            # Add t_steps of noise to x_0 - forward process
            t_step = torch.Tensor([t_step]).long().to(x_0.device)
            t_step = repeat(t_step, '1 -> b', b=x_0.shape[0])
            x_t, _ = self.diffusion_model.forward_diffusion_model(x_0=x_0, t=t_step, noise=noise)
            # Remove one step of noise from x_t - backward process
            _ = self.diffusion_model.model(x_t, t_step)
            # Resize features so that they all live in the image space
            for idx in self._features:
                activations.append(nn.functional.interpolate(self._features[idx], size=[x_0.shape[-1]] * 2))
            # Return activations
        return torch.cat(activations, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        features = self.extract_features(x).to(x.device)
        out = self.classifier(features)
        return out
