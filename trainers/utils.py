import os
import random

from argparse import Namespace
from inspect import isfunction
from numbers import Number
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(img: Tensor) -> Tensor:
    return (img + 1) * 0.5


def exists(x: Any) -> bool:
    """Checks if value is None"""
    return x is not None


def default(val: Any, d: Any) -> Any:
    """Returns d if val is None else val"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def get_index_from_list(
    vals: Tensor,
    t: Tensor,
    x_shape: Tuple[int, ...]
) -> Tensor:
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def sample_plot_image(diffusion_model, T: int, img_size: int, batch: int, channels:Optional[int]=1, cond:Optional[Tensor]=None,) -> Tensor:
    """_summary_

    Args:
        diffusion_model (nn.Module):    Diffusion model   
        T (int):                        Total number of diffusion timesteps
        img_size (int):                 Image size
        batch (int):                    Number of images to sample
        channels (optional):            Number of channels in the image. 
                                        For medical images this is usually one, but can be two if the second channel is the segmentation. 
                                        Defaults to 1.

    Returns:
        grid:                           Grid of randomly sampled images
    """
    device = next(diffusion_model.parameters()).device
    img = torch.randn((batch, channels, img_size, img_size), device=device)

    num_samples_per_img = 8
    stepsize = int(T / num_samples_per_img)
    imgs = []

    for t in range(0, T)[::-1]:
        # sample next timestep image (x_{t-1})
        img = diffusion_model.sample_timestep(img, t=t, cond=cond)  # (batch, channels, h, w) 
        if t % stepsize == 0:
            imgs.append(unnormalize_to_zero_to_one(img.detach().cpu()))

    imgs = torch.stack(imgs)  # (n_samples, batch, channels, h, w)
    imgs = rearrange(imgs, "n b c h w -> b n c h w", )
    grids = torch.stack([make_grid(img_row, nrow=4) for img_row in imgs]) # b n c h w -> b c H W where H = (h * n / 4) and W = w * 4
    if channels > 1:
        grids = rearrange(grids, "b c H W -> c b H W")
        grids = make_grid(grids, nrow=1) # c b H W -> b H W where H <- H * c
        grids = rearrange(grids, "b H W -> b 1 H W")
    return grids  # (batch, 1, H, W)


class TensorboardLogger(SummaryWriter):
    def __init__(
        self,
        log_dir: str = None,
        config: Namespace = None,
        enabled: bool = True,
        comment: str = '',
        purge_step: int = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = ''
    ):
        self.enabled = enabled
        if self.enabled:
            super().__init__(
                log_dir=log_dir,
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix
            )
        else:
            return

        # Add config
        if config is not None:
            self.add_hparams(
                {k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool, torch.Tensor))},
                {}
            )

    def log(self, data: Dict[str, Any], step: int) -> None:
        """Log each entry in data as its corresponding data type"""
        if self.enabled:
            for k, v in data.items():
                # Scalars
                if isinstance(v, Number):
                    self.add_scalar(k, v, step)

                # Images
                elif (isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)) and len(v.shape) >= 3:
                    if len(v.shape) == 3:
                        self.add_image(k, v, step)
                    elif len(v.shape) == 4:
                        self.add_images(k, v, step)
                    else:
                        raise ValueError(f'Unsupported image shape: {v.shape}')

                else:
                    raise ValueError(f'Unsupported data type: {type(v)}')


def compare_configs(config_old: Namespace, config_new: Namespace) -> bool:
    """
    Compares two configs and returns True if they are equal.
    """
    c_old = vars(config_old)
    c_new = vars(config_new)

    # Changed values
    for k, v in c_old.items():
        if k in c_new and c_new[k] != v:
            print(f'{k} differs - old: {v} new: {c_new[k]}')

    # New keys
    for k, v in c_new.items():
        if k not in c_old:
            print(f'{k} is new - {v}')

    # Removed keys
    for k, v in c_old.items():
        if k not in c_new:
            print(f'{k} is removed - {v}')


# adapted from https://github.com/krishnabits001/domain_specific_cl/blob/e5aae802fe906de8c46ed4dd26b2c75edb7abe39/utils.py#L526
# to be used with pytorch tensors + adding random box size
def crop_batch(ip_list,img_size,batch_size,box_dim_min=96,box_dim_y_min=96,low_val=0,high_val=32):
    '''
    To select a cropped part of the image and resize it to original dimensions
    input param:
        ip_list: input list of image, labels
        cfg: contains config settings of the image
        batch_size: batch size value
        box_dim_x,box_dim_y: co-ordinates of the cropped part of the image to be select and resized to original dimensions
        low_val : lowest co-ordinate value allowed as starting point of the cropped window
        low_val : highest co-ordinate value allowed as starting point of the cropped window
    return params:
        ld_img_re_bs: cropped images that are resized into original dimensions
        ld_lbl_re_bs: cropped masks that are resized into original dimensions

    '''
    #ld_label_batch = np.squeeze(np.zeros_like(ld_img_batch))
    #box_dim = 100  # 100*100
    if(len(ip_list)==2):
        ld_img_batch=ip_list[0]
        ld_label_batch=ip_list[1]
        ld_img_re_bs=torch.zeros_like(ld_img_batch)
        ld_lbl_re_bs=torch.zeros_like(ld_label_batch)
    else:
        ld_img_batch=ip_list[0]
        ld_img_re_bs=torch.zeros_like(ld_img_batch)

    x_dim,y_dim=img_size,img_size

    box_dim_arr_x=torch.randint(low=low_val,high=high_val,size=(batch_size,))
    box_dim_arr_y=torch.randint(low=low_val,high=high_val,size=(batch_size,))

    for index in range(0, batch_size):
        
        x,y=box_dim_arr_x[index],box_dim_arr_y[index]

        box_dim=torch.randint(low=box_dim_min,high=x_dim-x,size=(1,)).item()
        box_dim_y=torch.randint(low=box_dim_y_min,high=y_dim-y,size=(1,)).item()

        if(len(ip_list)==2):
            im_crop = ld_img_batch[index,:,x:x + box_dim, y:y + box_dim_y]
            ld_img_re_bs[index]=Resize((x_dim,y_dim))(im_crop)
            lbl_crop = ld_label_batch[index, :,x:x + box_dim, y:y + box_dim_y]
            ld_lbl_re_bs[index]=torch.round(Resize((x_dim,y_dim))(lbl_crop))
        else:
            im_crop = ld_img_batch[index,:,x:x + box_dim, y:y + box_dim_y]
            ld_img_re_bs[index]=Resize((x_dim,y_dim))(im_crop)

    if(len(ip_list)==2):
        return ld_img_re_bs,ld_lbl_re_bs
    else:
        return ld_img_re_bs