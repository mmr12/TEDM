import argparse
import os

from datetime import datetime
from pathlib import Path
import torch


this_dir = os.path.dirname(os.path.realpath(__file__))
default_logdir = os.path.join(this_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--mixed_precision', type=bool, default=False, help='Use mixed precision')
parser.add_argument('--resume_path', type=str, default=None, help='Path to checkpoint to resume from')

# Experiment parameters
parser.add_argument('--experiment', type=str, default="img_only",choices=[
    "PDDM",
    "baseline", 
    "LEDM", 
    "LEDMe", 
    "TEDM",
    "global_cl",
    "local_cl",
    "global_finetune",
    "glob_loc_finetune"
    ], help='Whether to generate only images or images and segmentations')
parser.add_argument('--dataset', type=str, default="JSRT",choices=["JSRT", "CXR14"], help='Dataset to use')

# Data parameters
parser.add_argument('--img_size', type=int, default=128, help='Height / width of the input image to the network')
parser.add_argument('--data_dir', type=str, help='Path to the dataset')
parser.add_argument('--num_workers', type=int, default=4, help='Number of subprocesses to use for data loading')

# Model parameters
parser.add_argument('--dim', type=int, default=64, help='Width of the U-Net')
parser.add_argument('--dim_mults', nargs='+', type=int, default=(1, 2, 4, 8), help='Dimension multipliers for U-Net levels')
# SegDiff model parameters
parser.add_argument('--seg_out_dim', type=int, default=1, help='Dimension of segmentation embedding')
parser.add_argument('--img_out_dim', type=int, default=4, help='Dimension of image embedding')
parser.add_argument('--img_inter_dim', type=int, default=32, help='Width of image embedding')

# Diffusion parameters
parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'])
parser.add_argument('--objective', type=str, default='pred_noise', help='Model output', choices=['pred_noise', 'pred_x_0'])

# CL parameters
parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter for contrastive loss')
parser.add_argument('--global_model_path', type=str, default=None, help='Path to global model checkpoint')
parser.add_argument('--glob_loc_model_path', type=str, default=None, help='Path to global & local CL model checkpoint')
parser.add_argument('--unfreeze_weights_at_step', type=int, default=0, help='Step at which to unfreeze pretrained weights. If 0, weights are not frozen')
parser.add_argument('--augment_at_finetuning', default=False, action='store_true', help='Whether to augment images during finetuning')

# Training parameters
parser.add_argument('--batch_size', type=int, default=16, help='Input batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
# parser.add_argument('--adam_betas', nargs=2, type=float, default=(0.9, 0.99), help='Betas for the Adam optimizer')
parser.add_argument('--max_steps', type=int, default=500000, help='Number of training steps to perform')
parser.add_argument('--p2_loss_weight_gamma', type=float, default=0., help='p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended')
parser.add_argument('--p2_loss_weight_k', type=float, default=1.)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--seed', type=int, default=0, help='Random seed')

# Logging parameters
parser.add_argument('--log_freq', type=int, default=100, help='Frequency of logging')
parser.add_argument('--val_freq', type=int, default=100, help='Frequency of validation')
parser.add_argument('--val_steps', type=int, default=250, help='Number of timestep to use for validation')
parser.add_argument('--log_dir', type=str, default=default_logdir, help='Logging directory')
parser.add_argument('--n_sampled_imgs', type=int, default=8, help='Number of images to sample during logging')
parser.add_argument('--max_val_steps', type=int, default=-1, help='Number of validation steps to perform')

# datasetGAN like segmentation model parameters
parser.add_argument("--saved_diffusion_model", type=str, help='Path to checkpoint of trained diffusion model', default="logs/20230127_164150/best_model.pt")
parser.add_argument("--t_steps_to_save", type=int, nargs='*', choices=range(1000), help='Diffusion steps to be used as features', default=[50, 200, 400, 600, 800])
parser.add_argument("--n_labelled_images", type=int, help='Number of labelled images to use for semi-supervised training', default=None, 
                    choices=[197, 98, 49, 24, 12, 6, 3, 1])

# other experiments I played with
parser.add_argument("--shared_weights_over_timesteps", help='In datasetDM, only use last timestep to predict, and intermediate timesteps to train', default=False, action='store_true')
parser.add_argument("--early_stop", help='In baseline, if validation loss increases by more than 50%, stop', default=False, action='store_true')