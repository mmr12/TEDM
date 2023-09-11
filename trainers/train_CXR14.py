import argparse
import os
from pathlib import Path
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from config import parser
from dataloaders.CXR14 import build_dataloaders
from models.diffusion_model import DiffusionModel
from trainers.utils import (TensorboardLogger, compare_configs, sample_plot_image,
                   seed_everything)


def train(config, model, optimizer, train_loader, val_loader, logger, scaler, step):
    best_val_loss = float('inf')
    train_losses = []
    pbar = tqdm(total=config.val_freq, desc='Training')
    while True:
        for x in train_loader:
            pbar.update(1)
            step += 1

            x = x.to(config.device)

            # Forward pass
            optimizer.zero_grad()
            with autocast(device_type=config.device, enabled=config.mixed_precision):
                loss = model.train_step(x)
            scaler.scale(loss).backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_description(f'Training loss: {loss.item():.4f}')

            if step % config.log_freq == 0 or config.debug:
                avg_train_loss = sum(train_losses) / len(train_losses)
                print(f'Step {step} - Train loss: {avg_train_loss:.4f}')
                logger.log({'train/loss': avg_train_loss}, step=step)

            if step % config.val_freq == 0 or config.debug:
                val_results = validate(config, model, val_loader)
                logger.log(val_results, step=step)

                if val_results['val/loss'] < best_val_loss and not config.debug:
                    print(f'Step {step} - New best validation loss: '
                          f'{val_results["val/loss"]:.4f}, saving model '
                          f'in {config.log_dir}')
                    best_val_loss = val_results['val/loss']
                    save(
                        model,
                        optimizer,
                        config,
                        config.log_dir + '/best_model.pt',
                        step
                    )

            if step >= config.max_steps or config.debug:
                return model


@torch.no_grad()
def validate(config, model, val_loader):
    model.eval()
    losses = []
    for i, x in tqdm(enumerate(val_loader), desc='Validating'):
        x = x.to(config.device)

        with autocast(device_type=config.device, enabled=config.mixed_precision):
            loss = model.train_step(x)
        losses.append(loss.item())

        if i + 1 == config.max_val_steps or config.debug:
            break

    avg_loss = sum(losses) / len(losses)
    print(f'Validation loss: {avg_loss:.4f}')

    with autocast(device_type=config.device, enabled=config.mixed_precision):
        sampled_imgs = sample_plot_image(
            model,
            config.timesteps,
            config.img_size,
            config.n_sampled_imgs if not config.debug else 1,
            normalized=config.normalize,
        )

    model.train()
    return {
        'val/loss': avg_loss,
        'val/sampled images': sampled_imgs
    }


def save(model, optimizer, config, path, step):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step
    }, path)


def load(new_config, path):
    checkpoint = torch.load(path, map_location=torch.device(new_config.device))
    old_config = checkpoint['config']
    compare_configs(old_config, new_config)
    model = DiffusionModel(old_config).to(new_config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=new_config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step


def main(config):
    # adjust logdir to include experiment name
    config.log_dir = Path(config.log_dir).parent / "CXR14" / Path(config.log_dir).name
    os.makedirs(config.log_dir, exist_ok=True)
    
    # save config namespace into logdir
    with open(config.log_dir / 'config.txt', 'w') as f:
        for k, v in vars(config).items():
            if type(v) not in [str, int, float, bool]:
                f.write(f'{k}: {str(v)}\n')
            else:
                f.write(f'{k}: {v}\n')

    # Random seed
    seed_everything(config.seed)

    # Init model and optimizer
    if config.resume_path is not None:
        print('Loading model from', config.resume_path)
        diffusion_model, optimizer, step = load(config, config.resume_path)
    else:
        diffusion_model = DiffusionModel(config)
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=config.lr)  # , betas=config.adam_betas)
        step = 0
    diffusion_model.to(config.device)
    diffusion_model.train()

    scaler = GradScaler()

    # Load data
    dataloaders = build_dataloaders(
        config.data_dir,
        config.img_size,
        config.batch_size,
        config.num_workers,
    )
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    # Logger
    logger = TensorboardLogger(config.log_dir, enabled=not config.debug)

    train(config, diffusion_model, optimizer, train_dl, val_dl, logger, scaler, step)
