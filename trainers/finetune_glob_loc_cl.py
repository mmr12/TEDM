import argparse
import os
from pathlib import Path
import torch
from torch import autocast, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from config import parser
from einops import rearrange, reduce, repeat
from dataloaders.JSRT import build_dataloaders
from models.unet_model import Unet
from trainers.train_baseline import validate, save
from trainers.utils import (TensorboardLogger, compare_configs, seed_everything, crop_batch)


def train(config, model, optimizer, train_dl, val_dl, logger, scaler, step):
    best_val_loss = float('inf')
    train_losses = []
    if config.dataset == "BRATS2D":
        train_losses_per_class = []
    elif config.shared_weights_over_timesteps and config.experiment == 'datasetDM':
        train_losses_per_timestep = []
        
    pbar = tqdm(total=config.val_freq, desc='Training')
    while True:
        for x, y in train_dl:
            if config.shared_weights_over_timesteps and config.experiment == 'datasetDM':
                y = repeat(y, 'b c h w -> (b step) c h w', step=len(model.steps))
            if config.augment_at_finetuning:
                x, y = crop_batch([x, y], config.img_size, config.batch_size)
                brightness = torch.rand((config.batch_size, 1, 1, 1), device=x.device)*.6 - .3        # random brightness adjustment between [-.3, .3]
                contrast = torch.rand((config.batch_size, 1, 1, 1), device=x.device)*.6 + .7          # random contrast adjustment between [.7, 1.3]
                x = (x + brightness) * contrast                                                # apply brightness and contrast

            x = x.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad()
            with autocast(device_type=config.device, enabled=config.mixed_precision):
                pred = model(x)
                # cross entropy loss
                #loss = - ((y * torch.log(torch.sigmoid(pred)) + (1 - y) * torch.log(1 - torch.sigmoid(pred)))).mean()
                if config.dataset == "BRATS2D":
                    weights = repeat(torch.Tensor(config.loss_weights).to(config.device), 'c -> b c h w', b=y.shape[0], h=y.shape[2], w=y.shape[3])
                else:
                    weights = None
                expanded_loss = reduce(binary_cross_entropy_with_logits(pred, y, weight=weights, reduction='none'), 'b c h w -> b c', 'mean') 
                loss = expanded_loss.mean()
            scaler.scale(loss).backward()
            optimizer.step()

            train_losses.append(loss.item())
            if config.dataset == "BRATS2D":
                loss_per_class = expanded_loss.mean(0)
                train_losses_per_class.append(loss_per_class.detach().cpu())
                pbar.set_description(f'Training loss: {loss.item():.4f} - {loss_per_class[0].item():.4f} - {loss_per_class[1].item():.4f} - {loss_per_class[2].item():.4f} - {loss_per_class[3].item():.4f}')
            else:
                pbar.set_description(f'Training loss: {loss.item():.4f}')

            pbar.update(1)
            step += 1
            
            if config.unfreeze_weights_at_step == step:
                for name, param in model.named_parameters():
                    if name.startswith('downs') or name.startswith('init_conv') or name.startswith('mid_'):
                        param.requires_grad = True

            if step % config.log_freq == 0 or config.debug:
                avg_train_loss = sum(train_losses) / len(train_losses)
                print(f'Step {step} - Train loss: {avg_train_loss:.4f}')
                logger.log({'train/loss': avg_train_loss}, step=step)
                if config.dataset == "BRATS2D":
                    avg_train_loss_per_class = torch.stack(train_losses_per_class).mean(0)
                    logger.log({'train_loss/0':avg_train_loss_per_class[0].item()}, step=step)
                    logger.log({'train_loss/1':avg_train_loss_per_class[1].item()}, step=step)
                    logger.log({'train_loss/2':avg_train_loss_per_class[2].item()}, step=step)
                    logger.log({'train_loss/3':avg_train_loss_per_class[3].item()}, step=step)
                if config.shared_weights_over_timesteps and config.experiment == 'datasetDM':
                    avg_train_loss_per_timestep = torch.stack(train_losses_per_timestep).mean(0)
                    for i, model_step in enumerate(model.steps):
                        logger.log({'train_loss/step_' + str(model_step): avg_train_loss_per_timestep[i].item()}, step=step)

            if step % config.val_freq == 0 or config.debug:
                val_results = validate(config, model, val_dl)
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
                        config.log_dir / 'best_model.pt',
                        step
                    )
                elif val_results['val/loss'] > best_val_loss * 1.5 and config.early_stop:
                    print(f'Step {step} - Validation loss increased by more than 50%')
                    return model

            if step >= config.max_steps or config.debug:
                return model




def load(config, path):
    raise NotImplementedError

def main(config):

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
        model, optimizer, step = load(config, config.resume_path)
    else:
        model = Unet(
            img_size=config.img_size,
            dim=config.dim,
            dim_mults=config.dim_mults,
            channels=config.channels,
            out_dim=config.out_channels)
        state_dict = torch.load(config.glob_loc_model_path, map_location='cpu')['model_state_dict']
        out = model.load_state_dict(state_dict=state_dict, strict=False)
        print("Loaded state dict. \n\tMissing keys: {}\n\tUnexpected keys: {}".format(out.missing_keys, out.unexpected_keys))
        print('Note that although the state dict of the decoder is loaded, its values are random.')
        if config.unfreeze_weights_at_step !=0:
            for name, param in model.named_parameters():
                if name.startswith('downs') or name.startswith('init_conv') or name.startswith('mid_'):
                    param.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # , betas=config.adam_betas)
        
        step = 0
    model.to(config.device)
    model.train()

    scaler = GradScaler()

    # Load data
    dataloaders = build_dataloaders(
        config.data_dir,
        config.img_size,
        config.batch_size,
        config.num_workers,
        n_labelled_images=config.n_labelled_images,
    )
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    print('Train dataset size:', len(train_dl.dataset))
    print('Validation dataset size:', len(val_dl.dataset))

    # Logger
    logger = TensorboardLogger(config.log_dir, enabled=not config.debug)

    train(config, model, optimizer, train_dl, val_dl, logger, scaler, step)