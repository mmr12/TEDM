import torch
import os
from argparse import Namespace
from pathlib import Path
from tqdm.auto import tqdm
from torch import autocast
from einops import rearrange, reduce, repeat
from torch.cuda.amp import GradScaler
from torch.nn.functional import binary_cross_entropy_with_logits
from trainers.utils import seed_everything, TensorboardLogger
from dataloaders.JSRT import build_dataloaders as build_dataloaders_JSRT
from models.unet_model import Unet
from trainers.train_base_diffusion import save



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
            pbar.update(1)
            step += 1
            if config.shared_weights_over_timesteps and config.experiment == 'datasetDM':
                y = repeat(y, 'b c h w -> (b step) c h w', step=len(model.steps))
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
            if config.shared_weights_over_timesteps and config.experiment == 'datasetDM':
                loss_per_timestep = reduce(expanded_loss, '(b step) c -> step', 'mean', step=len(model.steps))
                train_losses_per_timestep.append(loss_per_timestep.detach().cpu())

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


@torch.no_grad()
def validate(config, model, val_dl):
    model.eval()
    metrics = {
        'val/loss': [],
        'val/dice': [],
        'val/precision': [],
        'val/recall': [],
    }
    for i, (x, y) in tqdm(enumerate(val_dl), desc='Validating'):
        x = x.to(config.device)

        with autocast(device_type=config.device, enabled=config.mixed_precision):
            pred = model(x).detach().cpu()
        
        # label predictions
        if pred.shape[1] == 1:
            y_hat = torch.sigmoid(pred) > .5
        else:
            y_hat = torch.argmax(pred, dim=1)
            y_hat = torch.stack([y_hat == i for i in range(y.shape[1])], dim=1)
        # metrics
        if config.shared_weights_over_timesteps and config.experiment == 'datasetDM':
            y = repeat(y, 'b c h w -> (b step) c h w', step=len(model.steps))
        metrics['val/dice'].append(dice(y_hat, y))
        metrics['val/precision'].append(precision(y_hat, y))
        metrics['val/recall'].append(recall(y_hat, y))
        metrics['val/loss'].append(binary_cross_entropy_with_logits(pred, y, reduction='none'))
        
        if i + 1 == config.max_val_steps or config.debug:
            break

    # average metrics
    avg_loss = torch.cat(metrics['val/loss']).mean()
    print(f'Validation loss: {avg_loss:.4f}')
    if y_hat.shape[1] > 1:
        for i in range(1, y_hat.shape[1]):
            metrics[f'val_dice/{i}'] = torch.cat(metrics['val/dice'])[:, i].nanmean().item()
            metrics[f'val_precision/{i}'] = torch.cat(metrics['val/precision'])[:, i].nanmean().item()
            metrics[f'val_recall/{i}'] = torch.cat(metrics['val/recall'])[:,i].nanmean().item()
    metrics['val/loss'] = avg_loss.item()
    metrics['val/dice'] = torch.cat(metrics['val/dice']).nanmean().item() # exclude background + exclude classes not represented (through nanmean)
    metrics['val/precision'] = torch.cat(metrics['val/precision']).nanmean().item()
    metrics['val/recall'] = torch.cat(metrics['val/recall']).nanmean().item()
    model.train()
    return metrics

def dice(x_hat, x):
    x_n_x_hat = torch.logical_and(x_hat, x)
    dice = 2 * reduce(x_n_x_hat, 'b c h w -> b c', 'sum') / (reduce(x_hat, 'b c h w -> b c', 'sum') + reduce(x, 'b c h w -> b c', 'sum'))
    return dice

def precision(x_hat, x):
    TP = reduce(torch.logical_and(x, x_hat), 'b c h w -> b c', 'sum')
    FP = reduce(torch.logical_and(1 - x, x_hat), 'b c h w -> b c', 'sum')
    _precision = TP / (TP + FP)
    return _precision

def recall(x_hat, x):
    TP = reduce(torch.logical_and(x, x_hat), 'b c h w -> b c', 'sum')
    FN = reduce(torch.logical_and(x, ~x_hat), 'b c h w -> b c', 'sum')
    _recall = TP / (TP + FN)
    return _recall


def main(config:Namespace) -> None:
    
    # adjust logdir to include experiment name
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

    model = Unet(
            config.dim,
            dim_mults=config.dim_mults,
            channels=config.channels,
            out_dim=config.out_channels
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) 
    step = 0
    model.to(config.device)
    model.train()

    scaler = GradScaler()

    # Load data
    if config.dataset == "JSRT":
        build_dataloaders = build_dataloaders_JSRT
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    dataloaders = build_dataloaders(
        config.data_dir,
        config.img_size,
        config.batch_size,
        config.num_workers,
        config.n_labelled_images,
    )
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    print(f'Loaded {len(train_dl.dataset)} training and {len(val_dl.dataset)} validation images')

    # Logger
    logger = TensorboardLogger(config.log_dir, enabled=not config.debug)
    train(config, model, optimizer, train_dl, val_dl, logger, scaler, step)