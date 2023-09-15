import argparse
import os
from pathlib import Path
import torch
from torch import autocast, Tensor
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from config import parser
from einops import rearrange
from dataloaders.CXR14 import build_dataloaders
from models.global_local_cl import GlobalCL
from trainers.utils import (TensorboardLogger, compare_configs, seed_everything, crop_batch)


def save(model, optimizer, config, path, step):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step
    }, path)

def augment(x, img_size, batch_size):
    x = crop_batch([x], img_size, batch_size)                                      # random crop
    brightness = torch.rand((batch_size, 1, 1, 1), device=x.device)*.6 - .3        # random brightness adjustment between [-.3, .3]
    contrast = torch.rand((batch_size, 1, 1, 1), device=x.device)*.6 + .7          # random contrast adjustment between [.7, 1.3]
    x = (x + brightness) * contrast                                                # apply brightness and contrast
    return x

def augment_and_concat(x, img_size, batch_size):
    x_1 = augment(x, img_size, batch_size)
    x_2 = augment(x, img_size, batch_size)
    return torch.cat((x_1, x_2), dim=0) # 2b x c x h x w


def calculate_loss(features, batch_size, tau):
    norm_features = features / features.norm(dim=1, keepdim=True)
    similarity_matrix = torch.exp(norm_features @ norm_features.T / tau) # 2b x 2b [[b_1xb_1, b_1xb_2], [b_2xb_1, b_2xb_2]]
    positive_term_1 = torch.diagonal(similarity_matrix[:batch_size, batch_size:])
    negative_term_1 = similarity_matrix[:batch_size].sum(-1) - torch.diagonal(similarity_matrix[:batch_size, :batch_size]) # (b x 2b).sum(1) - (b_1 x b_1).diag() = b
    positive_term_2 = torch.diagonal(similarity_matrix[batch_size:, :batch_size])
    negative_term_2 = similarity_matrix[batch_size:].sum(-1) - torch.diagonal(similarity_matrix[batch_size:, batch_size:]) # (b x 2b).sum(1) - (b_2 x b_2).diag()= b
    loss = (-torch.log(positive_term_1 / negative_term_1).mean() - torch.log(positive_term_2 / negative_term_2).mean())/2
    return loss


@torch.no_grad()
def validate(config, model, val_loader):
    model.eval()
    losses = []
    for i, x in tqdm(enumerate(val_loader), desc='Validating'):
        batch_size = x.shape[0]
        x = x.to(config.device)
        x = augment_and_concat(x, config.img_size, batch_size) # 2b x c x h x w
        with autocast(device_type=config.device, enabled=config.mixed_precision):
            features = model(x) # 2b x emb_dim
            loss = calculate_loss(features, batch_size, config.tau)
        losses.append(loss.item())

        if i + 1 == config.max_val_steps or config.debug:
            break
    avg_loss = sum(losses) / len(losses)
    print(f'Validation loss: {avg_loss:.4f}')
    model.train()
    return {
        'val/loss': avg_loss,
    }

def train(config, model, optimizer, train_dl, val_dl, logger, scaler, step):
    best_val_loss = float('inf')
    train_losses = []
    pbar = tqdm(total=config.val_freq, desc='Training')
    while True:
        for x in train_dl:
            pbar.update(1)
            step += 1
            x = x.to(config.device)
            batch_size = x.shape[0]

            x = augment_and_concat(x, config.img_size, batch_size) # 2b x c x h x w
            optimizer.zero_grad()
            with autocast(device_type=config.device, enabled=config.mixed_precision):
                features = model(x) # 2b x emb_dim
                loss = calculate_loss(features, batch_size, config.tau)

            scaler.scale(loss).backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_description(f'Training loss: {loss.item():.4f}')
            if step % config.log_freq == 0 or config.debug:
                avg_train_loss = sum(train_losses) / len(train_losses)
                print(f'Step {step} - Train loss: {avg_train_loss:.4f}')
                logger.log({'train/loss': avg_train_loss}, step=step)

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

            if step >= config.max_steps or config.debug:
                return model


            # implementing 


def load(new_config, path):
    checkpoint = torch.load(path, map_location=torch.device(new_config.device))
    old_config = checkpoint['config']
    compare_configs(old_config, new_config)
    model = GlobalCL(
            img_size=old_config.img_size,
            dim=old_config.dim,
            dim_mults=old_config.dim_mults,
            channels=old_config.channels,
            out_dim=old_config.out_channels).to(new_config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=new_config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step


def main(config):
    # adjust logdir to include experiment name
    config.log_dir = Path(config.log_dir).parent / Path(config.log_dir).name
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
        encoder_model, optimizer, step = load(config, config.resume_path)
    else:
        encoder_model = GlobalCL(
            img_size=config.img_size,
            dim=config.dim,
            dim_mults=config.dim_mults,
            channels=config.channels,
            out_dim=config.out_channels)
        optimizer = torch.optim.Adam(encoder_model.parameters(), lr=config.lr)  # , betas=config.adam_betas)
        step = 0
    encoder_model.to(config.device)
    encoder_model.train()

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
    print('Train dataset size:', len(train_dl.dataset))
    print('Validation dataset size:', len(val_dl.dataset))

    # Logger
    logger = TensorboardLogger(config.log_dir, enabled=not config.debug)

    train(config, encoder_model, optimizer, train_dl, val_dl, logger, scaler, step)
