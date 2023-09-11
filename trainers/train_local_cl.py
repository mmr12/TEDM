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
from models.global_local_cl import LocalCL
from trainers.train_global_cl import augment_and_concat, save
from trainers.utils import (TensorboardLogger, compare_configs, seed_everything, crop_batch)

def calculate_loss_elements(logits, batch_size, n_regions, diag_offset):
    pos_mask = torch.zeros((batch_size * n_regions*2, batch_size * n_regions*2), device=logits.device) + \
                torch.diag(torch.ones(batch_size * n_regions * 2 - abs(-n_regions * batch_size + diag_offset), device=logits.device), diagonal=-n_regions * batch_size + diag_offset) +\
                    torch.diag(torch.ones(batch_size * n_regions * 2 - abs(n_regions * batch_size + diag_offset), device=logits.device), diagonal=n_regions * batch_size + diag_offset)
               #torch.diag(torch.ones(batch_size*n_regions-diag_offset, device=logits.device), diagonal= batch_size*n_regions + diag_offset) + \
               #torch.diag(torch.ones(batch_size*n_regions+diag_offset, device=logits.device), diagonal=-batch_size*n_regions + diag_offset)
    pos_mask[:batch_size*n_regions, :batch_size*n_regions] = 0
    pos_mask[batch_size*n_regions:, batch_size*n_regions:] = 0
    
    neg_mask = torch.zeros((batch_size * n_regions*2, batch_size * n_regions*2), device=logits.device)
    for region in range(-2*n_regions+1,2*n_regions):
        neg_mask += torch.diag(torch.ones(batch_size * n_regions * 2 - abs(region * batch_size + diag_offset), device=logits.device), diagonal=region * batch_size + diag_offset) 
    neg_mask[:batch_size*n_regions, :batch_size*n_regions] = 0
    neg_mask[batch_size*n_regions:, batch_size*n_regions:] = 0
    #neg_mask = torch.diag(torch.ones(batch_size - abs(diag_offset), device=logits.device), diagonal=diag_offset).repeat(n_regions * 2, n_regions * 2)
    #neg_mask -= torch.diag(torch.ones(batch_size * n_regions * 2 - abs(diag_offset), device=logits.device), diag_offset)

    pos_logits = (logits*pos_mask).sum(1)
    neg_logits = torch.exp(logits*neg_mask).mean(1)
    return pos_logits[pos_mask.sum(1).bool()], neg_logits[pos_mask.sum(1).bool()]


def  calculate_loss(features, batch_size, tau):
    n_regions = 20 # sample 15x15 regions from each image
    # sample 15x15 3x3 regions from each image

    x_center_samples = torch.randperm(features.shape[2]-2).to(features.device)[:n_regions]+1
    y_center_samples = torch.randperm(features.shape[3]-2).to(features.device)[:n_regions]+1
    # one sample
    regions = torch.stack([features[:,:,x_center_samples[i]-1:x_center_samples[i]+2, y_center_samples[i]-1:y_center_samples[i]+2] for i in range(n_regions)], dim=1) # (n_views x b) x n_regions x emb_dim x 3 x 3
    regions = rearrange(regions, 'bn r c h w -> bn r (c h w)') # (n_views x b) x n_regions x (emb_dim x 3 x 3)
    norm_regions = regions / regions.norm(dim=2, keepdim=True)

    contrast_feature = torch.cat(torch.unbind(norm_regions, dim=1), dim=0) # b_1.reg1, b_1.reg2, ..., b_2.reg1, b_2.reg2, ...
    # compute logits - note: no numerical stability tricks here
    logits = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), tau
        ) 
    loss = 0
    for diag_offset in range(-batch_size + 1, batch_size):
        pos_logits, neg_logits = calculate_loss_elements(logits, batch_size, n_regions, diag_offset)
        loss += (- pos_logits + neg_logits).mean()
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
                pbar = tqdm(total=config.val_freq, desc='Training')

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
        partial_model, optimizer, step = load(config, config.resume_path)
    else:
        partial_model = LocalCL(
            img_size=config.img_size,
            dim=config.dim,
            dim_mults=config.dim_mults,
            channels=config.channels,
            out_dim=config.out_channels)
        state_dict = torch.load(config.global_model_path, map_location='cpu')['model_state_dict']
        partial_model.load_state_dict(state_dict=state_dict, strict=False)
        params_to_optimise = []
        names_to_optimise = []
        for name, param in partial_model.ups[:partial_model.l].named_parameters():
            params_to_optimise.append(param)
            names_to_optimise.append(name)
        optimizer = torch.optim.Adam(params_to_optimise, lr=config.lr)  # , betas=config.adam_betas)
        # freeze the remaining layers
        names_to_optimise = [f'ups.{n}' for n in names_to_optimise]
        for name, param in partial_model.named_parameters():
            if name not in names_to_optimise:
                param.requires_grad = False
        step = 0
    partial_model.to(config.device)
    partial_model.train()

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

    train(config, partial_model, optimizer, train_dl, val_dl, logger, scaler, step)