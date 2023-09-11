import os
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from dataloaders.JSRT import build_dataloaders as build_dataloaders_JSRT
from models.diffusion_model import DiffusionModel
from trainers.utils import (TensorboardLogger, compare_configs, sample_plot_image,
                   seed_everything)


def train(config, model, optimizer, train_loader, val_loader, logger, scaler, step):
    best_val_loss = float('inf')
    train_losses = []
    if config.joint_training and config.experiment == 'joint_and_cond':
        img_losses = []
        seg_losses = []
    pbar = tqdm(total=config.val_freq, desc='Training')
    cond = None
    while True:
        for x, y in train_loader:
            pbar.update(1)
            step += 1

            if config.experiment == "joint":
                x = torch.concat([x,y], dim=1)
            elif config.experiment == "conditional":
                cond = x # condition on the image
                cond = cond.to(config.device)
                x = y # predict the segmentation
            elif config.experiment == "joint_and_cond":
                cond = y.to(config.device)
            x = x.to(config.device)

            # Forward pass
            optimizer.zero_grad()
            with autocast(device_type=config.device, enabled=config.mixed_precision):
                loss = model.train_step(x, cond)
            if config.joint_training and config.experiment == 'joint_and_cond':
                img_loss, seg_loss = loss
                loss = img_loss + seg_loss
                img_losses.append(img_loss.item())
                seg_losses.append(seg_loss.item())
            
            scaler.scale(loss).backward()
            optimizer.step()

            train_losses.append(loss.item())
            if config.joint_training and config.experiment == 'joint_and_cond':
                pbar.set_description(f'Training loss: {loss.item():.4f} (img: {img_loss.item():.4f}, seg: {seg_loss.item():.4f})')
            else:
                pbar.set_description(f'Training loss: {loss.item():.4f}')

            if step % config.log_freq == 0 or config.debug:
                avg_train_loss = sum(train_losses) / len(train_losses)
                print(f'Step {step} - Train loss: {avg_train_loss:.4f}')
                logger.log({'train/loss': avg_train_loss}, step=step)
                if config.joint_training and config.experiment == 'joint_and_cond':
                    avg_img_loss = sum(img_losses) / len(img_losses)
                    avg_seg_loss = sum(seg_losses) / len(seg_losses)
                    logger.log({'train_loss/img': avg_img_loss}, step=step)
                    logger.log({'train_loss/seg': avg_seg_loss}, step=step)

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
                        config.log_dir / 'best_model.pt',
                        step
                    )

            if step >= config.max_steps or config.debug:
                return model


@torch.no_grad()
def validate(config, model, val_loader):
    model.eval()
    losses = []
    if config.joint_training and config.experiment == 'joint_and_cond':
        img_losses = []
        seg_losses = []
    cond=None
    for i, (x, y) in tqdm(enumerate(val_loader), desc='Validating'):
        if config.experiment == "joint":
            x = torch.cat([x,y], dim=1)
        elif config.experiment == "conditional":
            cond = x # condition on the image
            cond = cond.to(config.device)
            x = y # predict the segmentation
        elif config.experiment == "joint_and_cond":
            cond = y.to(config.device)
        x = x.to(config.device)

        with autocast(device_type=config.device, enabled=config.mixed_precision):
            if len(val_loader.dataset) > 1000:
                # val set is too large to evaluate for each timestep, take random timesteps
                loss = model.train_step(x, cond)
            else:
                loss = model.val_step(x, cond, t_steps=config.val_steps)
        if config.joint_training and config.experiment == 'joint_and_cond':
            img_loss, seg_loss = loss
            loss = img_loss + seg_loss
            img_losses.append(img_loss.item())
            seg_losses.append(seg_loss.item())

        losses.append(loss.item())

        if i + 1 == config.max_val_steps or config.debug:
            break

    avg_loss = sum(losses) / len(losses)
    if config.joint_training and config.experiment == 'joint_and_cond':
        avg_img_loss = sum(img_losses) / len(img_losses)
        avg_seg_loss = sum(seg_losses) / len(seg_losses)
        print(f'Validation loss: {avg_loss:.4f} (img: {avg_img_loss:.4f}, seg: {avg_seg_loss:.4f})')
    else:
        print(f'Validation loss: {avg_loss:.4f}')

    # Build visualisations
    # select images for visualisation
    if config.experiment == "conditional":
        # note that there is no randomness on how the images are selected
        # if there are enough images on the last validation batch, then we keep those
        # otherwise, we take the first images from a new validation epoch batch
        config.n_sampled_imgs = config.n_sampled_imgs if not config.debug else 1
        N_missing = config.n_sampled_imgs - cond.shape[0]
        iter_val_loader = iter(val_loader) # new validation epoch
        while N_missing > 0: # are we missing images?
            new_cond, _ = next(iter_val_loader)
            cond = torch.cat([cond, new_cond], dim=0)
            N_missing = config.n_sampled_imgs - cond.shape[0]
        del iter_val_loader
        if N_missing < 0: # do we have too many images?
            cond = cond[:config.n_sampled_imgs]
    
    with autocast(device_type=config.device, enabled=config.mixed_precision):
        sampled_imgs = sample_plot_image(
            model,
            config.timesteps,
            config.img_size,
            config.n_sampled_imgs, # this defaults to 1 if in debug mode
            config.channels if config.experiment != "joint_and_cond" else config.channels + config.out_channels, # out_channels stands for channels in segmentation
            cond=cond,
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
    model = DiffusionModel(old_config)        
    model.to(new_config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=new_config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return model, optimizer, step


def main(config):
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

    # Init model and optimizer
    if config.resume_path is not None:
        print('Loading model from', config.resume_path)
        diffusion_model, optimizer, step = load(config, config.resume_path)
    else:
        if config.experiment in ["img_only", "joint", "conditional"]:
            diffusion_model = DiffusionModel(config)
        else:
            raise ValueError(f"Unknown experiment: {config.experiment}")

        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # , betas=config.adam_betas)
        step = 0
    diffusion_model.to(config.device)
    diffusion_model.train()

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

    # Logger
    logger = TensorboardLogger(config.log_dir, enabled=not config.debug)

    train(config, diffusion_model, optimizer, train_dl, val_dl, logger, scaler, step)
