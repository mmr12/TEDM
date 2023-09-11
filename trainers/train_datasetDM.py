from argparse import Namespace
import os
from pathlib import Path
from dataloaders.JSRT import build_dataloaders
import torch
from trainers.utils import seed_everything, TensorboardLogger
from torch.cuda.amp import GradScaler
from trainers.train_baseline import train
from models.datasetDM_model import DatasetDM



def main(config: Namespace) -> None:
    # adjust logdir to include experiment name
    os.makedirs(config.log_dir, exist_ok=True)
    print('Experiment folder: %s' % (config.log_dir))

    # save config namespace into logdir
    with open(config.log_dir / 'config.txt', 'w') as f:
        for k, v in vars(config).items():
            if type(v) not in [str, int, float, bool]:
                f.write(f'{k}: {str(v)}\n')
            else:
                f.write(f'{k}: {v}\n')

    # Random seed
    seed_everything(config.seed)

    model = DatasetDM(config)
    if config.shared_weights_over_timesteps:
        import torch.nn as nn
        from einops.layers.torch import Rearrange
        model.classifier = nn.Sequential(
            Rearrange('b (step act) h w -> (b step) act h w', step=len(model.steps)),
            nn.Conv2d(960, 128, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, config.out_channels)
            )
    model = model.to(config.device)
    model.train()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # , betas=config.adam_betas)
    step = 0

    scaler = GradScaler()

    dataloaders = build_dataloaders(
        config.data_dir,
        config.img_size,
        config.batch_size,
        config.num_workers,
        config.n_labelled_images
    )
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    # Logger
    logger = TensorboardLogger(config.log_dir, enabled=not config.debug)

    train(config, model, optimizer, train_dl, val_dl, logger, scaler, step)