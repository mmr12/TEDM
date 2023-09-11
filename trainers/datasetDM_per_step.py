from argparse import Namespace
import os
from pathlib import Path
from dataloaders.JSRT import build_dataloaders
import torch
from tqdm.auto import tqdm
from trainers.utils import seed_everything, TensorboardLogger
from torch.cuda.amp import GradScaler
from torch import Tensor, nn
from typing import Dict, Optional
from trainers.train_baseline import train
from models.datasetDM_model import DatasetDM
from einops import repeat
from einops.layers.torch import Rearrange


class ModDatasetDM(DatasetDM):
    # the idea here is to pool info per timestep, 
    # so that we can then use the aggregate for feature importance

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.mean = torch.zeros(len(self.steps) * 960, args.img_size, args.img_size, requires_grad=False)
        self.mean_squared = torch.zeros(len(self.steps) * 960, args.img_size, args.img_size, requires_grad=False)
        self.std = torch.zeros(len(self.steps) * 960, args.img_size, args.img_size, requires_grad=False)
        self.classifier = nn.Conv2d(len(self.steps) * 960, 1, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        features = self.extract_features(x).to(x.device)
        out = (features - self.mean ) / self.std
        out = self.classifier(features)
        return out
    
class OneStepPredDatasetDM(DatasetDM):
    # the idea here is to pool info per timestep, 
    # so that we can then use the aggregate for feature importance

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.mean = torch.zeros(len(self.steps) * 960, args.img_size, args.img_size, requires_grad=False)
        self.mean_squared = torch.zeros(len(self.steps) * 960, args.img_size, args.img_size, requires_grad=False)
        self.std = torch.zeros(len(self.steps) * 960, args.img_size, args.img_size, requires_grad=False)
        self.classifier = nn.Sequential(
            Rearrange('b (step act) h w -> (b step) act h w', step=len(self.steps)),
            nn.Conv2d(960, 128, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, args.out_channels)
            )
    

    def forward(self, x: Tensor) -> Tensor:
        features = self.extract_features(x).to(x.device)
        out = (features - self.mean ) / self.std
        out = self.classifier(features)
        return out


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

    model = ModDatasetDM(config)
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


    # do a loop to calculate mean and variance of the features
    # then use those to normalize the features
    model.to(config.device)
    for x, _ in tqdm(train_dl, desc="Calculating mean and variance"):
        x = x.to(config.device)
        features = model.extract_features(x)
        model.mean += features.sum(dim=0)
        model.mean_squared += (features ** 2).sum(dim=0)
    model.mean = model.mean / len(train_dl.dataset)
    model.std = (model.mean_squared / len(train_dl.dataset) - model.mean ** 2).sqrt() + 1e-6

    model.mean = model.mean.to(config.device)
    model.std = model.std.to(config.device)
        
    train(config, model, optimizer, train_dl, val_dl, logger, scaler, step)
