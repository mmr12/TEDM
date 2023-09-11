import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from tqdm.auto import tqdm
from torch import autocast
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from einops import rearrange
import sys
HEAD = Path(os.getcwd()).parent.parent
sys.path.append(HEAD)
from models.datasetDM_model import DatasetDM
from trainers.train_baseline import dice, precision, recall
from dataloaders.JSRT import build_dataloaders
from dataloaders.NIH import NIHDataset
from dataloaders.Montgomery import MonDataset

NIHPATH = "<PATH_TO_DATA>/NIH/"
NIHFILE = "correspondence_with_chestXray8.csv" # saved in data
MONPATH = "<PATH_TO_DATA>/NLM/MontgomerySet/"
MONFILE = "patient_data.csv"


if __name__ == "__main__":
    # load config file and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', "-e", type=str, help='Experiment path', default="logs/JSRT_conditional/20230213_171633")
    parser.add_argument('--rerun', "-r", help='Run the test again', default=False, action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.experiment):
        print("Experiment path identified as a directory")
    else:
        raise ValueError("Experiment path is not a directory")
    files = os.listdir(args.experiment)
    torch_file = None
    if {'JSRT_val_predictions.pt', 'JSRT_test_predictions.pt', 'NIH_predictions.pt', 'Montgomery_predictions.pt'} <= set(files) and not args.rerun:
        print("Experiment already tested")
        sys.exit(0)

    for f in files:
        if "model" in f:
            torch_file = f
            break
    if torch_file is None:
        raise ValueError("No checkpoint file found in experiment directory")
    
    print(f"Loading experiment from {torch_file}")
    data = torch.load(Path(args.experiment) / torch_file)
    config = data["config"]

    # pick model
    if config.experiment == "datasetDM":
        model = DatasetDM(config)
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
    else:
        raise ValueError(f"Experiment {config.experiment} not recognized")
    model.load_state_dict(data['model_state_dict'])

    # Gather model output
    model.eval().to(config.device)

    # Load data
    dataloaders = build_dataloaders(
        config.data_dir,
        config.img_size,
        config.batch_size,
        config.num_workers,
    )
    datasets_to_test = {
        "JSRT_val": dataloaders["val"],
        "JSRT_test": dataloaders["test"],
        "NIH": DataLoader(NIHDataset(NIHPATH, NIHPATH, NIHFILE, config.img_size), 
                          config.batch_size, num_workers=config.num_workers),
        "Montgomery": DataLoader(MonDataset(MONPATH, MONPATH, MONFILE, config.img_size),
                                 config.batch_size, num_workers=config.num_workers)

    }

    for dataset_key in datasets_to_test:
        if f"{dataset_key}_predictions.pt" in files and not args.rerun:
            print(f"{dataset_key} already tested")
            output = torch.load(Path(args.experiment) / f'{dataset_key}_predictions.pt')
            print(f"{dataset_key} metrics: \n\tdice:      {output['dice'].mean():.3}+/-{output['dice'].std():.3}")
            print(f"\tprecision: {output['precision'].mean():.3}+/-{output['precision'].std():.3}")
            print(f"\trecall:    {output['recall'].mean():.3}+/-{output['recall'].std():.3}")
            continue

        print(f"Testing {dataset_key} set")
        y_hats = []
        y_star = []
        for i, (x, y) in tqdm(enumerate(datasets_to_test[dataset_key]), desc='Validating'):
            x = x.to(config.device)

            with autocast(device_type=config.device, enabled=config.mixed_precision):
                with torch.no_grad():
                    # all depths
                    pred = torch.sigmoid(model(x))
            y_hats.append(pred.detach().cpu())
            y_star.append(y)
        
        # save predictions
        y_star = torch.cat(y_star, 0)
        y_hats = torch.cat(y_hats, 0)
        y_hats = rearrange(y_hats, '(b step) 1 h w -> step b 1 h w', step=len(model.steps))
        for i, y_hat in enumerate(y_hats):
            output = {
                'y_hat': y_hat, 
                'y_star': y_star,
                'dice':dice(y_hat>.5, y_star),
                'precision':precision(y_hat>.5, y_star),
                'recall':recall(y_hat>.5, y_star),}
            
            print(f"{dataset_key} {model.steps[i]} metrics: \n\tdice:      {output['dice'].mean():.3}+/-{output['dice'].std():.3}")
            print(f"\tprecision: {output['precision'].mean():.3}+/-{output['precision'].std():.3}")
            print(f"\trecall:    {output['recall'].mean():.3}+/-{output['recall'].std():.3}")
            torch.save(output, Path(args.experiment) / f'{dataset_key}_timestep{model.steps[i]}_predictions.pt')
        y_hat = y_hats.mean(0)
        output = {
                'y_hat': y_hat, 
                'y_star': y_star,
                'dice':dice(y_hat>.5, y_star),
                'precision':precision(y_hat>.5, y_star),
                'recall':recall(y_hat>.5, y_star),}
            
        print(f"{dataset_key} metrics: \n\tdice:      {output['dice'].mean():.3}+/-{output['dice'].std():.3}")
        print(f"\tprecision: {output['precision'].mean():.3}+/-{output['precision'].std():.3}")
        print(f"\trecall:    {output['recall'].mean():.3}+/-{output['recall'].std():.3}")
        torch.save(output, Path(args.experiment) / f'{dataset_key}_predictions.pt')
    