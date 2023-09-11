import argparse
from pathlib import Path
import os
import torch
from tqdm.auto import tqdm
from torch import autocast
from torch.utils.data import DataLoader
import sys
HEAD = Path(os.getcwd()).parent.parent
sys.path.append("/vol/biomedic3/mmr12/projects/TEDM/")
from models.diffusion_model import DiffusionModel
from models.unet_model import Unet
from models.datasetDM_model import DatasetDM
from trainers.datasetDM_per_step import ModDatasetDM
from trainers.train_baseline import dice, precision, recall
from dataloaders.JSRT import build_dataloaders
from dataloaders.NIH import NIHDataset
from dataloaders.Montgomery import MonDataset


NIHPATH = "/vol/biodata/data/chest_xray/NIH/"
NIHFILE = "correspondence_with_chestXray8.csv"
MONPATH = "/vol/biodata/data/chest_xray/NLM/MontgomerySet/"
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
        for file in ['JSRT_val_predictions.pt', 'JSRT_test_predictions.pt', 'NIH_predictions.pt', 'Montgomery_predictions.pt']:
            output = torch.load(Path(args.experiment) / file)
            dataset_key = file.split("_")[0]
            print(f"{dataset_key} metrics: \n\tdice:      {output['dice'].mean():.3}+/-{output['dice'].std():.3}")
            print(f"\tprecision: {output['precision'].mean():.3}+/-{output['precision'].std():.3}")
            print(f"\trecall:    {output['recall'].mean():.3}+/-{output['recall'].std():.3}")
            #torch.save(output, Path(args.experiment) / f'{dataset_key}_predictions.pt')
        exit(0)

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
    if config.experiment in ["baseline", "global_finetune", "glob_loc_finetune"]:
        model = Unet(**vars(config))
    elif config.experiment == "datasetDM":
        model = DatasetDM(config)
    elif config.experiment == "simple_datasetDM":
        model = ModDatasetDM(config)
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
    if config.experiment == "simple_datasetDM":
        # re-calculate mean and var as they were not saved in the model dict
        train_dl = dataloaders["train"]
        for x, _ in tqdm(train_dl, desc="Calculating mean and variance"):
            x = x.to(config.device)
            features = model.extract_features(x)
            model.mean += features.sum(dim=0)
            model.mean_squared += (features ** 2).sum(dim=0)
        model.mean = model.mean / len(train_dl.dataset)
        model.std = (model.mean_squared / len(train_dl.dataset) - model.mean ** 2).sqrt() + 1e-6

        model.mean = model.mean.to(config.device)
        model.std = model.std.to(config.device)
    
    for dataset_key in datasets_to_test:
        if f"{dataset_key}_predictions.pt" in files and not args.rerun:
            print(f"{dataset_key} already tested")
            output = torch.load(Path(args.experiment) / f'{dataset_key}_predictions.pt')
            print(f"{dataset_key} metrics: \n\tdice:      {output['dice'].mean():.3}+/-{output['dice'].std():.3}")
            print(f"\tprecision: {output['precision'].mean():.3}+/-{output['precision'].std():.3}")
            print(f"\trecall:    {output['recall'].mean():.3}+/-{output['recall'].std():.3}")
            continue

        print(f"Testing {dataset_key} set")
        y_hat = []
        y_star = []
        for i, (x, y) in tqdm(enumerate(datasets_to_test[dataset_key]), desc='Validating'):
            x = x.to(config.device)

            if config.experiment == "conditional":
                # sample n = 5 different segmetations
                y_hats = []
                for _ in range(5):
                    img = torch.randn(x.shape, device=config.device)
                    for t in tqdm(range(0, config.timesteps)[::-1]):
                        # sample next timestep image (x_{t-1})
                        with autocast(device_type=config.device, enabled=config.mixed_precision):
                            with torch.no_grad():
                                img = model.sample_timestep(img, t=t, cond=x)
                    y_hats.append(img.detach().cpu() / 2 + .5)
                # take the average over the 5 samples
                y_hats = torch.stack(y_hats, -1).mean(-1)   

                # record
                y_hat.append(y_hats)
                y_star.append(y)

            elif config.experiment in ["baseline", "datasetDM",  "simple_datasetDM", "global_finetune", "glob_loc_finetune"] :
                with autocast(device_type=config.device, enabled=config.mixed_precision):
                    with torch.no_grad():
                        pred = torch.sigmoid(model(x))
                y_hat.append(pred.detach().cpu())
                y_star.append(y)

            else:
                raise ValueError(f"Experiment {config.experiment} not recognized")
        
        # save predictions
        y_hat = torch.cat(y_hat, 0)
        y_star = torch.cat(y_star, 0)
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