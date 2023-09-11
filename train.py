from config import parser
import argparse
from pathlib import Path
from trainers.train_CXR14 import main as train_CXR14
from trainers.train_baseline import main as train_baseline
from trainers.train_base_diffusion import main as train_JSRT
from trainers.train_datasetDM import main as train_datasetDM
from trainers.datasetDM_per_step import main as train_simple_datasetDM
from trainers.train_global_cl import main as train_global_cl
from trainers.train_local_cl import main as train_local_cl
from trainers.finetune_glob_cl import main as train_global_finetune
from trainers.finetune_glob_loc_cl import main as train_global_local_finetune


if __name__=="__main__":
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    config = parser.parse_args()

    # catch exeptions
    #if len(config.loss_weights) != 4:
    #    raise ValueError('loss_weights must be a list of 4 values')
    
    config.normalize = True
    config.log_dir = Path(config.log_dir).parent / config.experiment / str(config.n_labelled_images) /  Path(config.log_dir).name
    config.channels = 1
    config.out_channels = 1
    if config.dataset == "CXR14":
        config.data_dir = Path("<PATH_TO_DATA>/ChestXray-NIHCC/images")
    elif config.dataset == "JSRT":
        config.data_dir = Path("<PATH_TO_DATA>/JSRT")
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    

    if config.experiment == "img_only":
        train_CXR14(config)
    elif config.experiment == "baseline":
        train_baseline(config)
    elif config.experiment == "LEDM":
        config.t_steps_to_save = [50, 150, 250]
        train_datasetDM(config)
    elif config.experiment == "LEDMe":
        config.t_steps_to_save = [1, 10, 25, 50, 200, 400, 600, 800]
        train_datasetDM(config)
    elif config.experiment == "TEDM":
        config.shared_weights_over_timesteps = True
        config.t_steps_to_save = [1, 10, 25, 50, 200, 400, 600, 800]
        train_datasetDM(config)
    elif config.experiment == 'global_cl':
        train_global_cl(config)
    elif config.experiment == 'local_cl':
        train_local_cl(config)
    elif config.experiment == 'global_finetune':
        train_global_finetune(config)
    elif config.experiment == 'glob_loc_finetune':
        train_global_local_finetune(config)
