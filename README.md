# Novel Diffusion Model (DM) based approach for semi-supervised medical image segmentation  


## Training

- training the backbone

```python train.py --dataset CXR14 --data_dir <PATH TO CXR14 DATASET>```

- our method

```python train.py --experiment TEDM --data_dir <PATH TO JSRT DATASET> --n_labelled_images <TRAINING SET SIZE>```

- LEDM method

```python train.py --experiment LEDM --data_dir <PATH TO JSRT DATASET> --n_labelled_images <TRAINING SET SIZE>```

- LEDMe method

```python train.py --experiment LEDMe --data_dir <PATH TO JSRT DATASET> --n_labelled_images <TRAINING SET SIZE>```

- baseline method

```python train.py --experiment JSRT_baseline --data_dir <PATH TO JSRT DATASET> --n_labelled_images <TRAINING SET SIZE>```

## Testing

- update 
    - `DATADIR` in paths `dataloaders/JSRT.py`, `dataloaders/NIH.py` and `dataloaders/Montgomery.py`
    - `NIHPATH`, `NIHFILE`, `MONPATH` and `MONFILE` in paths `auxiliary/postprocessing/run_tests.py` and `auxiliary/postprocessing/testing_shared_weights.py`

- for baseline and LEDM methods, run

```python auxiliary/postprocessing/run_tests.py --experiment <PATH TO LOG FOLDER>```

- for our method, run

```python auxiliary/postprocessing/testing_shared_weights.py --experiment <PATH TO LOG FOLDER>```

## Figures and reporting

VS Code notebooks can be found in `auxiliary/notebooks_and_reporting`.
