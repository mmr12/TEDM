import logging
import os
from typing import List, Tuple, TypeVar
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torchvision import transforms
from pathlib import Path

PathLike = TypeVar("PathLike", str, Path, None)
log = logging.getLogger(__name__)


PROJECT_DIR = Path(os.path.realpath(__file__)).parent.parent
DATADIR = Path("<PATH_TO_DATA>/ChestXray-NIHCC/images")
# can be found at https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345


def build_dataloaders(
        data_dir: str=DATADIR,
        img_size: int=128,
        batch_size: int=16,
        num_workers: int=1,
) -> Tuple[List, List, List]:
    """
    Build dataloaders for the CXR14 dataset.
    """
    train_ds = CXR14Dataset(data_dir, PROJECT_DIR / 'data' / 'train_split.csv', img_size)
    val_ds = CXR14Dataset(data_dir, PROJECT_DIR / 'data' / 'train_split.csv', img_size)
    test_ds = CXR14Dataset(data_dir, PROJECT_DIR / 'data' / 'train_split.csv', img_size)

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True)
    dataloaders['val'] = DataLoader(val_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True)
    dataloaders['test'] = DataLoader(test_ds, batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers,
                                     pin_memory=True)

    return dataloaders



class CXR14Dataset(Dataset):
    def __init__(
        self,
        data_path: PathLike,
        csv_path: PathLike,
        img_size: int,
    ) -> None:
        super().__init__()
        assert(os.path.isdir(data_path))
        assert(os.path.isfile(csv_path))

        self.data_path = Path(data_path)
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, fname: str) -> Tensor:
        img = Image.open(self.data_path /fname).convert('L').resize((self.img_size, self.img_size))
        img = transforms.ToTensor()(img).float()
        return img

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        img = self.load_image(self.df.loc[index, "Image Index"])
        return img
