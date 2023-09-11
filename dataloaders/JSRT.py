from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, TypeVar, Optional
import pandas as pd
import os
from PIL import Image
from torch import Tensor
from torchvision import transforms
import torch

PathLike = TypeVar("PathLike", str, Path, None)

PROJECT_DIR = Path(os.path.realpath(__file__)).parent.parent
DATADIR = Path("<PATH_TO_DATA>/JSRT")
# can be found at http://db.jsrt.or.jp/eng.php

def build_dataloaders(
        data_dir: str=DATADIR,
        img_size: int=128,
        batch_size: int=16,
        num_workers: int=1,
        n_labelled_images: Optional[int] = None,
        **kwargs
) -> Tuple[List, List, List]:
    """
    Build dataloaders for the JSRT dataset.
    """
    train_ds = JSRTDataset(data_dir, PROJECT_DIR / "data", "JSRT_train_split.csv", img_size)
    if n_labelled_images is not None:
        train_ds = torch.utils.data.Subset(train_ds, range(n_labelled_images))
        print(f"Using {n_labelled_images} labelled images")
    val_ds = JSRTDataset(data_dir, PROJECT_DIR / "data", "JSRT_val_split.csv", img_size)
    test_ds = JSRTDataset(data_dir, PROJECT_DIR / "data", "JSRT_test_split.csv", img_size)

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


class JSRTDataset(Dataset):
    def __init__(self, base_path:PathLike, 
                 csv_path:PathLike,  
                 csv_name:str, 
                 img_size:int=128,
                 labels:List[str] =('right lung', 'left lung', ),
                 **kwargs) -> None:
        self.df = pd.read_csv(os.path.join(csv_path, csv_name))
        self.base_path = Path(base_path)
        self.labels = labels 
        self.img_size = img_size


    def load_image(self, fname: str) -> Tensor:
        img = Image.open(self.base_path /fname).convert('L').resize((self.img_size, self.img_size))
        img = transforms.ToTensor()(img).float()
        return img

    def load_labels(self, fnames: List[str]) -> Tensor:
        labels = []
        for fname in fnames:
            label = Image.open(self.base_path /fname).convert('L').resize((self.img_size, self.img_size))
            # convert to tensor
            label = transforms.ToTensor()(label).float()
            # make binary
            label = (label > .5).float()
            labels.append(label)
        # append all labels and merge
        label = torch.stack(labels).sum(0)
        # lungs have no overlap (right?)
        if (label > 1).sum()>0:
            print("overlapping lungs!", fnames)
            label = (label > .5)
        return label

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        i = self.df.index[index]
        img = self.load_image(self.df.loc[i, "path"])
        
        label_paths = ["SCR/masks/" + item + "/" + self.df.loc[i, 'id']+ ".gif" for item in self.labels]
        labels = self.load_labels(label_paths)

        return img, labels

    def __len__(self):
        return len(self.df)
