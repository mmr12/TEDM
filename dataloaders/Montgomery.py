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
# can be found at https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html


class MonDataset(Dataset):
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
        img = self.load_image(self.df.loc[i, "scan"])
        
        fnames = [ self.df.loc[i, l] for l in self.labels]
        labels = self.load_labels(fnames)

        return img, labels

    def __len__(self) -> int:
        return len(self.df)
