from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, TypeVar
import pandas as pd
import os
from PIL import Image
from torch import Tensor
from torchvision import transforms


PathLike = TypeVar("PathLike", str, Path, None)
# can be found at https://www.kaggle.com/datasets/nih-chest-xrays/data

class NIHDataset(Dataset):
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

    def load_labels(self, fname: str) -> Tensor:

        label = Image.open(self.base_path /fname).convert('L').resize((self.img_size, self.img_size))
        # convert to tensor
        label = transforms.ToTensor()(label).float()
        # make binary
        label = (label > .5).float()
        
        return label

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        i = self.df.index[index]
        img = self.load_image(self.df.loc[i, "scan"])
        labels = self.load_labels(self.df.loc[i, "mask"])

        return img, labels

    def __len__(self):
        return len(self.df)
