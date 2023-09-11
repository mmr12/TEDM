# %%
import pandas as pd
from pathlib import Path
import numpy as np
import os

CWDIR = Path(os.getcwd()).parent.parent
DATADIR = Path("<PATH_TO_DATA>/ChestXray-NIHCC")
if not os.path.isdir(DATADIR):
    print(f"Data directory {DATADIR} not found")


df = pd.concat([pd.read_csv(DATADIR / "train_val_list.csv"),pd.read_csv(DATADIR / "test_list.csv")])
df.reset_index(inplace=True)
# %%
from tqdm import tqdm
items = []
for el in tqdm(df["Image Index"]):
    items.append(os.path.isfile(DATADIR / "images"/ el))
# %% Shuffle and remove 20% for test and val
idx = np.arange(len(df))
np.random.shuffle(idx)
n1 = int(len(df)*.8)
n2 = int(len(df)*.9)
idxs = [idx[:n1], idx[n1:n2], idx[n2:]]
for i in range(3):
    print(len(df.loc[idxs[i]]))
# %%
df.loc[idxs[0]].to_csv(CWDIR / 'data' / 'train_split.csv', index=False)
df.loc[idxs[1]].to_csv(CWDIR / 'data' / 'val_split.csv', index=False)
df.loc[idxs[2]].to_csv(CWDIR / 'data' / 'test_split.csv', index=False)