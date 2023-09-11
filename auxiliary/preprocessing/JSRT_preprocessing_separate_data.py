# %%
import pandas as pd
from pathlib import Path
import numpy as np
import os

CWDIR = Path(os.getcwd()).parent.parent


head = Path("<PATH_TO_DATA>/JSRT")

df = pd.read_csv(head / "jsrt_metadata_with_masks.csv")
df.reset_index(inplace=True)

# %% Shuffle and remove 20% for test and val
idx = np.arange(len(df))
np.random.shuffle(idx)
n1 = int(len(df)*.8)
n2 = int(len(df)*.9)
idxs = [idx[:n1], idx[n1:n2], idx[n2:]]
for i in range(3):
    print(len(df.loc[idxs[i]]))
# %%
df.loc[idxs[0]].to_csv(CWDIR / 'data' / 'JSRT_train_split.csv', index=False)
df.loc[idxs[1]].to_csv(CWDIR / 'data' / 'JSRT_val_split.csv', index=False)
df.loc[idxs[2]].to_csv(CWDIR / 'data' / 'JSRT_test_split.csv', index=False)