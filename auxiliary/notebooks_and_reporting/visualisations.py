# %%
import numpy as np
import torch
from pathlib import Path
import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
HEAD = Path(os.getcwd()).parent.parent
head = HEAD / 'logs'
sys.path.append(HEAD)
from dataloaders.JSRT import JSRTDataset
from dataloaders.NIH import NIHDataset
from dataloaders.Montgomery import MonDataset
NIHPATH = "<PATH_TO_DATA>/NIH/"
NIHFILE = "correspondence_with_chestXray8.csv"
MONPATH = "<PATH_TO_DATA>/MontgomerySet/"
MONFILE = "patient_data.csv"
JSRTPATH = "<PATH_TO_DATA>/JSRT"

if __name__=="__main__":
    predictions = {'baseline':{'JSRT':{}, 'NIH':{}, 'Montgomery':{}},
                'LEDM':{'JSRT':{}, 'NIH':{}, 'Montgomery':{}},
                'TEDM':{'JSRT':{}, 'NIH':{}, 'Montgomery':{}},}
    files_needed = ["JSRT_val_predictions.pt", "JSRT_test_predictions.pt",  "NIH_predictions.pt", "Montgomery_predictions.pt",]
    for exp in ['baseline', 'LEDM', "TEDM"]:
        for datasize in [1,3,6,12,24,49,98,197]:
            if len(set(files_needed) - set(os.listdir(head / exp / str(datasize) ))) == 0:
                for file in files_needed[1:]:
                    output = torch.load(head / exp / str(datasize) / file)
                    metrics_datasize = 197 if datasize == "None" else int(datasize)
                    predictions[exp][file.rsplit("_")[0]][metrics_datasize]= output['y_hat']
            else:
                    print(f"Experiment {exp} is missing files")
    # %%

    img_size = 128
    NIH_dataset = NIHDataset(NIHPATH, NIHPATH, NIHFILE, img_size)
    JSRT_dataset = JSRTDataset(JSRTPATH, HEAD/ "data/", "JSRT_test_split.csv", img_size)
    MON_dataset = MonDataset(MONPATH, MONPATH, MONFILE, img_size)
    
    # %%
    loaders = {'JSRT': JSRT_dataset, 'NIH': NIH_dataset, 'Montgomery': MON_dataset}
    m ="dice"
    sz=4
    ftsize= 40
    fig, all_axs = plt.subplots(6, 21, figsize=(21*sz, 6*sz))
    all_patients = [17, 13, 0, 1, 72, 78]

    # JSRT
    dataset ="JSRT"
    patient = np.random.randint(0, len(loaders[dataset]))
    patient = all_patients[0]
    print("JSRT1 - ", patient)
    out = loaders[dataset][patient]
    axs = all_axs[:3, :7]
    for rowax, exp in zip(axs, ['baseline', 'LEDM', 'TEDM']):
        rowax[0].imshow(out[0][0].numpy(), cmap='gray')
        rowax[1].imshow(out[1][0].numpy(), interpolation='none', cmap='gray')
        for ax, dssize in zip(rowax[2:], [1, 3, 6, 12, 197]):
            ax.imshow(predictions[exp][dataset][dssize][patient].numpy()[0]>.5, interpolation='none')
    axs[0, 0].set_title("JSRT - Image", fontsize=ftsize)
    axs[0, 1].set_title("JSRT - GT", fontsize=ftsize)
    axs[0, 2].set_title("1 (1%)"  , fontsize=ftsize)
    axs[0, 3].set_title("3 (2%)", fontsize=ftsize)
    axs[0, 4].set_title("6 (3%)", fontsize=ftsize)
    axs[0, 5].set_title("12 (6%)", fontsize=ftsize)
    axs[0, 6].set_title("197 (100%)", fontsize=ftsize)
    axs[0,0].set_ylabel("Baseline", fontsize=ftsize)
    axs[1,0].set_ylabel("LEDM", fontsize=ftsize)
    axs[2,0].set_ylabel("TEDM", fontsize=ftsize)
    #
    axs = all_axs[3:, :7]
    dataset ="JSRT"
    patient = np.random.randint(0, len(loaders[dataset]))
    patient = all_patients[1]
    print("JSRT2 - ", patient)
    out = loaders[dataset][patient]
    for rowax, exp in zip(axs, ['baseline', 'LEDM', 'TEDM']):
        rowax[0].imshow(out[0][0].numpy(), cmap='gray')
        rowax[1].imshow(out[1][0].numpy(), interpolation='none', cmap='gray')
        for ax, dssize in zip(rowax[2:], [1, 3, 6, 12, 197]):
            ax.imshow(predictions[exp][dataset][dssize][patient].numpy()[0]>.5, interpolation='none')
    axs[0,0].set_ylabel("Baseline", fontsize=ftsize)
    axs[1,0].set_ylabel("LEDM", fontsize=ftsize)
    axs[2,0].set_ylabel("TEDM", fontsize=ftsize)
    #
    axs = all_axs[:3, 7:14]
    dataset ="NIH"
    patient = np.random.randint(0, len(loaders[dataset]))
    patient = all_patients[2]
    print("NIH1 - ", patient)
    out = loaders[dataset][patient]
    for rowax, exp in zip(axs, ['baseline', 'LEDM', 'TEDM']):
        rowax[0].imshow(out[0][0].numpy(), cmap='gray')
        rowax[1].imshow(out[1][0].numpy(), interpolation='none', cmap='gray')
        for ax, dssize in zip(rowax[2:], [1, 3, 6, 12, 197]):
            ax.imshow(predictions[exp][dataset][dssize][patient].numpy()[0]>.5, interpolation='none')
    axs[0, 0].set_title("NIH - Image", fontsize=ftsize)
    axs[0, 1].set_title("NIH - GT", fontsize=ftsize)
    axs[0, 2].set_title("1 (1%)"  , fontsize=ftsize)
    axs[0, 3].set_title("3 (2%)", fontsize=ftsize)
    axs[0, 4].set_title("6 (3%)", fontsize=ftsize)
    axs[0, 5].set_title("12 (6%)", fontsize=ftsize)
    axs[0, 6].set_title("197 (100%)", fontsize=ftsize)
    #
    #
    axs = all_axs[3:, 7:14]
    dataset ="NIH"
    patient = np.random.randint(0, len(loaders[dataset]))
    patient = all_patients[3]
    print("NIH2 - ", patient)
    out = loaders[dataset][patient]
    for rowax, exp in zip(axs, ['baseline', 'LEDM', 'TEDM']):
        rowax[0].imshow(out[0][0].numpy(), cmap='gray')
        rowax[1].imshow(out[1][0].numpy(), interpolation='none', cmap='gray')
        for ax, dssize in zip(rowax[2:], [1, 3, 6, 12, 197]):
            ax.imshow(predictions[exp][dataset][dssize][patient].numpy()[0]>.5, interpolation='none')
    #
    #
    axs = all_axs[:3, 14:]
    dataset ="Montgomery"
    patient = np.random.randint(0, len(loaders[dataset]))
    patient = all_patients[4]
    print("MON1 - ",patient)
    out = loaders[dataset][patient]
    for rowax, exp in zip(axs, ['baseline', 'LEDM', 'TEDM']):
        rowax[0].imshow(out[0][0].numpy(), cmap='gray')
        rowax[1].imshow(out[1][0].numpy(), interpolation='none', cmap='gray')
        for ax, dssize in zip(rowax[2:], [1, 3, 6, 12, 197]):
            ax.imshow(predictions[exp][dataset][dssize][patient].numpy()[0]>.5, interpolation='none')
    axs[0, 0].set_title("Mont. - Image", fontsize=ftsize)
    axs[0, 1].set_title("Mont. - GT", fontsize=ftsize)
    axs[0, 2].set_title("1 (1%)", fontsize=ftsize)
    axs[0, 3].set_title("3 (2%)", fontsize=ftsize)
    axs[0, 4].set_title("6 (3%)", fontsize=ftsize)
    axs[0, 5].set_title("12 (6%)", fontsize=ftsize)
    axs[0, 6].set_title("197 (100%)", fontsize=ftsize)
    #
    axs = all_axs[3:, 14:]
    dataset ="Montgomery"
    patient = np.random.randint(0, len(loaders[dataset]))
    patient = all_patients[5]
    print("MON2 - ",patient)
    out = loaders[dataset][patient]
    for rowax, exp in zip(axs, ['baseline', 'LEDM', 'TEDM']):
        rowax[0].imshow(out[0][0].numpy(), cmap='gray')
        rowax[1].imshow(out[1][0].numpy(), interpolation='none', cmap='gray')
        for ax, dssize in zip(rowax[2:], [1, 3, 6, 12, 197]):
            ax.imshow(predictions[exp][dataset][dssize][patient].numpy()[0]>.5, interpolation='none')


    # remove ticks
    for ax in all_axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    plt.subplots_adjust(wspace=0.00, 
                        hspace=0.00)
    plt.tight_layout()
    plt.savefig("visualisations2.pdf", bbox_inches='tight')
    plt.show()
