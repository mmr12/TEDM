# %%
import numpy as np
import torch
from pathlib import Path
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
HEAD = Path(os.getcwd()).parent.parent

if __name__=="__main__":
    # load baseline and LEDM data
    metrics = {"dice": [], "precision": [], "recall": [], "exp": [], "datasize": [], "dataset":[]}
    files_needed = ["JSRT_val_predictions.pt", "JSRT_test_predictions.pt",  "NIH_predictions.pt", "Montgomery_predictions.pt",]
    head = HEAD / 'logs'
    for exp in ['baseline', 'LEDM']:
         for datasize in [1, 3, 6, 12, 24, 49, 98, 197]:
            if len(set(files_needed) - set(os.listdir(head / exp / str(datasize)))) == 0:
                print(f"Experiment {exp} {datasize}")
                output = torch.load(head / exp / str(datasize) / "JSRT_val_predictions.pt")
                print(f"{output['dice'].mean()}\t{output['dice'].std()}")
                for file in files_needed[1:]:
                    output = torch.load(head / exp / str(datasize)  / file)
                    metrics_datasize = 197 if datasize == "None" else int(datasize)
                    metrics["dice"].append(output["dice"].numpy())
                    metrics["precision"].append(output["precision"].numpy())
                    metrics["recall"].append(output["recall"].numpy())
                    metrics["exp"].append(np.array([exp] * len(output["dice"])))
                    metrics["datasize"].append(np.array([int(datasize)] * len(output["dice"])))
                    metrics["dataset"].append(np.array([file.split("_")[0]]*len(output["dice"])))
            else:
                    print(f"Experiment {exp} is missing files")

    for key in metrics:
        metrics[key] = np.concatenate([el.squeeze() for el in metrics[key]])
    df = pd.DataFrame(metrics)
    df.head()


    # %% Load step data
    metrics2 = {"dice": [], "precision": [], "recall": [], "exp": [], "datasize": [], "dataset":[], 'timestep':[]}
    for timestep in [1, 10, 25, 50, 500, 950]:
        exp = f"Step_{timestep}"
        for datasize in [197, 98, 49, 24, 12, 6, 3, 1]:
            if os.path.isdir(head / exp / str(datasize)):
                    if len(set(files_needed) - set(os.listdir(head / exp / str(datasize)))) == 0:
                        print(f"Experiment {datasize} {timestep}")
                        output = torch.load(head / exp / str(datasize)/  "JSRT_val_predictions.pt")
                        print(f"{output['dice'].mean()}\t{output['dice'].std()}")
                        for file in files_needed[1:]:
                            output = torch.load(head / exp / str(datasize) / file)
                            metrics_datasize = datasize if datasize is not None else 197
                            metrics2["dice"].append(output["dice"].numpy())
                            metrics2["precision"].append(output["precision"].numpy())
                            metrics2["recall"].append(output["recall"].numpy())
                            metrics2["exp"].append(np.array([exp] * len(output["dice"])))
                            metrics2["datasize"].append(np.array([metrics_datasize] * len(output["dice"])))
                            metrics2["dataset"].append(np.array([file.split("_")[0]]*len(output["dice"])))
                            metrics2["timestep"].append(np.array([timestep] * len(output["dice"])))
                    else:
                            print(f"Experiment {datasize} is missing files")


    for key in metrics2:
        metrics2[key] = np.concatenate(metrics2[key]).squeeze()
        print(key, metrics2[key].shape)
    df2 = pd.DataFrame(metrics2)

    # %%  figure with line for baseline and datasetDM and boxplots for the rest
    #  separating dice from precision and recall
    font = 16
    x = [1, 1, 3, 3, 6, 6, 12, 12, 24, 24, 49, 49, 197, 197]
    plot_x = np.concatenate([np.array([-.4, .4]) + i for i in range(len(x)//2)]).flatten()
    fig, axs = plt.subplots(3, 1, figsize=[12, 10])
    sns.set_style("whitegrid")
    m = 'dice'
    for i, dataset in enumerate(["JSRT", "NIH", "Montgomery"]):
        ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'baseline') & (df.datasize == _x), m].to_numpy() for _x in x])
        ys_std = np.quantile(ys, (.25, .75), axis=1, )
        axs[i ].fill_between(plot_x, ys_std[0], ys_std[1], alpha=.2, zorder=0, color='C6')
        ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'LEDM') & (df.datasize == _x), m].to_numpy() for _x in x])
        ys_std = np.quantile(ys, (.25, .75), axis=1, )
        axs[i ].fill_between(plot_x, ys_std[0], ys_std[1], alpha=.2, zorder=0, color='C8')
        ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'baseline') & (df.datasize == _x), m].to_numpy() for _x in x])
        ys_mean = np.quantile(ys, .5, axis=1)
        axs[i ].plot(plot_x, ys_mean, label="baseline", c='C6', zorder=0)
        ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'LEDM') & (df.datasize == _x), m].to_numpy() for _x in x])
        ys_mean = np.quantile(ys, .5, axis=1)
        axs[i ].plot(plot_x, ys_mean, label="LEDM" , c='C7', zorder=0)


    for i, dataset in enumerate(["JSRT", "NIH", "Montgomery"]):
        temp_df = df2[(df2.dataset == dataset) & (df2.datasize != 98)]
        out = sns.boxplot(data=temp_df, x="datasize", y=m, hue="timestep", ax=axs[i ],  showfliers=False, saturation=1,)
        axs[i ].set_title(f"{dataset}", fontsize=font)
        axs[i ].set_xlabel("" )
        y_min, _ = axs[i ].get_ylim()
        axs[i ].set_ylim(y_min, 1)
        h, l = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
        axs[i].set_ylabel("Dice", fontsize=font)
    sns.despine(ax=axs[0 ], offset=10, trim=True, bottom=True)
    sns.despine(ax=axs[1 ], offset=10, trim=True, bottom=True)
    sns.despine(ax=axs[2 ], offset=10, trim=True)
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[-1 ].set_xlabel("Training dataset size", fontsize=font)
    # Shrink current axis by 20%
    for i, ax in enumerate(axs):
        box = ax.get_position()
        ax.tick_params(axis='both', labelsize=font)
        ax.set_position([box.x0, box.y0, box.width , box.height])

    # Put a legend to the right of the current axis

    fig.legend(h, ['baseline', 'LEDM'] + ['step ' + _l for _l in l[2:]], title="", ncol=4, 
            loc='center left', bbox_to_anchor=(0.2, -0.03), fontsize=font)
    plt.tight_layout()
    #plt.savefig("results_per_timestep.png")
    plt.savefig("results_per_timestep_dice.pdf", bbox_inches='tight')
    plt.show()
    # %%
    x = [1, 1, 3, 3, 6, 6, 12, 12, 24, 24, 49, 49, 197, 197]
    plot_x = np.concatenate([np.array([-.4, .4]) + i for i in range(len(x)//2)]).flatten()
    fig, axs = plt.subplots(3, 2, figsize=[15, 15])
    sns.set_style("whitegrid")
    for j, m in enumerate(["precision", "recall"]):
        for i, dataset in enumerate(["JSRT", "NIH", "Montgomery"]):
            ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'baseline') & (df.datasize == _x), m].to_numpy() for _x in x])
            ys_std = np.quantile(ys, (.25, .75), axis=1, )
            axs[i, j].fill_between(plot_x, ys_std[0], ys_std[1], alpha=.2, zorder=0, color='C6')
            ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'LEDM') & (df.datasize == _x), m].to_numpy() for _x in x])
            ys_std = np.quantile(ys, (.25, .75), axis=1, )
            axs[i, j].fill_between(plot_x, ys_std[0], ys_std[1], alpha=.2, zorder=0, color='C8')
            ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'baseline') & (df.datasize == _x), m].to_numpy() for _x in x])
            ys_mean = np.quantile(ys, .5, axis=1)
            axs[i, j].plot(plot_x, ys_mean, label="baseline", c='C6', zorder=0)
            ys = np.stack([df.loc[(df.dataset == dataset)& (df.exp == 'LEDM') & (df.datasize == _x), m].to_numpy() for _x in x])
            ys_mean = np.quantile(ys, .5, axis=1)
            axs[i, j].plot(plot_x, ys_mean, label="LEDM" , c='C7', zorder=0)


            ##
            temp_df = df2[(df2.dataset == dataset) & (df2.datasize != 98)]
            out = sns.boxplot(data=temp_df, x="datasize", y=m, hue="timestep", ax=axs[i,j],  showfliers=False, saturation=1)
            axs[i,j].set_title(f"{dataset}", fontsize=font)
            y_min, _ = axs[i,j].get_ylim()
            axs[i,j].set_ylim(y_min, 1)
            sns.despine(ax=axs[i,j], offset=10, trim=True)
            h, l = axs[i,j].get_legend_handles_labels()
            axs[i,j].get_legend().remove()
            axs[i, 0].set_ylabel("Precison", fontsize=font)
            axs[i, 1].set_ylabel("Recall", fontsize=font)
            axs[i,j].set_xlabel("")

    for ax in axs.flatten():
        ax.tick_params(axis='both', labelsize=font)
    for ax in [axs[:, 0], axs[:, 1]]:
        sns.despine(ax=ax[0 ], offset=10, trim=True, bottom=True)
        sns.despine(ax=ax[1 ], offset=10, trim=True, bottom=True)
        sns.despine(ax=ax[2 ], offset=10, trim=True)
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[-1 ].set_xlabel("Training dataset size", fontsize=font)
    # Put a legend to the right of the current axis


    fig.legend(h, ['baseline', 'LEDM'] + ['step ' + _l for _l in l[2:]], title="", ncol=4, 
            loc='center left', bbox_to_anchor=(0.25, -0.03), fontsize=font)
    plt.tight_layout()
    #plt.savefig("results_per_timestep.png")
    plt.savefig("results_per_timestep_prec_recall.pdf", bbox_inches='tight')
    plt.show()

# %%
