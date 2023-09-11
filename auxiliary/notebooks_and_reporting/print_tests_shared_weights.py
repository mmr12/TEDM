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

    # %% load TEDM data
    metrics3 = {"dice": [], "precision": [], "recall": [], "exp": [], "datasize": [], "dataset":[], }
    exp = "TEDM"
    for datasize in [1, 3, 6, 12, 24, 49, 98, 197]:
                    if len(set(files_needed) - set(os.listdir(head / exp / str(datasize) ))) == 0:
                        print(f"Experiment {datasize}")
                        output = torch.load(head / exp / str(datasize)/ "JSRT_val_predictions.pt")
                        print(f"{output['dice'].mean()}\t{output['dice'].std()}")
                        for file in files_needed[1:]:
                            output = torch.load(head / exp / str(datasize) /  file)
                        
                            metrics_datasize = datasize if datasize is not None else 197
                            metrics3["dice"].append(output["dice"].numpy())
                            metrics3["precision"].append(output["precision"].numpy())
                            metrics3["recall"].append(output["recall"].numpy())
                            metrics3["exp"].append(np.array(['TEDM'] * len(output["dice"])))
                            metrics3["datasize"].append(np.array([metrics_datasize] * len(output["dice"])))
                            metrics3["dataset"].append(np.array([file.split("_")[0]]*len(output["dice"])))
                            
                    else:
                            print(f"Experiment {datasize} is missing files")

    for key in metrics3:
        metrics3[key] = np.concatenate(metrics3[key]).squeeze()
        print(key, metrics3[key].shape)
    df3 = pd.DataFrame(metrics3)
    # %% Boxplot of TEDM vs LEDM and baseline
    df4 = pd.concat([df, df3])
    df4.datasize = df4.datasize.astype(int)
    m='dice'
    dataset="JSRT"
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    for j, m in enumerate(["dice", "precision", "recall"]):
        #axs[0,j].set_ylim(0.8, 1)
        #axs[0,j].set_ylim(0.6, 1)
        #axs[0,j].set_ylim(0.7, 1)
        for i, dataset in enumerate(["JSRT", "NIH", "Montgomery"]):
            temp_df = df4[(df4.dataset == dataset)]
            #sns.lineplot(data=df[df.dataset == dataset], x="datasize", y=m, hue="exp", ax=axs[i,j])
            sns.boxplot(data=temp_df, x="datasize", y=m,  ax=axs[i,j], hue="exp", showfliers=False, saturation=1,
                        hue_order=['baseline', 'LEDM', 'TEDM'])
            axs[i,j].set_title(f"{dataset} {m}")
            axs[i,j].set_xlabel("Training dataset size")
            h, l = axs[i,j].get_legend_handles_labels()
            axs[i,j].legend(h, ['Baseline', 'LEDM', 'TEDM (ours)'], title="", loc='lower right')
    plt.tight_layout()
    plt.savefig("results_shared_weights.pdf")
    plt.show()
    # %% Load LEDMe and Step 1
    metrics2 = {"dice": [], "precision": [], "recall": [], "exp": [], "datasize": [], "dataset":[], }
    for exp in ["LEDMe", 'Step_1']:
        for datasize in [1, 3, 6, 12, 24, 49, 98, 197]:
            if len(set(files_needed) - set(os.listdir(head / exp / str(datasize) ))) == 0:
                print(f"Experiment {exp} {datasize}")
                output = torch.load(head / exp / str(datasize)/ "JSRT_val_predictions.pt")
                print(f"{output['dice'].mean()}\t{output['dice'].std()}")
                for file in files_needed[1:]:
                    output = torch.load(head / exp / str(datasize) / file)
                    #print(f"{output['dice'].mean()*100:.3}\t{output['dice'].std()*100:.3}\t{output['precision'].mean()*100:.3}\t{output['precision'].std()*100:.3}\t{output['recall'].mean()*100:.3}\t{output['recall'].std()*100:.3}",
                    #    end="\n\n\n\n")
                    metrics_datasize = 197 if datasize == "None" else datasize
                    metrics2["dice"].append(output["dice"].numpy())
                    metrics2["precision"].append(output["precision"].numpy())
                    metrics2["recall"].append(output["recall"].numpy())
                    metrics2["exp"].append(np.array([exp] * len(output["dice"])))
                    metrics2["datasize"].append(np.array([int(metrics_datasize)] * len(output["dice"])))
                    metrics2["dataset"].append(np.array([file.split("_")[0]]*len(output["dice"])))
            else:
                    print(f"Experiment {exp} is missing files")

    for key in metrics2:
        metrics2[key] = np.concatenate(metrics2[key]).squeeze()
        print(key, metrics2[key].shape)
    df2 = pd.DataFrame(metrics2)
    # %% Boxplot of TEDM vs LEDM and baseline, Step 1 and LEDMe
    df4 = pd.concat([df, df3, df2])
    df4.datasize = df4.datasize.astype(int)


    m='dice'
    dataset="JSRT"
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    for j, m in enumerate(["dice", "precision", "recall"]):

        for i, dataset in enumerate(["JSRT", "NIH", "Montgomery"]):
            temp_df = df4[(df4.dataset == dataset)]
            #sns.lineplot(data=df[df.dataset == dataset], x="datasize", y=m, hue="exp", ax=axs[i,j])
            sns.boxplot(data=temp_df, x="datasize", y=m,  ax=axs[i,j], hue="exp", showfliers=False, saturation=1,
                        hue_order=['baseline', 'LEDM', 'Step_1', 'LEDMe', 'TEDM', ])
            axs[i,j].set_title(f"{dataset} {m}")
            axs[i,j].set_xlabel("Training dataset size")
            h, l = axs[i,j].get_legend_handles_labels()
            axs[i,j].legend(h, ['Baseline', 'LEDM',   'Step 1', 'LEDMe', 'TEDM'], title="", loc='lower right')
    plt.tight_layout()
    plt.savefig("results_shared_weights.pdf")
    plt.show()
    # %% Load TEDM ablation studies
    metrics4 = {"dice": [], "precision": [], "recall": [], "exp": [], "datasize": [], "dataset":[], }
    exp = "TEDM"
    for datasize in [1, 3, 6, 12, 24, 49, 98, 197]:
                if len(set(files_needed) - set(os.listdir(head / exp / str(datasize)))) == 0:
                    print(f"Experiment {datasize} ")
                    for step in [1,10,25]:
                        for file in files_needed[1:]:
                            output = torch.load(head / exp / str(datasize) /  file.replace("predictions", f"timestep{step}_predictions"))
                            #print(f"{output['dice'].mean()*100:.3}\t{output['dice'].std()*100:.3}\t{output['precision'].mean()*100:.3}\t{output['precision'].std()*100:.3}\t{output['recall'].mean()*100:.3}\t{output['recall'].std()*100:.3}",
                            #    end="\n\n\n\n")
                            metrics_datasize = datasize if datasize is not None else 197
                            metrics4["dice"].append(output["dice"].numpy())
                            metrics4["precision"].append(output["precision"].numpy())
                            metrics4["recall"].append(output["recall"].numpy())
                            metrics4["exp"].append(np.array([f'Step {step} (MLP)'] * len(output["dice"])))
                            metrics4["datasize"].append(np.array([metrics_datasize] * len(output["dice"])))
                            metrics4["dataset"].append(np.array([file.split("_")[0]]*len(output["dice"])))
                        #metrics3["timestep"].append(np.array(timestep * len(output["dice"])))
                else:
                        print(f"Experiment {datasize} is missing files")

    for key in metrics3:
        metrics4[key] = np.concatenate(metrics4[key]).squeeze()
        print(key, metrics4[key].shape)
    df4 = pd.DataFrame(metrics4)
    # %% Print inputs to paper table
    df_all = pd.concat([df, df3, df2, df4])
    df_all.datasize = df_all.datasize.astype(int)
    for i, dataset in enumerate(["JSRT", "NIH", "Montgomery"]):
        temp_df = df_all.loc[(df_all.dataset == dataset) & (df_all.datasize.isin([1, 3, 6, 12, 197])), ["exp", "datasize", "dice"]]
        print(dataset)
        mean = temp_df.groupby(["exp", "datasize"]).mean().unstack() * 100
        std = temp_df.groupby(["exp", "datasize"]).std().unstack() * 100
        for exp, exp_name in zip(['baseline', 'LEDM','Step_1', 'Step 1 (MLP)',
                                'Step 10 (MLP)','Step 25 (MLP)', 'LEDMe', 'TEDM'],
        ['Baseline', 'DatasetDDPM', 'Step 1 (linear)','Step 1 (MLP)', 'Step 10 (MLP)','Step 25 (MLP)','DatasetDDPMe', 'Ours', ]):
        
            print(exp_name, end='&\t')
            print(f"{round(mean.loc[exp, ('dice', 1)],2):.3} $\pm$ {round(std.loc[exp, ('dice', 1)],1)}", end='&\t')
            print(f"{round(mean.loc[exp, ('dice', 3)], 2):.3} $\pm$ {round(std.loc[exp, ('dice', 3)],1)}", end='&\t')
            print(f"{round(mean.loc[exp, ('dice', 6)], 2):.3} $\pm$ {round(std.loc[exp, ('dice', 6)],1)}", end='&\t')
            print(f"{round(mean.loc[exp, ('dice', 12)], 2):.3} $\pm$ {round(std.loc[exp, ('dice', 12)],1)}", end='&\t')
            print(f"{round(mean.loc[exp, ('dice', 197)], 2):.3} $\pm$ {round(std.loc[exp, ('dice', 197)],1)}", end="""\\\\""")
            
            print()
        
    # %% Print inputs to paper appendix table
    for i, dataset in enumerate(["JSRT", "NIH", "Montgomery"]):
        print("\n" + dataset)
        for m in ["precision", "recall"]:
            temp_df = df_all.loc[(df_all.dataset == dataset) & (df_all.datasize.isin([1, 3, 6, 12, 24, 49, 98, 197])), ["exp", "datasize", m]]
            print("\n"+m)
            mean = temp_df.groupby(["exp", "datasize"]).mean().unstack() * 100
            std = temp_df.groupby(["exp", "datasize"]).std().unstack() * 100
            for exp, exp_name in zip(['baseline', 'LEDM','Step_1', 'LEDMe', 'TEDM'],
            ['Baseline', 'LEDM', 'Step 1 (linear)','LEDMe', 'TEDM (ours)',]):
            
                print(exp_name, end='&\t')
                print(f"{round(mean.loc[exp, (m, 1)],2):.3} $\pm$ {round(std.loc[exp, (m, 1)],1)}", end='&\t')
                print(f"{round(mean.loc[exp, (m, 3)],2):.3} $\pm$ {round(std.loc[exp, (m, 3)],1)}", end='&\t')
                print(f"{round(mean.loc[exp, (m, 6)],2):.3} $\pm$ {round(std.loc[exp, (m, 6)],1)}", end='&\t')
                print(f"{round(mean.loc[exp, (m, 12)],2):.3} $\pm$ {round(std.loc[exp, (m, 12)],1)}", end='&\t')
                print(f"{round(mean.loc[exp, (m, 197)],2):.3} $\pm$ {round(std.loc[exp, (m, 197)],1)}", end='\\\\')
                
                
                print()  

    # %% Wilcoxon tests - to use interactively
    from scipy.stats import wilcoxon
    m ="precision"
    m='recall'
    dataset ="Montgomery"
    dssize =12

    exp = "baseline"
    exp = 'Step_1'
    exp = "LEDM"
    exp="TEDM"
    exp_2= 'LEDMe'

    x = df_all.loc[(df_all.dataset == dataset) & (df_all.exp == exp_2) & (df_all.datasize == dssize), m].to_numpy()
    y = df_all.loc[(df_all.dataset == dataset) & (df_all.exp == exp)& (df_all.datasize == dssize), m].to_numpy()
    print(f"{m} - {dataset} - {dssize} - {exp_2}: {x.mean():.4}+/-{x.std():.3} ")
    print(f"{m} - {dataset} - {dssize} - {exp}: {y.mean():.4}+/-{y.std():.3} ")
    print(f"{m} - {dataset} - {dssize}: {wilcoxon(x, y=y, zero_method='wilcox', correction=False, alternative='two-sided',).pvalue:.3} obs given equal   ")
    print(f"{m} - {dataset} - {dssize}: {wilcoxon(x, y=y, zero_method='wilcox', correction=False, alternative='greater',).pvalue:.3} obs given {exp_2} < {exp} ")
    print(f"{m} - {dataset} - {dssize}: {wilcoxon(x, y=y, zero_method='wilcox', correction=False, alternative='less',).pvalue:.3} obs given {exp_2} > {exp} ")
