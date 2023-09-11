import numpy as np
import gradio as gr
from PIL import Image
import torch
from torch import nn
from einops.layers.torch import Rearrange
from torchvision import transforms
from models.unet_model import Unet
from models.datasetDM_model import DatasetDM
from skimage import measure, segmentation
import cv2
from tqdm import tqdm
from einops import repeat

img_size = 128
font = cv2.FONT_HERSHEY_SIMPLEX


## %%
def load_img(img_file):
        # assert type of input
    if isinstance(img_file, np.ndarray):
        img = torch.Tensor(img_file).float()
        # make sure img is between 0 and 1
        if img.max() > 1:
            img /= 255
        # resize 
        img = transforms.Resize(img_size)(img)
    elif isinstance(img_file, str):
        img = Image.open(img_file).convert('L').resize((img_size, img_size))
        img = transforms.ToTensor()(img).float()
    elif isinstance(img_file, Image.Image):
        img = img_file.convert('L').resize((img_size, img_size))
        img = transforms.ToTensor()(img).float()
    else:
        raise TypeError("Input must be a numpy array, PIL image, or filepath")
    if len(img.shape) == 2:
        img = img[None, None]
    elif len(img.shape) == 3:
        img = img[None]
    else:
        raise ValueError("Input must be a 2D or 3D array")
    return img

def predict_baseline(img, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["config"]
    baseline = Unet(**vars(config))
    baseline.load_state_dict(checkpoint["model_state_dict"])
    baseline.eval()
    return (torch.sigmoid(baseline(img)) > .5).float().squeeze().numpy()

def predict_LEDM(img, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["config"]
    config.verbose = False
    LEDM = DatasetDM(config)
    LEDM.load_state_dict(checkpoint["model_state_dict"])
    LEDM.eval()
    return (torch.sigmoid(LEDM(img)) > .5).float().squeeze().numpy()

def predict_TEDM(img, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["config"]
    config.verbose = False
    TEDM = DatasetDM(config)
    TEDM.classifier = nn.Sequential(
        Rearrange('b (step act) h w -> (b step) act h w', step=len(TEDM.steps)),
        nn.Conv2d(960, 128, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 32, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 1, config.out_channels)
        )
    TEDM.load_state_dict(checkpoint["model_state_dict"])
    TEDM.eval()
    return (torch.sigmoid(TEDM(img)).mean(0) > .5).float().squeeze().numpy()

predictors = {'Baseline': predict_baseline, 
              'Global CL': predict_baseline,
              'Global & Local CL': predict_baseline,
              'LEDM': predict_LEDM, 
              'LEDMe': predict_LEDM,
              'TEDM': predict_TEDM}
model_folders = {
    'Baseline': 'baseline',
    'Global CL': 'global_finetune',
    'Global & Local CL': 'glob_loc_finetune',
    'LEDM': 'LEDM',
    'LEDMe': 'LEDMe',
    'TEDM': 'TEDM'
}


def postprocess(pred, img):
    all_labels = measure.label(pred, background=0)
    _, cn = np.unique(all_labels, return_counts=True)
    # find the two largest connected components that are not the background
    if len(cn) >= 3:
        lungs = np.argsort(cn[1:])[-2:] + 1
        all_labels[(all_labels!=lungs[0]) & (all_labels!=lungs[1])] = 0
        all_labels[(all_labels==lungs[0]) | (all_labels==lungs[1])] = 1
    # put all_labels into a cv2 object
    if len(cn) > 1:
        img = segmentation.mark_boundaries(img, all_labels, color=(1,0,0), mode='outer', background_label=0)
    else:
        img = repeat(img, 'h w -> h w c', c=3)
    return img



def predict(img_file, models:list, training_sizes:list, seg_img=False, progress=gr.Progress()):
    max_progress = len(models) * len(training_sizes)
    n_progress = 0
    progress((n_progress, max_progress), desc="Starting")
    img = load_img(img_file)
    print(img.shape)
    preds = []
    # sorting models so that they show as  baseline - LEDM - LEDMe - TEDM
    models = sorted(models, key=lambda x: 0 if x == 'Baseline' else 1 if x == 'Global CL' else 2 if x == 'Global & Local CL' else 3 if x == 'LEDM' else 4 if x == 'LEDMe' else 5)
    
    for model in models:
        print(model)
        model_preds = []
        for training_size in sorted(training_sizes):
            #if n_progress < max_progress:
            progress((n_progress, max_progress) , desc=f"Predicting {model} {training_size}")
            n_progress += 1
            print(training_size)
            out = predictors[model](img, f"logs/{model_folders[model]}/{training_size}/best_model.pt")
            writing_colour = (.5,.5,.5)
            if seg_img:
                out = postprocess(out, img.squeeze().numpy())
                writing_colour = (1,1,1)
            out = cv2.putText(np.array(out),f"{model} {training_size}",(5,125), font, .5, writing_colour,1, cv2.LINE_AA)
            #ImageDraw.Draw(out).text((0,128), f"{model} {training_size}", fill=(255,0,0))
            model_preds.append(np.asarray(out))
        preds.append(np.concatenate(model_preds, axis=1))
    prediction = np.concatenate(preds, axis=0)
    if (prediction.shape[1] <=128*2):
        pad = (330 - prediction.shape[1])//2
        if len(prediction.shape) == 2:
            prediction = np.pad(prediction, ((0,0), (pad, pad)), 'constant', constant_values=1)
        else:
            prediction = np.pad(prediction, ((0,0), (pad, pad), (0,0)), 'constant', constant_values=1)
    return prediction


## %%
input = gr.Image( label="Chest X-ray", shape=(img_size, img_size), type="pil")
output = gr.Image(label="Segmentation", shape=(img_size, img_size))
## %%
demo = gr.Interface(
    fn=predict,
    inputs=[input, 
            gr.CheckboxGroup(["Baseline", "Global CL", "Global & Local CL", "LEDM", "LEDMe", "TEDM"], label="Model", value=["Baseline", "LEDM", "LEDMe", "TEDM"]),
            gr.CheckboxGroup([1,3,6,12,197], label="Training size", value=[1,3,6,12,197]),
            gr.Checkbox(label="Show masked image (otherwise show binary segmentation)", value=True),],

    outputs=output,
    examples = [
    ['img_examples/NIH_0006.png'], 
    ['img_examples/NIH_0076.png'], 
    ["img_examples/00016568_041.png"], 
    ['img_examples/NIH_0024.png'], 
    ['img_examples/00015548_000.png'], 
    ['img_examples/NIH_0019.png'], 
    ['img_examples/NIH_0094.png'],
    ['img_examples/NIH_0051.png'], 
    ['img_examples/NIH_0012.png'], 
    ['img_examples/NIH_0014.png'], 
    ['img_examples/NIH_0055.png'], 
    ['img_examples/NIH_0035.png'], 
                ],
    title="Chest X-ray Segmentation with TEDM.",
    description="""<img src="file/img_examples/TEDM-model-visualisation.png"
     alt="Markdown Monster icon"
     style="margin-right: 10px;" />"""+
    "\nMedical image segmentation is a challenging task, made more difficult by many datasets' limited size and annotations. Denoising diffusion probabilistic models (DDPM) have recently shown promise in modelling " + 
    "the distribution of natural images and were successfully applied to various medical imaging tasks. This work focuses on semi-supervised image segmentation using diffusion models, particularly addressing domain " + 
    "generalisation. Firstly, we demonstrate that smaller diffusion steps generate latent representations that are more robust for downstream tasks than larger steps. Secondly, we use this insight to propose an improved " + 
    "esembling scheme that leverages information-dense small steps and the regularising effect of larger steps to generate predictions. Our model shows significantly better performance in domain-shifted settings while " +  
    "retaining competitive performance in-domain. Overall, this work highlights the potential of DDPMs for semi-supervised medical image segmentation and provides insights into optimising their performance under domain shift."+ 
    "\n\n\n When choosing 'Show masked image', we post-process the segmentation by choosing up to two largest connected components and drawing their outline. "+
    "\nNote that each model takes 10-35 seconds to run on CPU. Choosing all models and all training sizes will take some time. "+
    "We noticed that gradio sometimes fails on the first try. If it doesn't work, try again.",
    cache_examples=False,
)
demo.queue().launch(debug=True)
#demo.queue().launch(share=True)
