<div align="center">


<!-- TITLE -->
# Biodifusion
<div align="left">

## Description
This repository serves as the primary codebase for implementing a Diffusion model, specifically designed for the generation of synthetic signals. The model is a pivotal component used in the research paper titled "BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis" ([accessible here](https://arxiv.org/abs/2401.10282)).

## Setup

Before running the code, ensure that you have the following prerequisites installed:

- Python 3.x
- PyTorch
- Nvidia CUDA toolkit and cuDNN (for GPU acceleration)

```bash
pip install torch torchvision
conda install cudatoolkit
```


### Conda Virtual Environment

Create the Conda virtual environment using the [environment file](environment.yml):
```bash
conda env create -f environment.yml

# dynamically set python path for the environment
conda activate ml
conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
```


<!-- USAGE -->
## Training
In order to use our model first you will need to retrain it: 
### Unconditional diffusion model for 3 channel data:
```python ddpm.py```
### Conditional diffusion model for 3 channel data:
```python ddpm_conditional.py```
### Classifier-free diffusion model for 1 channel data:
```cd signal/
python load_dataset.py #KaggleKey #KaggleName
python ddpm1d_cls_free.py
```
### Signal conditional diffusion model for 1 channel data:
```cd signal/
python load_dataset.py #KaggleKey #KaggleName
python ddpm1d_sign_cond.py
```
## Sampling

After [training](#training), you can sample from the trained model using the following steps:

### Unconditional Diffusion Model for 3 Channel Data:
```python
# Set device and load the pre-trained UNet model
device = "cuda:2"
model = UNet().to(device)
ckpt = torch.load("../../src/models/DDPM_Unconditional/ckpt.pt")
model.load_state_dict(ckpt)
# Create a Diffusion model instance and sample from it
diffusion = Diffusion(img_size=32, device=device)
x = diffusion.sample(model, 10)
```

### Conditional Diffusion Model for 3 Channel Data:
```python
# Set the number of samples and device
n = 10
device = "cuda:3"
# Create a Diffusion model instance and load the trained model checkpoint
diffusion = Diffusion(img_size=32, device=device)
diffusion.load("../../src/models/DDPM_conditional")
# Prepare labels and sample from the diffusion model
labels = torch.full((n,), 1).long().to(diffusion.device)
sampled_images = diffusion.sample(use_ema=False, labels=labels)
```

### Classifier-Free Diffusion Model for 1 Channel Data:
```python
# Set the number of samples, device, and create the Unet1D_cls_free model
n = 10
device = "cuda:3"
model = Unet1D_cls_free(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_classes=5,
    cond_drop_prob=0.5,
    channels=1
)
# Load the pre-trained model checkpoint
ckpt = torch.load("../../src/signal/checkpoint/DDPM1D_cls_free_MITBIH/checkpoint.pt")
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
# Create the GaussianDiffusion1D_cls_free model and sample from it
diffusion = GaussianDiffusion1D_cls_free(
    model,
    seq_length=128,
    timesteps=1000
).to(device)
y = torch.Tensor([0] * n).long().to(device)
x = diffusion.sample(classes=y, cond_scale=3.)
```

### Self-Conditional Diffusion Model for 1 Channel Data:
```python
# Set the device and create the Unet1D model with self-conditioning
device = "cuda:3"
model = Unet1D(
    dim=64,
    self_condition=True,
    dim_mults=(1, 2, 4, 8),
    channels=1
)
# Load the pre-trained model checkpoint
ckpt = torch.load("../../src/signal/checkpoint/DDPM1D_Selfconditional_maskedCond/checkpoint.pt")
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
# Create the GaussianDiffusion1D model and sample from it
# seq_length must be able to be divided by dim_mults
diffusion = GaussianDiffusion1D(
    model,
    seq_length=128,
    timesteps=1000,
    objective='pred_v'
).to(device)
```

#### Make sure to adjust the file paths and model names as needed.


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


<!-- CITATION -->
## Citation
Feel free to cite our paper using this .bibtex or .cff formats in this repository. 



```bibtex
@misc{li2024biodiffusion,
      title={BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis}, 
      author={Xiaomin Li and Mykhailo Sakevych and Gentry Atkinson and Vangelis Metsis},
      year={2024},
      eprint={2401.10282},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```

