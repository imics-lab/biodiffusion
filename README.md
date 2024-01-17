<div align="center">


<!-- TITLE -->
# Project Template Biodifusion
A new project template for research projects. 
<div align="left">
<!-- DESCRIPTION -->
## Description
This repository serves as the primary codebase for implementing a Diffusion model, specifically designed for the generation of synthetic signals. The model is a pivotal component used in the research paper titled "BioDiffusion: A Versatile Diffusion Model for Biomedical Signal Synthesis" ([accessible here](change.link)).
<!-- SETUP -->
## Setup

Before running the code, ensure that you have the following prerequisites installed:

- Python 3.x
- PyTorch
- Nvidia CUDA toolkit and cuDNN (for GPU acceleration)

```bash
pip install torch torchvision
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
## Usage
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


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


<!-- CITATION -->
## Citation
Feel free to cite our paper using this .bibtex or .cff formats in this repository. 



```bibtex
@article{Xiomin Li, Mykhailo Sakevych,
  title={BioDiffusion: A Versatile Diffusion Model for Biomedical Signal
Synthesis},
  author={IMICS},
  year={2024}
}
```

