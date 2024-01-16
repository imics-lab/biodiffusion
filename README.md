> `### <<< DELETE ME:` ***Template***
>  
> This template contains some useful structure and convention for new research
> projects that will help you get started more quickly, and will make your
> code more accessible, maintainable, and reproducible. This will make your
> work more likely to be adopted by others!
>
> I highly recommend taking a second to read Patrick Mineault's
> [Good Research Code Handbook](https://goodresearch.dev/index.html) if you are
> in the process of starting a new project for many tips that will help you
> beyond the initial setup phase.
>
> > *Note:* You should delete everything within markdown blockquotes `>` before
> going live with your project.
> 
> `### DELETE ME >>>`


<div align="center">


<!-- TITLE -->
# Project Template `> REPLACE ME`
A new project template for research projects. `> REPLACE ME`

<!-- BADGES -->
> <div align="left">
> 
> `### <<< DELETE ME` ***Badges*** *(optional)*
>  
> If you have an arXiv paper, you can add and update the `[arXiv]` badge by
> replacing `1234.56789` with the arXiv ID of your paper and the arXiv
> subject `cs.LG` with the main subject. Else, delete it (or comment out).
>
> if your paper is published at a conference, you can add and update the
> `[Conference]` badge by replacing `Conference`, `Year`, and the link fields.
> Else, delete it (or comment out).
> 
> It is also useful to add a CI build/test [status badge](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/adding-a-workflow-status-badge)
> to your project. A base Continuous Integration pipeline has been defined in
> [.github/workflows/conda-test.yml](.github/workflows/conda-test.yml)
> GitHub will automatically register and run this pipeline when you push to the
> `main` branch.
> 
> Copy the workflow badge from the corresponding workflow in the Actions tab
> (click the breadcrumbs) and overwrite the Conda Test badge below.
> 
> 
> `### DELETE ME >>>`
>
> </div>

[![Conda Test](https://github.com/ellisbrown/research-project/actions/workflows/conda-test.yml/badge.svg)](https://github.com/ellisbrown/research-project/actions/workflows/conda-test.yml)
[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)
[![Conference](https://img.shields.io/badge/Conference-year-4b44ce.svg)](https://yourconference.org/2020)

</div>


<!-- DESCRIPTION -->
## Description
> `### <<< DELETE ME:` ***Description***
>  
> Fill in a succinct description of your project.
> 
> `### DELETE ME >>>`


<!-- SETUP -->
## Setup

> `### <<< DELETE ME:` ***Setup***
>  
> Below are some base instructions for setting up a conda environment. See this
> [guide](https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/)
> to Conda to learn some great best practices!
> 
> Add any instructions necessary to setup the project. The best time to create
> this is ***as you are developing*** the project, while you remember the steps
> you have taken.
>
> *Brownie points*: try to follow your setup instructions to replicate the setup
> from scratch on another machine to ensure that it is truly reproducible.
> 
> `### DELETE ME >>>`


### Conda Virtual Environment

Create the Conda virtual environment using the [environment file](environment.yml):
```bash
conda env create -f environment.yml

# dynamically set python path for the environment
conda activate YOUR_PROJECT
conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
```


<!-- USAGE -->
## Usage
> `### <<< DELETE ME:` ***Usage***
>  
> Provide information on how to run your project. Delete the below example.
> 
> `### DELETE ME >>>`

```python
from foo import bar

bar.baz("hello world")
```

```bash
python -m foo.bar "hello world"
```


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


<!-- CITATION -->
## Citation

> `### <<< DELETE ME:` ***Citation***
>  
> Adding a citation to your README will make it easier for others to cite your
> work. Add your bibtext citation to the README below. GitHub also will
>  automatically detect [Citation File Format (`.cff`) files](https://citation-file-format.github.io/),
> rendering a "Cite this repository" button. See [GitHub's tutorial](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
> for more information. The example from this tutorial is included in 
> [CITATION.cff](CITATION.cff), and should be modified or deleted.
> 
> `### DELETE ME >>>`


```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  year={Year}
}
```

