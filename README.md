# fastMRI deep learning

The field of MRI reconstruction has previously utilized deep learning techniques to produce high-quality MRI images from raw MRI data, reducing the amount of time to produce an MRI image. These reconstruction techniques optimize for the entire image at once, however, for medical professionals, the MRI reconstruction must have a high quality in diagnostically relevant regions. Therefore, our work introduces methods for improving region-specific MRI reconstruction for diagnostic quality. These changes ensure that the diagnostically relevant regions of interest have greater importance and quality during MRI reconstruction.

# Experiment results:
Available at the following link: https://wandb.ai/ebruda01-georgia-institute-of-technology/deep_learning_fastmri_project/table?nw=nwuserebruda01
### Based on

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md)
[![Build and Test](https://github.com/facebookresearch/fastMRI/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/facebookresearch/fastMRI/actions/workflows/build-and-test.yml)

[Website](https://fastMRI.org) |
[Dataset](https://fastmri.med.nyu.edu/) |
[GitHub](https://github.com/facebookresearch/fastMRI) |
[Publications](#list-of-papers)

[fastMRI](https://fastMRI.org) is a collaborative research project from
Facebook AI Research (FAIR) and NYU Langone Health to investigate the use of AI
to make MRI scans faster. NYU Langone Health has released fully anonymized knee
and brain MRI datasets that can be downloaded from
[the fastMRI dataset page](https://fastmri.med.nyu.edu/). Publications
associated with the fastMRI project can be found
[at the end of this README](#list-of-papers).

### The fastMRI Dataset

There are multiple publications describing different subcomponents of the data
(e.g., brain vs. knee) and associated baselines. All of the fastMRI data can be
downloaded from the [fastMRI dataset page](https://fastmri.med.nyu.edu/).

* **Meta Project Summary, Datasets, Baselines:** [fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)

* **Knee Data:** [fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning ({F. Knoll*, J. Zbontar*} et al., 2020)](https://doi.org/10.1148/ryai.2020190007)

* **Brain Dataset Properties:** [Supplemental Material](https://ieeexplore.ieee.org/ielx7/42/9526230/9420272/supp1-3075856.pdf?arnumber=9420272) of [Results of the 2020 fastMRI Challenge for Machine Learning MR Image Reconstruction ({M. Muckley*, B. Riemenschneider*} et al., 2021)](https://doi.org/10.1109/TMI.2021.3075856)

* **Prostate Data:** [FastMRI Prostate: A Publicly Available, Biparametric MRI Dataset to Advance Machine Learning for Prostate Cancer Imaging (Tibrewala et al., 2023)](https://arxiv.org/abs/2304.09254)

## Running the project
### Pace Access
- Connect to the Georgia Tech VPN using GlobalProtect or https://vpn.gatech.edu/ 
- Go to https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard/
- Click on Interactive Apps
- Click on Visual Studio Code
- Request a GPU (Ideally an NVIDIA H100 or H200 but any NVIDIA GPU works as well)
- Click request and then wait to connect

### Conda setup

- If you haven’t set up conda before for your PACE account follow the steps below. If not, skip to step #2
  - Go to your home directory /hice1/userId
  - Follow the steps here to download miniconda3: https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation 
  - Check that the installation worked by typing conda in the terminal
- Link to your conda environment to your scratch folder. This is important since the packages take up so much space
  - Tutorial here: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0041621 
  - Note: for the p-<pi-username>-<number> mentioned, use your gt username like ebruda3
- Setup the environment for this project
  - Create the conda environment by running conda env create -f dl_environment.yml
  - Activate the environment
  - If you haven’t done this yet, run pip install -e .
    - This installs the fastmri project


### File Setup
- Clone the following Github repo and put it in your scratch folder (ex: ebruda3/scratch) https://github.com/Deep-Learning-Project-FastMRI/fastMRI_deep_learning 
  - Make sure you clone so you can do a git push later
- Download the dataset zip files by running the curl commands from the NYU email
- Extract each of the .tar files by running these commands:
  - tar -xf knee_singlecoil_[training_mode].tar.xz
- Put all of the files in a data folder. Ex: ebruda3/scratch/fastmri_deep_learning/data/

### Training the model
- cd into fastmri_deep_learning/fastmri_examples/unet/
- Activate the dl_proj_2 conda environment 
- Change the knee_path in the fastmri_dirs.yaml file to be wherever your data is saved
  - Ex: knee_path: "/home/hice1/ebruda3/scratch/fastMRI_deep_learning/data/"
- Start training the model by running python train_unet_demo.py

### For benchmark Train, Test, Val
- python train_unet_demo.py --experiment_mode=benchmark --mode=train
- python train_unet_demo.py --experiment_mode=benchmark --mode=test
- python train_unet_demo.py --experiment_mode=benchmark --mode=val

### For manual Train, Test, Val
- python train_unet_demo.py --experiment_mode=manual --mode=train
- python train_unet_demo.py --experiment_mode=manual --mode=test
- python train_unet_demo.py --experiment_mode=manual --mode=val

### For heatmap Train, Test, Val
- python train_unet_demo.py --experiment_mode=heatmap --mode=train
- python train_unet_demo.py --experiment_mode=heatmap --mode=test
- python train_unet_demo.py --experiment_mode=heatmap --mode=val

### For Attention train, Test, Val
- python train_unet_demo.py --experiment_mode=attention --mode=train
- python train_unet_demo.py --experiment_mode=attention --mode=test
- python train_unet_demo.py --experiment_mode=attention --mode=val

### Running any command in the background
- nohup python -u train_unet_demo.py --mode "MODE" --experiment_mode "EXPERIMENT" > "LOG_FILE_NAME".log 2>&1 &



## License

fastMRI is MIT licensed, as found in the [LICENSE file](https://github.com/facebookresearch/fastMRI/tree/master/LICENSE.md).

## Cite

If you use the fastMRI data or code in your project, please cite the arXiv
paper:

```BibTeX
@misc{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```

If you use the fastMRI prostate data or code in your project, please cite that
paper:

```BibTeX
@misc{tibrewala2023fastmri,
  title={{FastMRI Prostate}: A Publicly Available, Biparametric {MRI} Dataset to Advance Machine Learning for Prostate Cancer Imaging},
  author={Tibrewala, Radhika and Dutt, Tarun and Tong, Angela and Ginocchio, Luke and Keerthivasan, Mahesh B and Baete, Steven H and Chopra, Sumit and Lui, Yvonne W and Sodickson, Daniel K and Chandarana, Hersh and Johnson, Patricia M},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint={2304.09254},
  year={2023}
}
```

