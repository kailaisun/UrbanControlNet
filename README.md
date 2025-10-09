# UrbanControlNet


The repository contains the code implementation of the paper: Envisioning Global Urban Development with Satellite Imagery and Generative AI.

<img src="git1.png" width="100%">


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Image Prediction](#image-prediction)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)
- [Contact Us](#contact-us)

## Installation

Download or clone the repository.

```shell
git clone https://github.com/kailaisun/UrbanControlNet.git
cd UrbanControlNet
```

We recommend using Conda ([Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)) for installation. 

### Environment Installation 

Create a virtual environment named `mambacontrol` and activate it.

```shell
conda env create -f environment.yaml
conda activate mambacontrol
```


## Dataset Preparation

#### Dataset Download

Image and label download address: . It includes:


## Model Training

#### Finetuning

```shell
cd Urbancontrolnet
python train_density.py 
```

#### Finetuning decoder

Set sd_locked = False, then

```shell
python train_density.py
```

## Model Testing

#### Checkpoints:
-Finetuning:[Download](https://1drv.ms/u/s!AnkbiBgsbBltncF0u-3e5rkH2yOTkg?e=WQ1wle)

#### Urban Satelliate Image Generation:

```shell
python results_view_loop_density.py
```


## Metric Prediction Model 
#### Training 

```shell
python prediction_train.py
```


## Model evaluation

For computing metric (e.g., FID, ID, SSIM, FSIM, PSNR, etc.), please see our another repo: [Evaluation-Metrics](https://github.com/T5-AI/Evaluation-Metrics)

## Acknowledgements


## Citation



## License

The repository is licensed under the [Apache 2.0 license](LICENSE).

## Contact Us

If you have other questions❓, please contact us in time 👬
