# UrbanControlNet

## Introduction
The repository contains the code implementation of the paper: Envisioning Global Urban Development with Satellite Imagery and Generative AI.

<img src="git1.png" width="100%">


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Model evaluation](#model-evaluation)
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

The dataset is built from publicly available global sources:
- **Urban boundaries** — [GHS Urban Centre Database (2023)](https://human-settlement.emergency.copernicus.eu/ghs_ucdb_2024.php), covering 500 metropolitan areas with 400 m × 400 m grids.
- **Satellite imagery** — [Mapbox Static Tiles API](https://docs.mapbox.com/api/maps/static-tiles/).
- **Population and building data** — GHSL P2023A (2020): [GHS-BUILT-S](https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S), [GHS-BUILT-V](https://human-settlement.emergency.copernicus.eu/ghs_buV2023.php), [GHS-POP](https://human-settlement.emergency.copernicus.eu/ghs_pop2019.php).
- **Environmental constraints** — [OpenStreetMap](https://www.openstreetmap.org), including major roads, water bodies, and railways.


#### Dataset Download

Download land use, building, and basemap data:

```shell
python download_mapbox_tiles.py  
python download_osm_landuse_building.py  
```

Create DEM, hint, and satellite images:

```shell
python create_dem_image.py  
python create_hint_image.py  
python create_satellite_image.py  
```

Compute grid density and land use–road metrics:

```shell
python compute_grid_density_gee.py  
python create_landuse_road_metrics.py  
```

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
