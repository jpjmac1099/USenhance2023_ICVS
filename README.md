
# Sparse edge- and quality-aware CycleGAN for ultrasound image enhancement

This github presents the code used by the ICVS team for the USenhance2023 MICCAI Challenge.

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/jpjmac1099/USenhance2023_ICVS
cd USenhance2023_ICVS
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### Model train/test
- Train a model:
```
python train.py --dataroot ./datasets/Challenge --name train_1 --model cycle_gan_perceptual_edge_sparse --preprocess scale_width_and_crop --dataset_mode aligned
```
To see more intermediate results, check out `./checkpoints/train_1/web/index.html`.
- Test the model:
```
python test.py --dataroot ./datasets/Challenge --name train_1 --model cycle_gan_perceptual_edge_sparse --dataset_mode aligned
```
- The test results will be saved to a html file here: `./results/train_1/latest_test/index.html`.
