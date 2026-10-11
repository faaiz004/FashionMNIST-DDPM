# FashionMNIST DDPM

This repo provides a DDPM implementation that learns to generate 28×28 grayscale FashionMNIST images. A convolutional U-Net is trained to predict the noise added to an image at a randomly selected diffusion timestep. At generation time, the model starts from random noise and applies the learned reverse process to produce images.

## Model

- **Data:** FashionMNIST training images, normalized to `[-1, 1]`.
- **Noise schedule:** 1,000 linearly spaced variance values, from `1e-4` to
  `0.02`.
- **Denoiser:** U-Net with residual convolution blocks, GroupNorm, SiLU
  activations, sinusoidal timestep embeddings, and skip connections.
- **Objective:** Mean squared error between the sampled noise and the U-Net's
  predicted noise.
- **Sampling:** Ancestral reverse diffusion, with optional intermediate states
  saved to visualize the denoising trajectory.

## Setup

Use Python 3.10 or newer. Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch automatically selects CUDA when available. Pass `--device cpu` or
`--device cuda` to choose a device explicitly.

## Train

Run training with the default settings (50 epochs and batch size 128):

```bash
python run.py train
```

The FashionMNIST training set is downloaded to `./data` when needed. Training
saves model weights to `./checkpoints/fashion_mnist_ddpm.pt` and loss and
gradient-norm plots to `./outputs/training_curves.png`.

Training options can be adjusted from the command line:

```bash
python run.py train --epochs 20 --batch-size 64 --device cpu
```

Use `python run.py train --help` to see all training options, including the
dataset, output, checkpoint, and data-loader worker paths.

## Generate images

After training, generate 64 images and a visualization of selected reverse
diffusion steps:

```bash
python run.py sample
```

To choose the number of images, output directory, or device:

```bash
python run.py sample --num-samples 32 --output-dir ./generated --device cpu
```

The output directory contains `generated_samples.png` and
`denoising_trajectory.png`.

## Project structure

| File | Purpose |
| --- | --- |
| `run.py` | Command-line entry point for training and image generation. |
| `dataset.py` | FashionMNIST transformation and data-loader setup. |
| `model.py` | Timestep embedding and conditional U-Net architecture. |
| `reverse_step.py` | Diffusion schedule and forward/reverse process calculations. |
| `train.py` | Noise-prediction training loop and checkpoint/loss-plot saving. |
| `inference.py` | Image sampling and output visualizations. |
