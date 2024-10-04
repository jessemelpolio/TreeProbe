# Continual Learning in Open-vocabulary Classification with Complementary Memory Systems

## TMLR 2024

[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=6j5M75iK3a)
[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-blue)](https://arxiv.org/abs/2307.01430)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://zzhu.vision/treeprobe/)
[![YouTube Video](https://img.shields.io/badge/YouTube-Video-red)](...)

## Authors
[Zhen Zhu](https://zzhu.vision) · Weijie Lyu · [Yao Xiao](https://avaxiao.github.io) · [Derek Hoiem](http://dhoiem.cs.illinois.edu)

## Overview

We introduce a method for flexible and efficient continual learning in open-vocabulary image classification, drawing inspiration from the complementary learning systems observed in human cognition. Specifically, we propose to combine predictions from a CLIP zero-shot model and the exemplar-based model, using the zero-shot estimated probability that a sample's class is within the exemplar classes. We also propose a "tree probe" method, an adaption of lazy learning principles, which enables fast learning from new examples with competitive accuracy to batch-trained linear models. We test in data incremental, class incremental, and task incremental settings, as well as ability to perform flexible inference on varying subsets of zero-shot and learned categories. Our proposed method achieves a good balance of learning speed, target task effectiveness, and zero-shot effectiveness.

## Hardware
We test our code on a single NVIDIA RTX 3090Ti GPU.

## Installation

### Prerequisites
- Anaconda or Miniconda
- Git

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/jessemelpolio/TreeProbe.git
   cd TreeProbe
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate TreeProbe
   ```


## Project Structure
- `data/`: Dataset handling and preprocessing
- `encode_features/`: Scripts for encoding features using CLIP
- `engines/`: Engine implementations for training and evaluation
- `models/`: Model architectures and components
- `options/`: Command-line argument parsing
- `scripts/`: Utility scripts
- `main_xx.py`: Main entry point for running experiments. `xx` can be `data`, `task`, or `class`.

## Usage
1. **Prepare datasets:**
   Our project uses various datasets for target tasks and zero-shot tasks.

   <details>
   <summary>Click to expand dataset details</summary>

   **Target Tasks:** CIFAR100, SUN397, EuroSAT, OxfordIIITPet, Flowers102, FGVCAircraft, StanfordCars, Food101

   **Zero-shot Tasks:** ImageNet, UCF101, DTD

   > **Note:** SUN397, EuroSAT, UCF101, and ImageNet require manual downloading from their original sources. Please follow the instructions in [`tutorials/download_data.md`](tutorials/download_data.md) to obtain these datasets. Other datasets can be easily downloaded using the `torchvision.datasets` package. We also provide additional datasets in the `data/` folder for your convenience but be aware that they are not tested rigorously and may not work with the codebase.
   </details>

   To encode the intermediate image representations of these datasets to speed up training, check the script in [`scripts/encode_features.sh`](scripts/encode_features.sh). After setting the correct data root in the script, you can run the script with:
   ```
   bash scripts/encode_features.sh
   ```

2. **Train and evaluate:**
   Example scripts for task, data, and class-incremental learning:
   <details>
   <summary>Click to expand example scripts</summary>

   ```
   bash scripts/task_incremental.sh
   ```
   ```
   bash scripts/data_incremental.sh
   ```
   ```
   bash scripts/class_incremental.sh
   ```
   </details>

## Warning
This codebase is only tested under a single GPU. If you want to use multiple GPUs, you need to modify the codebase. 

We'd appreciate it if you could report any issues you encounter.


## Configuration Options

Our approach offers various customization options to create different experimental settings. Refer to [`tutorials/configuration_options.md`](tutorials/configuration_options.md) for more details.


## Bibtex

If you use this code for your research, please consider citing:
```
@article{zhu2024treeprobe,
  author       = {Zhen Zhu and Weijie Lyu and Yao Xiao and Derek Hoiem},
  title        = {Continual Learning in Open-vocabulary Classification with Complementary Memory Systems},
  journal      = {Trans. Mach. Learn. Res.},
  volume       = {2024},
  year         = {2024},
  url          = {https://openreview.net/forum?id=6j5M75iK3a}
}
```

## Acknowledgements
- This project uses [DINOv2](https://github.com/facebookresearch/dinov2) by Facebook Research.
- The project incorporates [CLIP](https://github.com/openai/CLIP) for vision-language learning.
- The arguments configuration is inspired from [SPADE](https://github.com/NVlabs/SPADE).
- This codebase shares a lot in common with [AnytimeCL](https://github.com/jessemelpolio/AnytimeCL).
