# Bechmark PyTorch MPS

[Benchmark results for Macbook Air (M1, 2020)](https://github.com/gau-nernst/benchmark-pytorch-mps/issues/1)

Support calling models from:

- torchvision
- transformers

Some models can't be run on `mps` device due to unsupported ops e.g. Swin transformer.

## Environment setup

Starting from PyTorch 1.12 and torchvision 0.13, there are official binaries for Mac M1.

You might need to update to macOS >= 12.3

Install [miniforge](https://github.com/conda-forge/miniforge), then run the below

```bash
# install PyTorch nightly
conda create -n pytorch python=3.8
conda activate pytorch
conda install pytorch torchvision -c pytorch
conda install transformers      # for BERT
conda install pandas tabulate   # to print pretty tables
```

## Usage

```bash
python benchmark.py --model_name=resnet50 --batch_sizes=1,4,16,64,256 --size=224
```
