# Bechmark PyTorch MPS

## Results

- System: MacBook Air (M1, 2020), 16GB Memory, macOS 12.4
- Python env: Python 3.8, torch=1.12.0.dev20220520, torchvision=0.13.0a0+ac56f52
- Model: torchvision's ResNet-50
- Image size: 224 x 224

Batch size 1

|                     |       cpu |       mps |   speedup |
|:--------------------|----------:|----------:|----------:|
| Forward (inference) | 0.0357982 | 0.0202892 |  1.76439  |
| Forward (training)  | 0.0388275 | 0.0580117 |  0.669303 |
| Backward (training) | 0.0756178 | 0.058872  |  1.28444  |

Batch size 4

|                     |      cpu |       mps |   speedup |
|:--------------------|---------:|----------:|----------:|
| Forward (inference) | 0.1357   | 0.0509956 |   2.66101 |
| Forward (training)  | 0.130575 | 0.127025  |   1.02795 |
| Backward (training) | 0.29858  | 0.109849  |   2.7181  |

Batch size 16

|                     |      cpu |      mps |   speedup |
|:--------------------|---------:|---------:|----------:|
| Forward (inference) | 0.450937 | 0.176198 |   2.55926 |
| Forward (training)  | 0.513482 | 0.396306 |   1.29567 |
| Backward (training) | 1.01558  | 0.310709 |   3.26858 |

## Environment setup

You might need to update to macOS >= 12.3

Install [miniforge](https://github.com/conda-forge/miniforge), then run the below

```bash
# install PyTorch nightly
conda create -n pytorch python=3.8
conda activate pytorch
conda install pytorch -c pytorch-nightly

# build torchvision from source
git clone https://github.com/pytorch/vision
cd vision
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

# extra libraries to print pretty tables
conda install pandas tabulate
```
