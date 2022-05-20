# Bechmark PyTorch MPS

## Results

- System: MacBook Air (M1, 2020), 16GB Memory, macOS 12.4
- Python env: Python 3.8, torch=1.12.0.dev20220520, torchvision=0.13.0a0+ac56f52
- Image size: 224 x 224

torchvision's ResNet-50

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>cpu</th>
      <th>mps</th>
      <th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Forward (inference)</th>
      <th>batch 1</th>
      <td>0.019279</td>
      <td>0.009077</td>
      <td>2.124094</td>
    </tr>
    <tr>
      <th>batch 4</th>
      <td>0.061646</td>
      <td>0.017558</td>
      <td>3.510889</td>
    </tr>
    <tr>
      <th>batch 16</th>
      <td>0.236246</td>
      <td>0.051581</td>
      <td>4.580057</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Forward (training)</th>
      <th>batch 1</th>
      <td>0.018578</td>
      <td>0.029572</td>
      <td>0.628213</td>
    </tr>
    <tr>
      <th>batch 4</th>
      <td>0.067474</td>
      <td>0.063275</td>
      <td>1.066354</td>
    </tr>
    <tr>
      <th>batch 16</th>
      <td>0.237479</td>
      <td>0.179594</td>
      <td>1.322313</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Backward (training)</th>
      <th>batch 1</th>
      <td>0.035397</td>
      <td>0.020894</td>
      <td>1.694124</td>
    </tr>
    <tr>
      <th>batch 4</th>
      <td>0.150062</td>
      <td>0.033974</td>
      <td>4.416930</td>
    </tr>
    <tr>
      <th>batch 16</th>
      <td>0.447114</td>
      <td>0.079660</td>
      <td>5.612775</td>
    </tr>
  </tbody>
</table>

torchvision's EfficientNet-B0

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>cpu</th>
      <th>mps</th>
      <th>speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Forward (inference)</th>
      <th>batch 1</th>
      <td>0.166054</td>
      <td>0.015404</td>
      <td>10.780234</td>
    </tr>
    <tr>
      <th>batch 4</th>
      <td>0.325193</td>
      <td>0.044354</td>
      <td>7.331710</td>
    </tr>
    <tr>
      <th>batch 16</th>
      <td>0.774903</td>
      <td>0.165620</td>
      <td>4.678795</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Forward (training)</th>
      <th>batch 1</th>
      <td>0.215004</td>
      <td>0.085171</td>
      <td>2.524385</td>
    </tr>
    <tr>
      <th>batch 4</th>
      <td>0.393971</td>
      <td>0.151698</td>
      <td>2.597070</td>
    </tr>
    <tr>
      <th>batch 16</th>
      <td>0.999952</td>
      <td>0.504220</td>
      <td>1.983165</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Backward (training)</th>
      <th>batch 1</th>
      <td>0.277020</td>
      <td>0.074291</td>
      <td>3.728836</td>
    </tr>
    <tr>
      <th>batch 4</th>
      <td>0.578525</td>
      <td>0.189818</td>
      <td>3.047796</td>
    </tr>
    <tr>
      <th>batch 16</th>
      <td>1.457229</td>
      <td>0.628707</td>
      <td>2.317819</td>
    </tr>
  </tbody>
</table>

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
