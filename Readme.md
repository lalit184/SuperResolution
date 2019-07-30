# Didactic Supervised Super-resolution

## Introduction

This project incorporates a supervised super-resolution scheme by having the process of super-resolving the image to a slightly lesser resoltion. This approach allows the model to learn necessary features at each scale of resolution.This work is highly inspired by  [Yifan Wang et-al](https://igl.ethz.ch/projects/prosr/) who followed a simillar progressive super-resolution approach.

This repository contains the pytorch implementation of a sequential super resolution algorithm.The architecture and implementation detail are present in the report in the repository

## Getting Started
Copy the entire high res training dataset in png format in a folder called data in the working directory
Copy the low res image to be upscaled in the valid directory in the workspace

### Prerequisites
python3
pytorch
torchvision 

64 GB RAM or more
Atleast 1 GTX 1080ti or better


## Training

```shell
$python3 tain.py
```

## Testing
```shell
$python3 test.py
```
The	above command shall dump 2X,4X and an 8X resolved image in ./result2, ./result4 and ./result8

## Results

|                  2X Resolution                  |                  4X Resolution                  |                  8X Resolution                  |                  Ground Truth                   |
| :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: |
| <img src="/Assets/image--019.jpg" width ="300"> | <img src="/Assets/image--021.jpg" width ="300"> | <img src="/Assets/image--023.jpg" width ="300"> | <img src="/Assets/image--025.jpg" width ="300"> |
| <img src="/Assets/image--031.jpg" width ="300"> | <img src="/Assets/image--033.jpg" width ="300"> | <img src="/Assets/image--035.jpg" width ="300"> | <img src="/Assets/image--037.jpg" width ="300"> |
| <img src="/Assets/image--043.jpg" width ="200"> | <img src="/Assets/image--045.jpg" width ="200"> | <img src="/Assets/image--047.jpg" width ="200"> | <img src="/Assets/image--049.jpg" width ="200"> |
|                                                 |                                                 |                                                 |                                                 |
|                                                 |                                                 |                                                 |                                                 |
|                                                 |                                                 |                                                 |                                                 |

