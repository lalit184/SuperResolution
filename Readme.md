# Didactic Supervised Super-resolution

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
	
	$python3 tain.py

## Testing
	$python3 test.py
The	above command shall dump 2X,4X and an 8X resolved image in ./result2, ./result4 and ./result8
