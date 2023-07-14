# Food 101

Trains CV classification models on [Food 101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

Models:
- Vanilla CNN 
- ResNet 50

## Quickstart

With [conda](https://docs.conda.io/en/main/miniconda.html) installed, create and activate environment with the following bash commands:
```bash
>>> conda env create -f environment.yml
>>> conda activate py310_torch
```

###  Training

```bash
python train.py --model Resnet --workers 8 --gpu True --epochs 1 --warm_start True
```
Optional parameters: 
- `--model`
    - Specifies model to train: 
        - `ConvNet`: Vanilla CNN 
        - `Resnet`: Resnet 50, pretrained on Imagenet
        - Can specify any new model by adding to `model.py`
- `--workers`: specifies number of workers for dataloaders
- `--gpu`: 
    - `True`: Runs on CUDA or MPS
    - `False`: Runs on CPU
- `--epochs`: Number of training cycles through full dataset
- `--warm_start`:
    - `True`: Loads pretrained model if prior training was run
    - `False`: Trains new model
