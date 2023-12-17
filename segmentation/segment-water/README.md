## Introduction
Satellite imagery is a rich source of information, and the accurate segmentation of water bodies is crucial for understanding environmental patterns and changes over time. This project aims to provide a reliable and efficient tool for extracting water regions from raw satellite images.

## Dataset
The dataset for this project is gotten here [kaggle.com](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies). It consists of jpeg images of water bodies taken by satellites and their mask. More details of the dataset is provided on the website.

## Installation
To get started with the project you can clone this repo. 

```git clone https://github.com/busayojee/Satellite-imagery.git```

```cd segmentation/segment-water```

## Preprocessing
After downloading the images from kaggle, To preprocess the images you import the preprocess class from preprocess.py and run this code 

```train_ds, val_ds, test_ds = Preprocess.data_load(dataset, "/Masks", "/Images", split, shape, 16, channels=3)```

See [main.py](https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/main.py) to help.


## Model
This project was trained on 2 models. The <b>UNET</b> and the <b>RESNET</b> of which 2 different models were trained on different sizes of images and also different hyperparameters. 

## Training
To train the model using Unet 

```history1 = Unet.train(train_ds, val_ds,shape=shape,loss=Unet.loss,metrics = Unet.metrics,name="unet")```

To train the model using Resnet(256,256)

```
resnet1 = BackboneModels("resnet34", train_ds, val_ds, test_ds, name="resnet34")
resnet1.build_model(2)
history2 = resnet1.train()
```
