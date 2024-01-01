# SEGMENTATION OF WATER BODIES FROM SATELLITE IMAGES
## Introduction
Satellite imagery is a rich source of information, and the accurate segmentation of water bodies is crucial for understanding environmental patterns and changes over time. This project aims to provide a reliable and efficient tool for extracting water regions from raw satellite images.

## Dataset
The dataset for this project is gotten here [kaggle.com](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies). It consists of jpeg images of water bodies taken by satellites and their mask. More details of the dataset is provided on the website.

## Installation
To get started with the project you can clone this repo. 

```git clone https://github.com/busayojee/sat-water.git```

## Preprocessing
After downloading the images from kaggle, To preprocess the images you import the preprocess class from preprocess.py and run this code 

```train_ds, val_ds, test_ds = Preprocess.data_load(dataset, "/Masks", "/Images", split, shape, 16, channels=3)```

See [main.py](https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/main.py) to help.


## Model
This project was trained on 2 models. The <b>UNET</b> with no backbone and the UNET with a <b>RESNET34</b> backbone of which 2 different models were trained on different sizes of images and also different hyperparameters. 

## Training
To train the Unet model

```history1 = Unet.train(train_ds, val_ds,shape=shape,loss=Unet.loss,metrics = Unet.metrics,name="unet")```

To train the Resnet(256,256) model

```
resnet1 = BackboneModels("resnet34", train_ds, val_ds, test_ds, name="resnet34")
resnet1.build_model(2)
history2 = resnet1.train()
```

To train the Resnet(512,512) model
```
resnet2 = BackboneModels("resnet34", train_ds1, val_ds1, test_ds1,name="resnet34(2)")
resnet2.build_model(2)
history3 = resnet2.train()
```

#### Results

| Unet | Resnet(256,256) | Resnet(512,512)
:--------:|:--------:|:--------:
|<img width="250" alt="Unet" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/history_unet.png">|<img width="250" alt="Resnet1" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/history_resnet34.png">|<img width="250" alt="Resnet2" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/historyresnet34(2).png">|

## Inference
To run inference for UNET

```
inference_u = Inference(model="segmentation/segment-water/sat_water2.h5",name="unet")
inference_u.predict_ds(test_ds)
```

for RESNET 1 and 2

```
inference_r = Inference(model="segmentation/segment-water/sat_water_resnet34.h5",name="resnet34")
inference_r.predict_ds(test_ds)

inference_r2 = Inference(model="segmentation/segment-water/sat_water_resnet34(2).h5",name="resnet34(2)")
inference_r2.predict_ds(test_ds1)
```

For all 3 models together

```
models={"unet":"segmentation/segment-water/sat_water2.h5", "resnet34":"segmentation/segment-water/sat_water_resnet34.h5", "resnet34(2)":"segmentation/segment-water/sat_water_resnet34(2).h5"}
inference_multiple = Inference(model=models)
inference_multiple.predict_ds(test_ds)
```

#### Results
| Unet | Resnet(256,256) | Resnet(512,512) | 
:--------:|:--------:|:--------:
|<img width="250" alt="Unet" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/prediciton_unet.png">|<img width="250" alt="Resnet1" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/prediciton_resnet34.png">|<img width="250" alt="Resnet2" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/prediciton_resnet34(2).png">| 

## Single Test Instance
Using all models to predict a test instance gotten from google

| Test Image | Prediction |
:--------:|:--------:
|<img width="250" alt="test_instance" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/test2.jpg">| <img width="250" alt="Prediction" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/prediciton_test.png">|

Label overlaying the best prediction (Resnet(512,512))

 <img width="250" alt="Unet" src="https://github.com/busayojee/Satellite-imagery/blob/main/segmentation/segment-water/results/test2.png">

