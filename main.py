# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:33:34 2023

@author: Busayo Alabi

A guide on how to use the model.
"""


from preprocess import Preprocess
from models import Unet, BackboneModels
from inference import Inference

# dataset = "segmentation/segment-water/Water Bodies Dataset"
# shape = (128,128,3)
# shape2 = (512,512,3)
# split=[0.2,0.05]
# train_ds, val_ds, test_ds = Preprocess.data_load(dataset, "/Masks", "/Images", split, shape, 16, channels=3)
# Preprocess.plot_image(train_ds)

# # # Using Unet
# history1 = Unet.train(train_ds, val_ds,shape=shape,loss=Unet.loss,metrics = Unet.metrics,name="unet")
# Unet.plot_history(history1,22,model="Unet")
     
# # using Resnet34
# resnet1 = BackboneModels("resnet34", train_ds, val_ds, test_ds, name="resnet34")
# resnet1.build_model(2)
# history2 = resnet1.train()
# Unet.plot_history(history2,18,model="Resnet34")

# # Using(512,512) images
# train_ds1, val_ds1, test_ds1 = Preprocess.data_load(dataset, "/Masks", "/Images", split, shape2, 16, channels=3)
# Preprocess.plot_image(train_ds1)
# resnet2 = BackboneModels("resnet34", train_ds1, val_ds1, test_ds1,name="resnet34(2)")
# resnet2.build_model(2)
# history3 = resnet2.train()
# Unet.plot_history(history3,20,model="Resnet34(2)")

# # Unet
# inference_u = Inference(model="segmentation/segment-water/sat_water2.h5",name="unet")
# inference_u.predict_ds(test_ds)

# # Resnet
# inference_r = Inference(model="segmentation/segment-water/sat_water_resnet34.h5",name="resnet34")
# inference_r.predict_ds(test_ds)

# # Resnet (512,512)
# inference_r2 = Inference(model="segmentation/segment-water/sat_water_resnet34(2).h5",name="resnet34(2)")
# inference_r2.predict_ds(test_ds1)

#Testing multiple models
models={"unet":"segmentation/segment-water/sat_water2.h5", "resnet34":"segmentation/segment-water/sat_water_resnet34.h5", "resnet34(2)":"segmentation/segment-water/sat_water_resnet34(2).h5"}
inference_multiple = Inference(model=models)
# inference_multiple.predict_ds(test_ds)

# Testing an image instance from google or sentinel hub
inference_multiple.predict_inst("segmentation/segment-water/test2.jpg", fname="test2")
                                 