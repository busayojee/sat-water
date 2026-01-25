"""
Created on Wed Nov 18 13:22:57 2023

@author: Busayo Alabi
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from models import BackboneModels
from preprocess import Preprocess


class Inference:
    backbones = ["resnet34", "unet", "resnet34(2)"]

    def __init__(self, model="", name=None):
        if isinstance(model, str):
            self.model = tf.keras.models.load_model(model, compile=False)
            if name in Inference.backbones:
                self.name = name
            else:
                raise ValueError("name must be resnet34 or unet")

        elif isinstance(model, dict):
            self.model = {}
            for key, value in model.items():
                if key in Inference.backbones:
                    temp_model = tf.keras.models.load_model(value, compile=False)
                    self.model[f"{key}"] = temp_model
                else:
                    print(f"{key} not in list of backbones")
        else:
            raise TypeError("model can either be str or dict")

    def predict(self, image, mask="", image2=None, fname=None):
        mask = {"Mask": mask}
        if isinstance(self.model, dict):
            images = (image - tf.reduce_min(image)) / (
                tf.reduce_max(image) - tf.reduce_min(image)
            )
            n_image = images[tf.newaxis, ...]
            prediction1 = self.model["unet"].predict(n_image)
            prediction1 = tf.argmax(prediction1, axis=-1)
            prediction1 = tf.expand_dims(prediction1, axis=-1)
            prediction1 = prediction1[0, :, :, :]
            prediction1 = {"Unet": prediction1}
            prediction2 = self.model["resnet34"].predict(n_image)
            prediction2 = tf.argmax(prediction2, axis=-1)
            prediction2 = tf.expand_dims(prediction2, axis=-1)
            prediction2 = prediction2[0, :, :, :]
            prediction2 = {"Resnet34": prediction2}
            if image2 is not None:
                images2 = (image2 - tf.reduce_min(image2)) / (
                    tf.reduce_max(image2) - tf.reduce_min(image2)
                )
                n_image2 = images2[tf.newaxis, ...]
                prediction3 = self.model["resnet34(2)"].predict(n_image2)
                prediction3 = tf.argmax(prediction3, axis=-1)
                prediction3 = tf.expand_dims(prediction3, axis=-1)
                prediction3 = prediction3[0, :, :, :]
                prediction3 = 1 - prediction3
                Preprocess.label_overlay(
                    images2,
                    prediction3,
                    fname=f"segmentation/segment-water/results/{fname}.png",
                )
                prediction3 = {"Resnet34(512,512)": prediction3}
                Inference.plot_image(
                    images, prediction1, prediction2, prediction3, fname=fname
                )

            else:
                Inference.plot_image(
                    images, mask, prediction1, prediction2, fname="multiple"
                )
        else:
            if self.name == "unet":
                n_image = image[tf.newaxis, ...]
                inference = self.model.predict(n_image)
                prediction = tf.argmax(inference, axis=-1)
                prediction = tf.expand_dims(prediction, axis=-1)
                prediction = prediction[0, :, :, :]
                fname = "Unet"
            else:
                image, mask = BackboneModels.preprocess(image, mask, "resnet34")
                image = (image - tf.reduce_min(image)) / (
                    tf.reduce_max(image) - tf.reduce_min(image)
                )
                n_image = image[tf.newaxis, ...]
                prediction = self.model.predict(n_image)
                prediction = tf.argmax(prediction, axis=-1)
                prediction = tf.expand_dims(prediction, axis=-1)
                prediction = prediction[0, :, :, :]
                fname = "Resnet34"
                if self.name == "resnet34(2)":
                    prediction = 1 - prediction
                    fname = "Resnet34(2)"
            prediction = {self.name: prediction}
            Inference.plot_image(image, mask, prediction, fname=fname)

    def predict_ds(self, test_ds):
        for image, mask in test_ds.unbatch().shuffle(buffer_size=len(test_ds)).take(1):
            self.predict(image, mask=mask)

    @staticmethod
    def plot_image(image, *args, fname=None):
        arg = [*args]
        num = len(arg)
        # print(num)
        plt.figure(figsize=(12, 10))
        if num % 2 == 1:
            y = 2
        else:
            y = 3

        plt.subplot(num, y, 1)
        plt.title("Image")
        plt.imshow(image)

        n = 2
        for i in args:
            for key, val in i.items():
                plt.subplot(num, y, n)
                plt.title(f"{key}")
                plt.imshow(val)
            n = n + 1
        if fname:
            plt.savefig(f"segmentation/segment-water/results/prediciton_{fname}.png")
        plt.show()

    def predict_inst(self, img_path, fname=None):
        assert isinstance(img_path, str), "Image path must be a string"
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = Preprocess.normalization_layer(image)
        image1 = Preprocess.resize_test(image, (128, 128))
        image2 = Preprocess.resize_test(image, (512, 512))
        self.predict(image1, image2=image2, fname=fname)
