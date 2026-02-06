"""
Created on Wed Nov 17 11:05:25 2023

@author: Busayo Alabi
"""

import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except Exception as e:
    raise ImportError(
        "TensorFlow is required for sat-water inference/training.\n\n"
        "Install TensorFlow first, then reinstall sat-water.\n"
        "Recommended:\n"
        "  Linux/Windows: pip install 'tensorflow'\n"
        "  Apple Silicon: pip install 'tensorflow-macos' 'tensorflow-metal'\n\n"
        "If you are using segmentation-models with TF legacy Keras:\n"
        "  pip install tf-keras segmentation-models\n"
    ) from e

try:
    import segmentation_models as sm
except Exception as e:
    raise ImportError(
        "segmentation-models is required for training backbone models.\n"
        "Install it with:\n"
        "  pip install segmentation-models tf-keras\n"
    ) from e

from satwater.preprocess import Preprocess


class Unet:
    loss = sm.losses.categorical_focal_dice_loss
    metrics = [sm.metrics.iou_score, sm.metrics.f1_score]

    @staticmethod
    def dice_coef_water(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.cast(y_pred[..., 1], tf.float32)

        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f + y_pred_f, axis=1)

        dice = (2.0 * intersection + smooth) / (denom + smooth)
        return tf.reduce_mean(dice)

    @staticmethod
    def conv_block(input, filters):
        encoder = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(input)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation("relu")(encoder)
        encoder = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation("relu")(encoder)
        return encoder

    @staticmethod
    def encoder_block(input, filters):
        encoder = Unet.conv_block(input, filters)
        encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        return encoder_pool, encoder

    @staticmethod
    def decoder_block(input, concat, filters):
        decoder = tf.keras.layers.Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding="same"
        )(input)
        decoder = tf.keras.layers.concatenate([concat, decoder], axis=-1)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation("relu")(decoder)
        decoder = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation("relu")(decoder)
        decoder = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(decoder)
        decoder = tf.keras.layers.BatchNormalization()(decoder)
        decoder = tf.keras.layers.Activation("relu")(decoder)
        return decoder

    @staticmethod
    def models(input_shape, n_classes):
        inputs = tf.keras.layers.Input(shape=input_shape)
        encoder_pool0, encoder0 = Unet.encoder_block(inputs, 32)
        encoder_pool1, encoder1 = Unet.encoder_block(encoder_pool0, 64)
        encoder_pool2, encoder2 = Unet.encoder_block(encoder_pool1, 128)
        encoder_pool3, encoder3 = Unet.encoder_block(encoder_pool2, 256)
        encoder_pool4, encoder4 = Unet.encoder_block(encoder_pool3, 512)
        center = Unet.conv_block(encoder_pool4, 1024)
        decoder4 = Unet.decoder_block(center, encoder4, 512)
        decoder3 = Unet.decoder_block(decoder4, encoder3, 256)
        decoder2 = Unet.decoder_block(decoder3, encoder2, 128)
        decoder1 = Unet.decoder_block(decoder2, encoder1, 64)
        decoder0 = Unet.decoder_block(decoder1, encoder0, 32)
        outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation="softmax")(
            decoder0
        )
        models = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
        return models

    @staticmethod
    def checkpoint():
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_iou_score", patience=6
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=4, min_lr=0.00005
        )
        return [early_stopping, reduce_lr]

    @staticmethod
    def train(
        train_ds,
        val_ds,
        epochs=100,
        shape=(128, 128, 3),
        n_classes=2,
        lr=0.0001,
        loss="sparse_categorical_crossentropy",
        metrics=None,
        name="model",
    ):
        if metrics is None:
            metrics = ["accuracy"]
        model = Unet.models(shape, n_classes)
        callbacks = Unet.checkpoint()
        adam = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        model.compile(optimizer=adam, loss=loss, metrics=metrics)
        tf.keras.utils.plot_model(
            model, to_file=f"segmentation/segment-water/model_{name}.png"
        )
        print(model.summary())
        history = model.fit(
            train_ds,
            epochs=100,
            shuffle=True,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_ds,
        )
        model.save(f"{name}.h5")
        return history

    @staticmethod
    def plot_history(history, epochs, model=""):
        IOU = history.history["iou_score"]
        val_IOU = history.history["val_iou_score"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs_range = range(epochs)
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, IOU, label="Training IOU coefficient")
        plt.plot(epochs_range, val_IOU, label="Validation IOU coefficient")
        plt.legend(loc="upper right")
        plt.title("Training and Validation IOU coefficient")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.savefig(f"segmentation/segment-water/results/history{model}.png")
        plt.show()


class BackboneModels:
    parallel_calls = tf.data.AUTOTUNE

    def __init__(self, Backbone, train_ds, val_ds, test_ds, name="model"):
        assert isinstance(Backbone, str), "Backbone must be string"
        self.backbone = Backbone
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.name = name
        self.model = None

    @staticmethod
    def preprocess(images, label, Backbone):
        preprocessor = sm.get_preprocessing(Backbone)
        return preprocessor(images), label

    def build_model(self, n_classes, n=1, lr=0.0001):
        self.train_ds = self.train_ds.map(
            lambda x, y: BackboneModels.preprocess(x, y, self.backbone),
            num_parallel_calls=BackboneModels.parallel_calls,
        )
        self.val_ds = self.val_ds.map(
            lambda x, y: BackboneModels.preprocess(x, y, self.backbone),
            num_parallel_calls=BackboneModels.parallel_calls,
        )
        self.test_ds = self.test_ds.map(
            lambda x, y: BackboneModels.preprocess(x, y, self.backbone),
            num_parallel_calls=BackboneModels.parallel_calls,
        )
        model = sm.Unet(
            self.backbone,
            classes=n_classes,
            activation="softmax",
            encoder_weights="imagenet",
        )
        adam = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        model.compile(optimizer=adam, loss=Unet.loss, metrics=Unet.metrics)
        tf.keras.utils.plot_model(
            model, to_file=f"segmentation/segment-water/model_{self.name}.png"
        )
        self.model = model
        Preprocess.plot_image(self.train_ds, n)
        return self.model.summary()

    def train(self):
        if self.model:
            history1 = self.model.fit(
                self.train_ds,
                epochs=100,
                verbose=1,
                callbacks=Unet.checkpoint(),
                shuffle=True,
                validation_data=self.val_ds,
            )
            self.model.save(f"sat_water_{self.name}.h5")
            return history1
        else:
            return "A model has to be built with the build_model method first"
