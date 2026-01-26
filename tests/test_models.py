import tensorflow as tf

from models import BackboneModels, Unet, sm
from preprocess import Preprocess


def test_custom_unet_builds_and_forward_pass_runs():
    model = Unet.models(input_shape=(32, 32, 3), n_classes=2)
    x = tf.random.uniform((1, 32, 32, 3))
    y = model(x)
    assert y.shape == (1, 32, 32, 2)


def test_backbone_models_build_model_without_downloading_weights(monkeypatch):
    monkeypatch.setattr(tf.keras.utils, "plot_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(Preprocess, "plot_image", lambda *args, **kwargs: None)

    # identity preprocessing
    def dummy_get_preprocessing(_):
        def _preprocess(images):
            return images

        return _preprocess

    monkeypatch.setattr(sm, "get_preprocessing", dummy_get_preprocessing)

    # dummy sm.Unet model
    def dummy_sm_unet(backbone, classes, activation, encoder_weights):
        inp = tf.keras.Input(shape=(32, 32, 3))
        out = tf.keras.layers.Conv2D(classes, 1, activation=activation)(inp)
        return tf.keras.Model(inp, out)

    monkeypatch.setattr(sm, "Unet", dummy_sm_unet)

    # small dataset
    x = tf.random.uniform((8, 32, 32, 3))
    y = tf.one_hot(tf.random.uniform((8, 32, 32), maxval=2, dtype=tf.int32), depth=2)
    y = tf.cast(y, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
    bm = BackboneModels("resnet34", ds, ds, ds, name="test")

    bm.build_model(n_classes=2, n=1, lr=1e-4)
    assert bm.model is not None
