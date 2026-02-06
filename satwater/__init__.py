import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
