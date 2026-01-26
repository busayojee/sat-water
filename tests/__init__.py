import os


def pytest_configure():
    # Reduce TensorFlow noise in CI/local output and don't plot diagrams.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
