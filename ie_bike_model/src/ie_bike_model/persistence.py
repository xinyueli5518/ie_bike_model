# save and load respectively a scikit-learn classifier in a directory predefined by you
# using joblib as explained in the scikit-learn documentation on model persistence
import joblib

from tempfile import mkdtemp


def persist_model(model, filename):
    """save model"""
    """model: a scikit-learn classifier"""
    """filename: name for the model"""

    path = f"models/{filename}"
    joblib.dump(model, path)


def load_model(filename):
    """load model"""
    """filename: name for the saved model"""

    path = f"models/{filename}"
    return joblib.load(path)
