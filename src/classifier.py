from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import extract_features

CLASSES = ["circle","rectangle","square","triangle"]

def collect_dataset(folder):
    folder = Path(folder)
    X, y, paths = [], [], []
    for label in CLASSES:
        for p in sorted((folder/label).glob("*.png")):
            f, _, _ = extract_features(p)
            X.append(f); y.append(label); paths.append(str(p))
    return np.asarray(X), np.asarray(y), paths

def build_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ])

def train_model(train_dir="data/train"):
    X, y, _ = collect_dataset(train_dir)
    model = build_model()
    model.fit(X, y)
    return model, X, y

def evaluate_model(model, test_dir="data/test"):
    X, y, paths = collect_dataset(test_dir)
    pred = model.predict(X)
    return X, y, pred, paths
