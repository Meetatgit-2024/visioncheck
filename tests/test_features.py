from src.dataset import generate_dataset
from src.features import extract_features

def test_feature_extraction(tmp_path):
    generate_dataset(tmp_path, train_per_class=1, test_per_class=0)
    f, contour, binary = extract_features(tmp_path/"train"/"circle"/"circle_001.png")
    assert len(f) == 6
    assert contour is not None
    assert binary.shape == (256,256)
