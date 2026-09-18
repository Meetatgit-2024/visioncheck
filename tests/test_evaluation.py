from src.dataset import generate_dataset
from src.evaluate import run_evaluation

def test_train_test_evaluation(tmp_path):
    generate_dataset(tmp_path, train_per_class=8, test_per_class=3)
    r = run_evaluation(tmp_path/"train", tmp_path/"test", tmp_path/"outputs")
    assert r["training_samples"] == 32
    assert r["test_samples"] == 12
    assert 0 <= r["accuracy"] <= 1
    assert (tmp_path/"outputs"/"evaluation.json").exists()
