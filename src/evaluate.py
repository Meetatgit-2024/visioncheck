import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from .classifier import train_model, evaluate_model, CLASSES

def run_evaluation(train_dir="data/train", test_dir="data/test", output_dir="outputs"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model, X_train, y_train = train_model(train_dir)
    X_test, y_test, pred, paths = evaluate_model(model, test_dir)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=CLASSES, zero_division=0
    )
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    report = {
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "classes": CLASSES,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for c,p,r,f,s in zip(CLASSES,precision,recall,f1,support)
        },
        "confusion_matrix": cm.tolist()
    }
    (out/"evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("VisionCheck Test Confusion Matrix")
    plt.colorbar()
    ticks = range(len(CLASSES))
    plt.xticks(ticks, CLASSES, rotation=30)
    plt.yticks(ticks, CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out/"confusion_matrix.png", dpi=160)
    plt.close()
    return report

if __name__ == "__main__":
    r = run_evaluation()
    print(json.dumps(r, indent=2))
