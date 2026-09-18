# VisionCheck: Automated Shape Recognition and Geometric Analysis
##  Student Details

| Field                   | Value                       |
| ----------------------- | --------------------------- |
| **Name**                | Meet Chaure                 |
| **Registration Number** | 24BAI10922                  |
| **Course**              | Computer Vision             |
| **Date**                | September 2026              |


## 1. Overview
VisionCheck is a command-line computer-vision project that recognizes four geometric object classes—**circle, square, rectangle, and triangle**—from images. Unlike the earlier demonstration-only version, this implementation uses a **varied dataset with independent training and test partitions** and performs quantitative supervised evaluation.

The pipeline combines classical image processing with feature-based machine learning:

**Image → Preprocessing → Segmentation → Contour → Feature Extraction → Feature Scaling → k-NN Classification → Evaluation**

## 2. Objectives
- Construct a varied, reproducible image dataset.
- Separate training and testing data to evaluate generalization on unseen images.
- Extract meaningful geometric descriptors from contours.
- Train a k-nearest-neighbour classifier.
- Report accuracy, precision, recall, F1-score, and a confusion matrix.
- Provide a fully executable command-line workflow.

## 3. Features
- Four shape classes.
- 160 training images and 40 test images by default.
- Variation in object size, position, orientation, shade, background, and small image noise.
- Grayscale conversion, Gaussian filtering, Otsu thresholding, and morphology.
- External contour detection.
- Six geometric features: normalized area, normalized perimeter, aspect ratio, circularity, solidity, and polygon vertex count.
- StandardScaler + distance-weighted k-NN.
- JSON evaluation report.
- Confusion-matrix PNG.
- Automated tests.

## 4. Technologies
- Python 3.10+
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Pytest
- Git/GitHub

## 5. Repository Structure
```text
visioncheck/
├── README.md
├── statement.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── features.py
│   ├── classifier.py
│   ├── evaluate.py
│   └── main.py
├── tests/
│   ├── test_features.py
│   └── test_evaluation.py
├── data/
│   ├── train/
│   │   ├── circle/
│   │   ├── square/
│   │   ├── rectangle/
│   │   └── triangle/
│   └── test/
│       ├── circle/
│       ├── square/
│       ├── rectangle/
│       └── triangle/
├── outputs/
└── docs/
```

## 6. Installation

### Windows
```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS
```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Generate the Dataset
From the repository root:
```text
python -m src.main --generate-data
```

This creates:
- 40 training images per class × 4 classes = **160 training images**
- 10 test images per class × 4 classes = **40 test images**

The generator introduces controlled variation in:
- object position;
- size;
- rotation for non-circular shapes;
- intensity/shade;
- background;
- small Gaussian noise.

The test images are generated as a separate partition and are not used during classifier fitting.

## 8. Run Train/Test Evaluation
```text
python -m src.main --evaluate
```

The command:
1. extracts features from all training images;
2. fits a standardized distance-weighted k-NN classifier;
3. extracts features from the independent test set;
4. predicts the test labels;
5. computes accuracy, precision, recall, and F1-score;
6. creates a confusion matrix;
7. writes `outputs/evaluation.json`.

## 9. Run Both Steps
For a clean reproduction:
```text
python -m src.main --generate-data
python -m src.main --evaluate
```

## 10. Feature Engineering
For each detected contour, the system calculates:
- normalized contour area;
- normalized perimeter;
- bounding-box aspect ratio;
- circularity;
- solidity;
- polygon vertex count.

These features capture complementary geometric properties and are standardized before k-NN classification.

## 11. Train/Test Methodology
The training set is used only for fitting the k-NN model. The test set is held out and used only after training. Therefore, the reported test metrics measure performance on images that were not used to fit the classifier.

The dataset is synthetic and controlled, so the results should be interpreted as an evaluation of the implemented pipeline rather than evidence of performance on unconstrained real-world photographs.

## 12. Testing
Run:
```text
pytest -q
```

The tests cover feature extraction and a complete train/test evaluation using a temporary dataset.

## 13. Expected Outputs
After evaluation:
```text
outputs/
├── evaluation.json
└── confusion_matrix.png
```

The JSON file contains:
- number of training samples;
- number of test samples;
- overall accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- per-class metrics;
- confusion matrix.

## 14. Limitations
The dataset is synthetic and contains one prominent object per image. Performance may decrease with occlusion, multiple overlapping objects, cluttered backgrounds, severe illumination changes, perspective distortion, or shapes that are substantially different from the generated classes.

## 15. Reproducibility
Dataset generation uses a fixed default random seed. The complete workflow can therefore be recreated from the terminal after dependency installation.


