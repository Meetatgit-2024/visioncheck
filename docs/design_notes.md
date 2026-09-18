# Design and Evaluation Notes

## System Architecture
Input Image → Preprocessing → Segmentation → Contour Extraction → Feature Engineering → Standardization → k-NN Classifier → Evaluation

## Major Functional Modules
1. **Dataset Generation:** creates varied labeled images and maintains independent train/test partitions.
2. **Feature Extraction:** converts visual shape information into numerical geometric descriptors.
3. **Classification:** trains a distance-weighted k-NN model on standardized features.
4. **Evaluation:** calculates classification metrics and generates a confusion matrix.

## Non-Functional Requirements
- **Performance:** lightweight CPU-based processing for small images.
- **Reliability:** explicit errors for unreadable images or missing contours.
- **Maintainability:** responsibilities separated into dedicated modules.
- **Reproducibility:** fixed random seed and deterministic command-line workflow.
- **Usability:** installation and execution require only standard terminal commands.
- **Resource efficiency:** no GPU, cloud service, or large pretrained model is required.

## Why Train/Test Instead of Repeating Identical Shapes?
The purpose of the revised dataset is to evaluate whether the classifier can generalize from varied training examples to unseen examples. Each test image is independently generated and is not used for fitting.

## Why k-NN?
k-NN provides a transparent baseline for a small feature-based classification problem. It is computationally lightweight and makes the relationship between extracted geometric features and predicted class easy to explain during evaluation.

## Evaluation Metrics
Accuracy measures the proportion of correctly classified test samples. Precision measures correctness among samples assigned to a class, recall measures how many samples belonging to a class were recovered, and F1-score summarizes precision and recall. The confusion matrix provides class-level error analysis.
