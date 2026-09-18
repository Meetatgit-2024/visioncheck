# Project Statement

## Problem Statement
Manual identification of geometric objects in digital images is repetitive and may produce inconsistent classifications. VisionCheck addresses this problem by applying a reproducible computer-vision pipeline to detect a prominent object, extract geometric descriptors, classify its shape, and evaluate classification performance on previously unseen test images.

## Scope
The system focuses on four geometric classes: circle, square, rectangle, and triangle. It covers synthetic image generation, preprocessing, contour extraction, feature engineering, supervised classification using k-nearest neighbours, quantitative evaluation, confusion-matrix visualization, and command-line execution.

## Target Users
- Students learning applied computer vision and image processing.
- Academic evaluators assessing an end-to-end computer-vision implementation.
- Developers seeking a compact baseline for geometric object recognition.

## High-Level Features
- Varied synthetic dataset generation.
- Independent training and test sets.
- Image preprocessing and contour extraction.
- Geometric feature extraction.
- Standardized feature scaling.
- k-nearest-neighbour classification.
- Accuracy, precision, recall, F1-score, and confusion matrix.
- Reproducible command-line workflow.
