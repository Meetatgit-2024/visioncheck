import cv2
import numpy as np

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return gray, binary

def largest_contour(binary, min_area=200):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    return max(valid, key=cv2.contourArea) if valid else None

def extract_features(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    _, binary = preprocess(image)
    contour = largest_contour(binary)
    if contour is None:
        raise ValueError(f"No valid object detected in: {path}")

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x,y,w,h = cv2.boundingRect(contour)
    circularity = 4*np.pi*area/(perimeter**2) if perimeter else 0
    aspect_ratio = w/h if h else 0
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area/hull_area if hull_area else 0
    approx = cv2.approxPolyDP(contour, 0.04*perimeter, True)
    vertices = len(approx)
    # Scale area and perimeter by image dimensions for better invariance.
    diag = float(np.hypot(image.shape[1], image.shape[0]))
    norm_area = area/(image.shape[0]*image.shape[1])
    norm_perimeter = perimeter/diag
    return np.array([norm_area, norm_perimeter, aspect_ratio, circularity,
                     solidity, vertices], dtype=float), contour, binary
