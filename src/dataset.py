from pathlib import Path
import cv2
import numpy as np
import random

CLASSES = ["circle", "square", "rectangle", "triangle"]

def _background(rng, size=256):
    base = rng.integers(225, 256, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (7, 7), 0)

def _draw_shape(img, label, rng):
    h, w = img.shape[:2]
    cx = int(rng.integers(65, w-65))
    cy = int(rng.integers(65, h-65))
    angle = float(rng.uniform(0, 360))
    scale = float(rng.uniform(0.75, 1.15))
    shade = int(rng.integers(25, 90))

    layer = np.full_like(img, 255)
    if label == "circle":
        r = int(45 * scale)
        cv2.circle(layer, (cx, cy), r, (shade, shade, shade), -1)
    elif label == "square":
        s = int(75 * scale)
        pts = np.array([[-s,-s],[s,-s],[s,s],[-s,s]], np.float32)
        M = cv2.getRotationMatrix2D((0,0), angle, 1)
        pts = pts @ M[:,:2].T + M[:,2]
        pts += np.array([cx,cy])
        cv2.fillPoly(layer, [np.round(pts).astype(np.int32)], (shade,)*3)
    elif label == "rectangle":
        a, b = int(85*scale), int(45*scale)
        pts = np.array([[-a,-b],[a,-b],[a,b],[-a,b]], np.float32)
        M = cv2.getRotationMatrix2D((0,0), angle, 1)
        pts = pts @ M[:,:2].T + M[:,2]
        pts += np.array([cx,cy])
        cv2.fillPoly(layer, [np.round(pts).astype(np.int32)], (shade,)*3)
    else:
        a = int(rng.uniform(50, 75) * scale)
        pts = np.array([[0,-a],[int(a*rng.uniform(.75,1.2)),int(a*.8)],
                        [-int(a*rng.uniform(.75,1.2)),int(a*.8)]], np.float32)
        M = cv2.getRotationMatrix2D((0,0), angle, 1)
        pts = pts @ M[:,:2].T + M[:,2]
        pts += np.array([cx,cy])
        cv2.fillPoly(layer, [np.round(pts).astype(np.int32)], (shade,)*3)

    # Slight blur and sensor-like noise make the samples less identical.
    layer = cv2.GaussianBlur(layer, (3,3), float(rng.uniform(.1, .8)))
    mask = np.any(layer < 240, axis=2)
    img[mask] = layer[mask]
    noise = rng.normal(0, rng.uniform(0, 3.5), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)

def generate_dataset(root="data", train_per_class=40, test_per_class=10, seed=42):
    root = Path(root)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    for split, count in [("train", train_per_class), ("test", test_per_class)]:
        for label in CLASSES:
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = _background(rng)
                img = _draw_shape(img, label, rng)
                cv2.imwrite(str(folder / f"{label}_{i+1:03d}.png"), img)
    return train_per_class * len(CLASSES), test_per_class * len(CLASSES)

if __name__ == "__main__":
    tr, te = generate_dataset()
    print(f"Generated {tr} training images and {te} test images.")
