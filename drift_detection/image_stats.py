import cv2
import os
import numpy as np
from scipy.stats import entropy

def image_statistics(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    mean = np.mean(img)
    std = np.std(img)

    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist.flatten() + 1e-6
    hist /= hist.sum()

    ent = entropy(hist)

    return mean, std, ent


def compute_dataset_stats(image_dir):
    stats = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        s = image_statistics(img_path)
        if s:
            stats.append(s)

    return np.array(stats)
