#DRIFT TRANSFORM FUNCTIONS
import os
import cv2
import numpy as np
from tqdm import tqdm

SRC_IMAGES = "data/valid/images"

STREAMS = {
    "data/stream_day_1/images": "light",
    "data/stream_day_7/images": "medium",
    "data/stream_day_30/images": "heavy"
}

os.makedirs("data", exist_ok=True)

def adjust_brightness(img, factor):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def motion_blur(img, ksize):
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize-1)/2), :] = np.ones(ksize)
    kernel /= ksize
    return cv2.filter2D(img, -1, kernel)

def partial_occlusion(img):
    h, w, _ = img.shape
    x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
    x2, y2 = x1 + w//4, y1 + h//4
    img[y1:y2, x1:x2] = 0
    return img
#DRIFT BY SEVERITY
def apply_drift(img, level):
    if level == "light":
        img = adjust_brightness(img, 0.9)

    elif level == "medium":
        img = adjust_brightness(img, 0.7)
        img = add_gaussian_noise(img, 15)

    elif level == "heavy":
        img = adjust_brightness(img, 0.5)
        img = motion_blur(img, 9)
        img = partial_occlusion(img)

    return img
#MAIN EXECUTION LOOP
def main():
    image_files = os.listdir(SRC_IMAGES)

    for out_dir, drift_level in STREAMS.items():
        os.makedirs(out_dir, exist_ok=True)

        print(f"Creating {out_dir} with {drift_level} drift")

        for img_name in tqdm(image_files):
            src_path = os.path.join(SRC_IMAGES, img_name)
            dst_path = os.path.join(out_dir, img_name)

            img = cv2.imread(src_path)
            if img is None:
                continue

            drifted = apply_drift(img, drift_level)
            cv2.imwrite(dst_path, drifted)

    print("Drift simulation complete.")

if __name__ == "__main__":
    main()
