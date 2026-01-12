import os
import shutil

IMAGE_DIR = "data/replay_buffer/images"
LABEL_SRC_DIRS = [
    "data/valid/labels",
    "data/train/labels"
]
LABEL_DST = "data/replay_buffer/labels"

os.makedirs(LABEL_DST, exist_ok=True)

missing = 0

for img in os.listdir(IMAGE_DIR):
    label = img.replace(".jpg", ".txt")

    found = False
    for src in LABEL_SRC_DIRS:
        src_label = os.path.join(src, label)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(LABEL_DST, label))
            found = True
            break

    if not found:
        missing += 1

print(f"Labels copied. Missing labels: {missing}")
