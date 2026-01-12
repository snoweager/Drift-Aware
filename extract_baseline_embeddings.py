import os
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/baseline_yolo7/weights/best.pt"
VAL_IMAGES = "data/valid/images"
OUTPUT_PATH = "metrics/baseline_embeddings.npy"

def extract_features(img_path, model):
    results = model.predict(img_path, verbose=False)
    feats = results[0].boxes.data
    if feats.shape[0] == 0:
        return np.zeros(168)  # replace with feature length
    return np.mean(feats.cpu().numpy(), axis=0)

def main():
    model = YOLO(MODEL_PATH)
    embeddings = []

    for img_name in os.listdir(VAL_IMAGES)[:100]:
        img_path = os.path.join(VAL_IMAGES, img_name)
        emb = extract_features(img_path, model)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)  # shape: (n_images, n_features)
    np.save(OUTPUT_PATH, embeddings)
    print("Saved baseline embeddings:", embeddings.shape)

if __name__ == "__main__":
    main()
