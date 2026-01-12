import time
from ultralytics import YOLO

def evaluate_model(model_path, data_yaml):
    model = YOLO(model_path)

    # Accuracy metrics
    metrics = model.val(data=data_yaml, split="val")

    # Latency measurement
    start = time.time()
    model.predict(source="data/valid/images", imgsz=640, verbose=False)
    latency = time.time() - start

    return {
        "mAP50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "latency_sec": latency
    }
