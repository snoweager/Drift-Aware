from ultralytics import YOLO
import json
import os

MODEL_PATH = "models/baseline_yolo7/weights/best.pt"
DATA_YAML = "data/data.yaml"

def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(data=DATA_YAML)

    baseline_metrics = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr)
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=4)

    print("Baseline evaluation complete:")
    print(baseline_metrics)

if __name__ == "__main__":
    main()
