from evaluate_model import evaluate_model
from compare_models import compare_models
from shadow_test import shadow_test
import json
import os

BASELINE_MODEL = "models/baseline_yolo7/weights/best.pt"
NEW_MODEL = "models/incremental_model2/weights/best.pt"
DATA_YAML = "data/replay_buffer/data.yaml"

def main():
    print("Evaluating baseline model...")
    baseline_metrics = evaluate_model(BASELINE_MODEL, DATA_YAML)

    print("Evaluating incremental model...")
    new_metrics = evaluate_model(NEW_MODEL, DATA_YAML)

    print("Comparing models...")
    comparison = compare_models(baseline_metrics, new_metrics)

    print("Running shadow test...")
    drift_images = "data/stream_day_7/images"
    discrepancy_count = shadow_test(BASELINE_MODEL, NEW_MODEL, drift_images)

    comparison["shadow_discrepancies"] = discrepancy_count

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)

    print("\n--- MODEL COMPARISON REPORT ---")
    for k, v in comparison.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
