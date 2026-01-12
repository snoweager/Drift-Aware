import numpy as np
from drift_detection.image_stats import compute_dataset_stats
from drift_detection.detect_drift import detect_drift

BASELINE_IMAGES = "data/valid/images"
STREAM_IMAGES = "data/stream_day_7/images"

BASELINE_EMB = "metrics/baseline_embeddings.npy"

def main():
    print("Computing baseline image stats...")
    baseline_stats = compute_dataset_stats(BASELINE_IMAGES)

    print("Computing current stream image stats...")
    current_stats = compute_dataset_stats(STREAM_IMAGES)

    print("Loading baseline embeddings...")
    baseline_embeddings = np.load(BASELINE_EMB, allow_pickle=True)

    current_embeddings = baseline_embeddings + np.random.normal(
        0, 0.05, size=baseline_embeddings.shape
    )

    confidence_entropy = 1.35

    result = detect_drift(
        baseline_stats,
        current_stats,
        baseline_embeddings,
        current_embeddings,
        confidence_entropy
    )

    print("\n--- DRIFT REPORT ---")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
