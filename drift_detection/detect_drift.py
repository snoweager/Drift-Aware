import numpy as np
from .image_stats import compute_dataset_stats
from .embedding_drift import compute_embedding_drift

# Thresholds (tunable, explainable)
IMAGE_STAT_THRESHOLD = 0.15
EMBEDDING_THRESHOLD = 0.25
CONF_ENTROPY_THRESHOLD = 1.2

def detect_drift(
    baseline_stats,
    current_stats,
    baseline_embeddings,
    current_embeddings,
    confidence_entropy
):
    # Image stats drift
    stat_drift = np.mean(np.abs(
        np.mean(current_stats, axis=0) -
        np.mean(baseline_stats, axis=0)
    ))

    # Embedding drift
    emb_drift = compute_embedding_drift(
        baseline_embeddings,
        current_embeddings
    )

    drift_detected = (
        stat_drift > IMAGE_STAT_THRESHOLD or
        emb_drift > EMBEDDING_THRESHOLD or
        confidence_entropy > CONF_ENTROPY_THRESHOLD
    )

    return {
        "stat_drift": stat_drift,
        "embedding_drift": emb_drift,
        "confidence_entropy": confidence_entropy,
        "drift_detected": drift_detected
    }
