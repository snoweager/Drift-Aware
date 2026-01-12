import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def compute_embedding_drift(baseline_embeddings, current_embeddings):
    baseline_mean = np.mean(baseline_embeddings, axis=0)
    current_mean = np.mean(current_embeddings, axis=0)

    drift_score = cosine_distances(
        baseline_mean.reshape(1, -1),
        current_mean.reshape(1, -1)
    )[0][0]

    return drift_score
