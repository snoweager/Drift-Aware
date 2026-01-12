import numpy as np

def confidence_statistics(predictions):
    confidences = []
    entropies = []

    for pred in predictions:
        probs = np.array(pred)
        probs = probs / probs.sum()

        confidences.append(np.max(probs))
        entropies.append(-np.sum(probs * np.log(probs + 1e-6)))

    return np.mean(confidences), np.mean(entropies)
