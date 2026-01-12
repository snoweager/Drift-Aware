from decide_promotion import decide_and_promote
import json

MODEL_V1 = "models/model_v1.pt"
MODEL_V2 = "models/model_v2.pt"
STABLE_MODEL = "models/stable_model.pt"
METRICS_FILE = "metrics/model_comparison.json"

with open(METRICS_FILE) as f:
    metrics = json.load(f)

decision = decide_and_promote(
    metrics,
    MODEL_V1,
    MODEL_V2,
    STABLE_MODEL
)

print(f"FINAL DECISION: {decision}")
