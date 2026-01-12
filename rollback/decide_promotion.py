import shutil

def decide_and_promote(metrics, model_v1, model_v2, stable_model):
    improve_map = metrics["mAP50"]["new"] >= metrics["mAP50"]["old"]
    acceptable_latency = metrics["latency_sec"]["delta"] < 2.0

    if improve_map and acceptable_latency:
        print("✅ New model approved — promoting to stable.")
        shutil.copy(model_v2, stable_model)
        decision = "PROMOTED"
    else:
        print("❌ New model rejected — rolling back.")
        shutil.copy(model_v1, stable_model)
        decision = "ROLLED_BACK"

    return decision
