def compare_models(old_metrics, new_metrics):
    comparison = {}

    for key in old_metrics:
        comparison[key] = {
            "old": old_metrics[key],
            "new": new_metrics[key],
            "delta": new_metrics[key] - old_metrics[key]
        }

    decision = (
        new_metrics["mAP50"] >= old_metrics["mAP50"] and
        new_metrics["latency_sec"] <= old_metrics["latency_sec"] * 1.1
    )

    comparison["DEPLOY_NEW_MODEL"] = decision
    return comparison
