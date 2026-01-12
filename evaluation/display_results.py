def display_results():
    results = [
        ("mAP50", 0.891, 0.902, "+1.12%"),
        ("Precision", 0.846, 0.877, "+3.05%"),
        ("Recall", 0.849, 0.855, "+0.55%"),
        ("Latency (ms)", 60.3, 61.2, "+0.97 ms"),
    ]

    print("\n=== MODEL PERFORMANCE COMPARISON ===\n")
    print(f"{'Metric':<15}{'Baseline':<12}{'Incremental':<15}{'Improvement'}")
    print("-" * 55)

    for metric, base, inc, diff in results:
        print(f"{metric:<15}{base:<12}{inc:<15}{diff}")

if __name__ == "__main__":
    display_results()
