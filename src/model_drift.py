import json
import pandas as pd

prod = pd.read_csv("data/production_queries.csv")

# Simulated performance metric
baseline_accuracy = 0.95
production_accuracy = 0.82

drift = production_accuracy < baseline_accuracy - 0.05

report = {
    "baseline_accuracy": baseline_accuracy,
    "production_accuracy": production_accuracy,
    "difference": baseline_accuracy - production_accuracy,
    "model_drift_detected": drift
}

with open("model_drift_report.json", "w") as f:
    json.dump(report, f, indent=4)

print(json.dumps(report, indent=4))