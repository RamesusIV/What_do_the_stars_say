import pandas as pd
import json

train = pd.read_csv("data/training_dataset.csv")
prod = pd.read_csv("data/production_queries.csv")

# Automatically use the first column in production_queries.csv
prod_column = prod.columns[0]

train_avg = train["question"].astype(str).str.len().mean()
prod_avg = prod[prod_column].astype(str).str.len().mean()

drift = abs(train_avg - prod_avg) > 10

report = {
    "production_column_used": prod_column,
    "training_avg_query_length": float(train_avg),
    "production_avg_query_length": float(prod_avg),
    "difference": float(abs(train_avg - prod_avg)),
    "data_drift_detected": bool(drift)
}

with open("data_drift_report.json", "w") as f:
    json.dump(report, f, indent=4)

print(json.dumps(report, indent=4))