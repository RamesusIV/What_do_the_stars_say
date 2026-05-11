# src/train.py

import os
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --------------------------------------------------
# 1. Create output directory
# --------------------------------------------------
os.makedirs("models", exist_ok=True)

# --------------------------------------------------
# 2. Load generated training data
# --------------------------------------------------
df = pd.read_csv("data/training_dataset.csv")

# We embed only the questions
questions = df["question"].astype(str).tolist()

# --------------------------------------------------
# 3. Load pretrained transformer embedding model
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 4. Convert questions into embeddings
# --------------------------------------------------
embeddings = model.encode(
    questions,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Convert to float32 (required by FAISS)
embeddings = embeddings.astype("float32")

# --------------------------------------------------
# 5. Build FAISS index
# --------------------------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --------------------------------------------------
# 6. Save FAISS index
# --------------------------------------------------
faiss.write_index(index, "models/faiss_index.bin")

# --------------------------------------------------
# 7. Save metadata
# --------------------------------------------------
# Keep the columns needed to answer the question
metadata = df[
    [
        "constellation_name",
        "story",
        "brightest_star",
        "stars_forming_constellation",
    ]
].to_dict(orient="records")

with open("models/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# --------------------------------------------------
# 8. Save embedding model locally (optional but useful)
# --------------------------------------------------
model.save("models/sentence_transformer")

print("Training completed successfully.")
print(f"Indexed {len(questions)} questions.")
print("Saved:")
print(" - models/faiss_index.bin")
print(" - models/metadata.pkl")
print(" - models/sentence_transformer/")