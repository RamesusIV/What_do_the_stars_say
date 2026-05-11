import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load all assets once when the module is imported
encoder = SentenceTransformer("models/sentence_transformer")
index = faiss.read_index("models/faiss_index.bin")

with open("models/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


def predict_story(query: str) -> str:
    """
    Returns the story most relevant to the user's question.
    """
    # Create embedding for the query
    embedding = encoder.encode([query])
    embedding = np.array(embedding).astype("float32")

    # Search nearest neighbor
    distances, indices = index.search(embedding, k=1)

    best_index = indices[0][0]
    best_match = metadata[best_index]

    # Handle multiple metadata formats
    if isinstance(best_match, dict):
        if "story" in best_match:
            return best_match["story"]
        elif "answer" in best_match:
            return best_match["answer"]
        elif "response" in best_match:
            return best_match["response"]
        else:
            return str(best_match)

    return str(best_match)