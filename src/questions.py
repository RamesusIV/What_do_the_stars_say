from transformers import pipeline
import pandas as pd

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

df = pd.read_csv("data/constellation_mythology_unique_88.csv")
rows = []

for _, row in df.iterrows():
    name = row["constellation_name"]
    story = row["story"]

    prompt = f"""
Generate 20 different questions users might ask about the constellation {name}.
Story: {story}
Return one question per line.
"""

    result = generator(prompt, max_new_tokens=300)[0]["generated_text"]

    questions = [
        q.strip("- ").strip()
        for q in result.splitlines()
        if q.strip()
    ]

    for question in questions:
        rows.append({
            "question": question,
            "constellation_name": name,
            "story": story,
            "brightest_star": row["brightest_star"],
            "stars_forming_constellation": row["stars_forming_constellation"]
        })

pd.DataFrame(rows).to_csv("data/training_dataset.csv", index=False)