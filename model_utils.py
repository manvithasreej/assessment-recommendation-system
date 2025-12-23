import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load data once
df = pd.read_csv("data/clean_assessment_catalog.csv")

# Create embeddings once
assessment_embeddings = model.encode(
    df["final_text"].tolist(),
    convert_to_tensor=True
)

TECH_KEYWORDS = ["java", "python", "sql", "developer"]
BEHAV_KEYWORDS = ["teamwork", "communication", "collaboration"]

def detect_query_intent(query):
    q = query.lower()
    return {
        "technical": any(k in q for k in TECH_KEYWORDS),
        "behavioral": any(k in q for k in BEHAV_KEYWORDS)
    }

def recommend_assessments(query, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    similarity_scores = cosine_similarity(
        query_embedding.cpu(),
        assessment_embeddings.cpu()
    )[0]

    df_copy = df.copy()
    df_copy["score"] = similarity_scores
    df_copy = df_copy.sort_values("score", ascending=False)

    intent = detect_query_intent(query)

    final = []

    if intent["technical"]:
        final.extend(
            df_copy[df_copy["test_type"] == "Technical"]
            .head(2)
            .to_dict("records")
        )

    if intent["behavioral"]:
        final.extend(
            df_copy[df_copy["test_type"] == "Behavioral"]
            .head(1)
            .to_dict("records")
        )

    # Fill remaining (remove duplicates by URL)
    seen = set(r["assessment_url"] for r in final)

    for _, row in df_copy.iterrows():
        if row["assessment_url"] not in seen:
            final.append(row.to_dict())
            seen.add(row["assessment_url"])
        if len(final) == top_k:
            break

    return [
        {
            "assessment_name": r["final_text"],
            "assessment_url": r["assessment_url"]
        }
        for r in final
    ]
