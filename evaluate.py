import pandas as pd
from metrics import recall_at_k
from model_utils import recommend_assessments


def main():
    # Load human-labeled ground truth
    labels = pd.read_csv("data/labeled_data.csv")

    recall_scores = []

    for query_id, group in labels.groupby("query_id"):
        query_text = group["query"].iloc[0]

        # Ground truth relevant URLs
        relevant_urls = set(
            group[group["relevant"] == 1]["assessment_url"]
        )

        # Run recommender (same logic as API)
        recommendations = recommend_assessments(query_text, top_k=10)

        # Extract URLs from model output
        recommended_urls = [
            r["assessment_url"] for r in recommendations
        ]

        # Compute Recall@10
        recall = recall_at_k(
            recommended_urls,
            relevant_urls,
            k=10
        )

        recall_scores.append(recall)

        print(f"{query_id} → Recall@10 = {recall:.2f}")

    # Mean Recall@10 across all queries
    mean_recall = sum(recall_scores) / len(recall_scores)

    print(f"\n📊 Mean Recall@10: {mean_recall:.3f}")


if __name__ == "__main__":
    main()
