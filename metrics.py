def recall_at_k(recommended_urls, relevant_urls, k=10):
    """
    recommended_urls: list of URLs returned by model (ranked)
    relevant_urls: set of ground-truth relevant URLs
    """
    recommended_k = recommended_urls[:k]

    hits = len(set(recommended_k) & relevant_urls)
    total_relevant = len(relevant_urls)

    if total_relevant == 0:
        return 0.0

    return hits / total_relevant
