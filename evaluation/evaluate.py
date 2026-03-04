import json
import re
import numpy as np
from retrieval.hybrid import (
    hybrid_search,
    hybrid_search_multi,
    index,
    bm25,
    metadata,
    model
)
from llm.query_understanding import understand_query

K = 10
CANDIDATE_POOL = 200


def normalize_url(url):
    return url.replace("shl.com/solutions/products", "shl.com/products").rstrip("/")


def recall_at_k(relevant, predicted, k):
    predicted_k = predicted[:k]
    hits = len(set(relevant) & set(predicted_k))
    return hits / len(relevant) if relevant else 0


def tokenize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def evaluate():
    with open("evaluation/train.json") as f:
        train_data = json.load(f)

    dense_total = 0
    bm25_total = 0
    hybrid_total = 0
    final_total = 0

    print("\nRunning evaluation...\n")

    for item in train_data:
        query = item["query"]
        relevant_urls = [normalize_url(u) for u in item["relevant_urls"]]

        # -----------------------------
        # Query Expansion
        # -----------------------------
        structured = understand_query(query)
        expanded_query = structured.get("expanded_query", query)

        type_query_parts = (
            structured.get("inferred_test_types", []) +
            structured.get("required_skills", []) +
            structured.get("soft_skills", []) +
            ([structured["job_level"]] if structured.get("job_level") else [])
        )
        type_query = " ".join(type_query_parts)

        queries_to_run = [expanded_query]
        if type_query.strip() and type_query.strip() != expanded_query.strip():
            queries_to_run.append(type_query)

        # -----------------------------
        # DENSE ONLY
        # -----------------------------
        query_embedding = model.encode(
            expanded_query,
            normalize_embeddings=True
        )
        query_embedding = np.array([query_embedding]).astype("float32")

        _, dense_indices = index.search(query_embedding, CANDIDATE_POOL)

        dense_predicted = [
            normalize_url(metadata[i]["url"])
            for i in dense_indices[0][:K]
        ]

        dense_recall = recall_at_k(relevant_urls, dense_predicted, K)
        dense_total += dense_recall

        # -----------------------------
        # BM25 ONLY
        # -----------------------------
        tokenized_query = tokenize(expanded_query)
        sparse_scores = bm25.get_scores(tokenized_query)
        sparse_ids = np.argsort(sparse_scores)[::-1][:K]

        bm25_predicted = [
            normalize_url(metadata[i]["url"])
            for i in sparse_ids
        ]

        bm25_recall = recall_at_k(relevant_urls, bm25_predicted, K)
        bm25_total += bm25_recall

        # -----------------------------
        # HYBRID + FINAL
        # -----------------------------
        hybrid_results = hybrid_search_multi(queries_to_run, top_k=K)
        hybrid_predicted = [normalize_url(r["url"]) for r in hybrid_results]
        hybrid_recall = recall_at_k(relevant_urls, hybrid_predicted, K)
        hybrid_total += hybrid_recall

        # Final = Hybrid
        final_total += hybrid_recall

        # -----------------------------
        # Per-query miss analysis
        # -----------------------------
        misses = set(relevant_urls) - set(hybrid_predicted)
        if misses:
            top50 = hybrid_search_multi(queries_to_run, top_k=50)
            top50_urls = [normalize_url(r["url"]) for r in top50]
            print(f"Q: {query[:70]}")
            print(f"   Recall: {hybrid_recall:.2f}  ({len(relevant_urls) - len(misses)}/{len(relevant_urls)})")
            for m in misses:
                rank = top50_urls.index(m) + 1 if m in top50_urls else -1
                name = next(
                    (r["name"] for r in top50 if normalize_url(r["url"]) == m),
                    "NOT IN TOP-50"
                )
                rank_str = f"rank={rank:3d}" if rank > 0 else "outside top-50"
                print(f"   MISS {rank_str} | {name}")
            print()

    n = len(train_data)

    print("Evaluation Results")
    print("-------------------")
    print(f"Queries evaluated: {n}")
    print(f"Dense Mean Recall@{K}:   {dense_total / n:.4f}")
    print(f"BM25 Mean Recall@{K}:    {bm25_total / n:.4f}")
    print(f"Hybrid Mean Recall@{K}:  {hybrid_total / n:.4f}")
    print(f"Final Mean Recall@{K}:   {final_total / n:.4f}")


if __name__ == "__main__":
    evaluate()