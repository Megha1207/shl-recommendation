import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re

print("Loading retrieval components...")

index = faiss.read_index("embeddings/faiss.index")

with open("embeddings/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("embeddings/documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

print("Retriever ready.")


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def rrf_fusion(dense_ids, sparse_ids, k=60):
    scores = {}

    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked


def hybrid_search(query, top_k=10, dense_weight=0.50, use_rrf=False):
    candidate_k = max(top_k, 500)

    # Dense
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype("float32")
    dense_scores, dense_indices = index.search(query_embedding, candidate_k)
    dense_scores = dense_scores[0]
    dense_indices = dense_indices[0]

    # BM25
    tokenized_query = tokenize(query)
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(sparse_scores)[::-1][:candidate_k]
    sparse_values = sparse_scores[sparse_indices]

    if use_rrf:
        ranked_ids = rrf_fusion(dense_indices.tolist(), sparse_indices.tolist())
        return [metadata[i] for i in ranked_ids[:top_k]]

    # Weighted fusion
    dense_scores = (dense_scores - dense_scores.min()) / (
        dense_scores.max() - dense_scores.min() + 1e-8
    )
    if sparse_values.max() > 0:
        sparse_values = sparse_values / sparse_values.max()

    dense_dict = dict(zip(dense_indices, dense_scores))
    sparse_dict = dict(zip(sparse_indices, sparse_values))

    sparse_weight = 1.0 - dense_weight
    all_doc_ids = set(dense_dict) | set(sparse_dict)

    final_scores = {
        doc_id: dense_weight * dense_dict.get(doc_id, 0)
                + sparse_weight * sparse_dict.get(doc_id, 0)
        for doc_id in all_doc_ids
    }

    ranked_ids = sorted(final_scores, key=final_scores.get, reverse=True)
    return [metadata[i] for i in ranked_ids[:top_k]]


def hybrid_search_multi(queries: list, top_k: int = 10):

    scores = {}

    for query in queries:

        results = hybrid_search(query, top_k=50)

        for rank, r in enumerate(results):

            url = r["url"]

            score = 1 / (rank + 1)

            if url not in scores:
                scores[url] = (score, r)
            else:
                scores[url] = (scores[url][0] + score, r)

    ranked = sorted(
        scores.values(),
        key=lambda x: x[0],
        reverse=True
    )

    return [r for _, r in ranked[:top_k]]