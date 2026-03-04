import json
import pickle
import re
from llm.query_understanding import understand_query

def normalize_url(url):
    return url.replace("shl.com/solutions/products", "shl.com/products").rstrip("/")

with open("evaluation/train.json") as f:
    train_data = json.load(f)

with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

with open("embeddings/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Queries with 0 or low recall
failing_queries = [
    "KEY RESPONSIBITILES:\nManage the sound-scape of the station through ap",
    "We're looking for a Marketing Manager who can drive Recro's brand posi",
    "Find me 1 hour long assesment for the below job at SHL\nJob Description",
]

# Show expansions for failing queries
print("=== QUERY EXPANSIONS ===\n")
for item in train_data:
    q = item["query"]
    if any(q.startswith(f[:30]) for f in failing_queries):
        s = understand_query(q)
        print(f"Q: {q[:80]}")
        print(f"   expanded: {s['expanded_query'][:200]}")
        print(f"   relevant products:")
        for url in item["relevant_urls"]:
            norm = normalize_url(url)
            # Find in metadata
            for i, m in enumerate(metadata):
                if normalize_url(m["url"]) == norm:
                    print(f"     - {m['name']}")
                    print(f"       doc: {documents[i][:200]}")
                    break
            else:
                print(f"     - NOT FOUND: {norm}")
        print()