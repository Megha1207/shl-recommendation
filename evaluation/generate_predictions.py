# evaluation/generate_predictions.py

import json
import pandas as pd

from retrieval.hybrid import hybrid_search
from retrieval.rerank import balance_and_rerank
from llm.query_understanding import understand_query

K = 10


def normalize_url(url):
    return url.replace("/solutions", "").rstrip("/")


def generate_predictions():

    df = pd.read_excel(
        "Gen_AI Dataset.xlsx",
        sheet_name="Test-Set"
    )

    predictions = []

    print("\nGenerating predictions...\n")

    for _, row in df.iterrows():

        query = row["Query"]

        structured = understand_query(query)
        expanded_query = structured.get("expanded_query", query)

        candidates = hybrid_search(expanded_query, top_k=50)

        final = balance_and_rerank(
            candidates,
            query=query,
            top_k=K
            )

        urls = [normalize_url(r["url"]) for r in final]

        predictions.append({
            "query": query,
            "recommended_assessments": urls
        })

        print("Query:", query[:70])
        print("Returned:", len(urls))

    with open("evaluation/test_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("\nPredictions saved → evaluation/test_predictions.json")


if __name__ == "__main__":
    generate_predictions()