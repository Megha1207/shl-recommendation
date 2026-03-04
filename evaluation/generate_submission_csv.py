# evaluation/generate_submission_csv.py

import pandas as pd

from retrieval.hybrid import hybrid_search
from retrieval.rerank import balance_and_rerank
from llm.query_understanding import understand_query


TOP_K = 10


def normalize_url(url):
    return url.replace("/solutions", "").rstrip("/")


def generate_submission_csv():

    print("\nLoading test dataset...\n")

    df = pd.read_excel(
        "Gen_AI Dataset.xlsx",
        sheet_name="Test-Set"
    )

    rows = []

    for _, row in df.iterrows():

        query = str(row["Query"])

        print("Processing:", query[:70])

        # Query understanding
        structured = understand_query(query)
        expanded_query = structured.get("expanded_query", query)

        # Hybrid retrieval
        candidates = hybrid_search(expanded_query, top_k=50)

        # Reranking
        try:
            final = balance_and_rerank(
                candidates,
                query=query,
                top_k=TOP_K
            )
        except TypeError:
            # fallback if reranker signature differs
            final = balance_and_rerank(candidates)[:TOP_K]

        urls = [normalize_url(r["url"]) for r in final]

        rows.append({
            "query": query,
            "predictions": ",".join(urls)
        })

    output_file = "evaluation/submission_predictions.csv"

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False
    )

    print("\nSubmission CSV saved →", output_file)


if __name__ == "__main__":
    generate_submission_csv()