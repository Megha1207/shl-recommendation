import pandas as pd
import json


def prepare_train_data(excel_path):
    df = pd.read_excel(excel_path, sheet_name="Train-Set")

    grouped = {}

    for _, row in df.iterrows():
        query = row["Query"]
        url = row["Assessment_url"]

        if query not in grouped:
            grouped[query] = []

        grouped[query].append(url)

    train_data = []

    for query, urls in grouped.items():
        train_data.append({
            "query": query,
            "relevant_urls": urls
        })

    with open("evaluation/train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    print("Train data prepared successfully.")


if __name__ == "__main__":
    prepare_train_data("Gen_AI Dataset.xlsx")