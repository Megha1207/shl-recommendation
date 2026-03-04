import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import os
import re

with open("data/assessments.json", "r") as f:
    assessments = json.load(f)

print(f"Loaded {len(assessments)} assessments")

if len(assessments) < 377:
    print(f"WARNING: Only {len(assessments)} assessments — expected 377+")

TYPE_TO_ROLES = {
    "Personality & Behaviour": "personality culture fit leadership executive consultant manager behaviour values",
    "Ability & Aptitude": "cognitive reasoning aptitude graduate entry level analytical problem solving",
    "Knowledge & Skills": "technical skills programming coding developer engineer knowledge test",
    "Competencies": "leadership management teamwork collaboration interpersonal competencies",
    "Simulations": "coding programming technical simulation developer hands-on",
    "Biodata & Situational Judgement": "situational judgement biodata graduate entry level sales customer",
    "Development & 360": "development 360 feedback leadership growth",
    "Assessment Exercises": "exercise assessment centre leadership group activity",
}

NOISE_PHRASES = re.compile(
    r"Outdated browser detected.*?Latest browser options",
    re.IGNORECASE | re.DOTALL
)

def tokenize(text: str):
    return re.sub(r"[^\w\s]", "", text.lower()).split()

def clean_description(text: str) -> str:
    if not text:
        return ""
    text = NOISE_PHRASES.sub("", text)
    text = re.sub(r"^description\s*", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()

def build_document(a: dict) -> str:

    parts = []

    name = a.get("name", "")
    if name:
        parts.append(f"Assessment Name: {name}. {name}.")

    description = clean_description(a.get("description", ""))
    if description:
        parts.append(f"Description: {description}")

    test_types = a.get("test_types", [])

    if test_types:
        parts.append("Test Type: " + ", ".join(test_types))

    duration = a.get("duration")
    if duration:
        parts.append(f"Duration: {duration} minutes")

    if a.get("remote_support") == "Yes":
        parts.append("Remote testing supported")

    if a.get("adaptive_support") == "Yes":
        parts.append("Adaptive testing supported")

    # --------------------------------------------------
    # ROLE AUGMENTATION (this is the key improvement)
    # --------------------------------------------------

    role_context = []

    if "Knowledge & Skills" in test_types:
        role_context.append(
            "Used for hiring developers engineers programmers software testers data analysts technical roles"
        )

    if "Ability & Aptitude" in test_types:
        role_context.append(
            "Used for graduate hiring cognitive ability reasoning analytical roles entry level candidates"
        )

    if "Personality & Behaviour" in test_types:
        role_context.append(
            "Used for leadership management executive consultant culture fit personality evaluation"
        )

    if "Competencies" in test_types:
        role_context.append(
            "Used for teamwork collaboration leadership communication interpersonal skills"
        )

    if "Simulations" in test_types:
        role_context.append(
            "Hands-on coding programming technical simulation developer engineer evaluation"
        )

    if "Biodata & Situational Judgement" in test_types:
        role_context.append(
            "Situational judgement test used for sales customer service entry level hiring"
        )

    if "Assessment Exercises" in test_types:
        role_context.append(
            "Assessment centre exercise leadership group activity management simulation"
        )

    if "Development & 360" in test_types:
        role_context.append(
            "Leadership development feedback executive coaching management growth"
        )

    if role_context:
        parts.append("Hiring Context: " + " ".join(role_context))

    parts.append(
        "SHL assessment hiring recruitment talent evaluation psychometric testing"
    )

    return " | ".join(parts)


print("Building document corpus...")
documents = [build_document(a) for a in assessments]

print(f"\nSample document:\n{documents[0][:500]}\n")

print("Generating embeddings...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    normalize_embeddings=True,
    batch_size=32
)

embeddings = np.array(embeddings).astype("float32")
print(f"Embedding shape: {embeddings.shape}")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"FAISS index size: {index.ntotal}")

os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss.index")

print("Building BM25 index...")
tokenized_docs = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

with open("embeddings/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

with open("embeddings/metadata.pkl", "wb") as f:
    pickle.dump(assessments, f)

with open("embeddings/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"\n{'='*40}")
print(f"Assessments indexed: {len(assessments)}")
print(f"With duration:       {sum(1 for a in assessments if a.get('duration'))}")
print(f"With description:    {sum(1 for a in assessments if a.get('description'))}")
print(f"With test types:     {sum(1 for a in assessments if a.get('test_types'))}")
print(f"{'='*40}")