import json
from pathlib import Path

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os  # <-- ADD

os.environ["TRANSFORMERS_NO_TF"] = "1"
# --- Config ---
BASE_DIR = Path(__file__).parent
OUTPUT_DESCRIPTIONS = BASE_DIR / "entity_descriptions.json"
OUTPUT_EMBEDDINGS = BASE_DIR / "description_embeddings.json"

ENTITY_FILES_CONFIG = {
    "CASTE": {"file": "caste_list.csv", "name_col": "Caste Name", "desc_col": "Wikipedia Summary"},
    "GENDER": {"file": "gender_list.csv", "name_col": "Gender Name", "desc_col": "Wikipedia Content"},
    "POLITICAL_PARTY": {"file": "political_parties_list.csv", "name_col": "Party Name", "desc_col": "Wikipedia Content"},
    "REGION": {"file": "regions_list.csv", "name_col": "Region Name", "desc_col": "Wikipedia Content"},
    "RELIGION": {"file": "religion_list.csv", "name_col": "Religion Name", "desc_col": "Wikipedia Content"},
}

MIN_DESC_CHARS = 30  # skip ultra-short descriptions


def load_entity_tables():
    dfs = []
    for label, cfg in ENTITY_FILES_CONFIG.items():
        path = BASE_DIR / cfg["file"]
        if not path.exists():
            print(f"Missing file: {path}")
            continue
        df = pd.read_csv(path)
        # Standardize columns
        name_col = cfg["name_col"]
        desc_col = cfg["desc_col"]
        if name_col not in df.columns or desc_col not in df.columns:
            print(f"Column mismatch in {path.name}")
            continue
        df = df[[name_col, desc_col]].rename(
            columns={name_col: "entity_name", desc_col: "description"})
        df["label"] = label
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["entity_name", "description", "label"])
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["entity_name", "description"])
    merged["entity_name"] = merged["entity_name"].str.strip()
    merged["description"] = merged["description"].str.strip()
    merged = merged[merged["description"].str.len() >= MIN_DESC_CHARS]
    merged = merged.drop_duplicates(subset=["entity_name"])
    return merged


def build_spacy_pipeline(entity_df: pd.DataFrame):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "SpaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []
    for _, row in entity_df.iterrows():
        patterns.append({"label": row["label"], "pattern": row["entity_name"]})
    ruler.add_patterns(patterns)
    print(f"Loaded {len(patterns)} custom entity patterns.")
    return nlp


def compute_embeddings(entity_descriptions: dict):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = {}
    for name, desc in entity_descriptions.items():
        vec = model.encode(desc)
        embeddings[name] = vec.tolist()  # JSON serializable
    print(f"Computed embeddings for {len(embeddings)} entities.")
    return embeddings


def init_sentiment_pipeline():
    try:
        sent_pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True,
            framework="pt"  # <-- Force PyTorch
        )
        print("FinBERT sentiment pipeline ready (PyTorch).")
        return sent_pipe
    except Exception as e:
        print(f"Failed to init sentiment pipeline: {e}")
        return None


def main():
    entity_df = load_entity_tables()
    if entity_df.empty:
        print("No entity data loaded. Exiting.")
        return

    # Build SpaCy pipeline
    nlp = build_spacy_pipeline(entity_df)

    # Build description dict
    entity_descriptions = dict(
        zip(entity_df["entity_name"], entity_df["description"]))
    OUTPUT_DESCRIPTIONS.write_text(json.dumps(
        entity_descriptions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote entity descriptions to {OUTPUT_DESCRIPTIONS}")

    # Embeddings
    description_embeddings = compute_embeddings(entity_descriptions)
    OUTPUT_EMBEDDINGS.write_text(json.dumps(
        description_embeddings), encoding="utf-8")
    print(f"Wrote description embeddings to {OUTPUT_EMBEDDINGS}")

    # Sentiment pipeline (kept in memory for downstream use)
    sentiment_pipe = init_sentiment_pipeline()

    print("Pipeline objects available: nlp, sentiment_pipe, entity_descriptions, description_embeddings")
    print("Pipeline order:", nlp.pipe_names)


if __name__ == "__main__":
    main()
