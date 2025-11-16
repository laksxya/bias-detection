from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import spacy
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import argparse
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# -------------------- DEVICE SELECTION --------------------
SCORING_DEVICE = os.environ.get("SCORING_DEVICE", "auto").lower()
if SCORING_DEVICE == "cuda" and not torch.cuda.is_available():
    SCORING_DEVICE = "cpu"
if SCORING_DEVICE not in ("cpu", "cuda"):
    SCORING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INDEX = 0 if SCORING_DEVICE == "cuda" else -1

# -------------------- CONFIG --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENTITY_DIR = os.path.join(BASE_DIR, "entity_mappings")

ENTITY_FILES = {
    "CASTE": ("caste_list.csv", "Caste Name", "Wikipedia Summary"),
    "GENDER": ("gender_list.csv", "Gender Name", "Wikipedia Content"),
    "POLITICAL_PARTY": ("political_parties_list.csv", "Party Name", "Wikipedia Content"),
    "REGION": ("regions_list.csv", "Region Name", "Wikipedia Content"),
    "RELIGION": ("religion_list.csv", "Religion Name", "Wikipedia Content"),
}

# -------------------- HELPERS --------------------


def get_st_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=SCORING_DEVICE)


def load_articles(json_path: str) -> pd.DataFrame:
    if not os.path.isfile(json_path):
        print(f"[WARN] Articles JSON not found: {json_path}")
        return pd.DataFrame()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        print("[WARN] Unexpected JSON structure (expected list).")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "text" not in df.columns and "content" in df.columns:
        df = df.rename(columns={"content": "text"})
    elif "text" not in df.columns:
        df["text"] = ""
    for col in ["url", "title"]:
        if col not in df.columns:
            df[col] = ""
    return df


def load_entity_descriptions() -> dict:
    json_path = os.path.join(ENTITY_DIR, "entity_descriptions.json")
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            desc = json.load(f)
        return {k: v for k, v in desc.items() if isinstance(v, str) and v.strip()}

    all_rows = []
    for label, (fname, name_col, desc_col) in ENTITY_FILES.items():
        fpath = os.path.join(ENTITY_DIR, fname)
        if not os.path.isfile(fpath):
            print(f"[WARN] Missing entity file: {fpath}")
            continue
        df_temp = pd.read_csv(fpath)
        if name_col not in df_temp.columns or desc_col not in df_temp.columns:
            print(f"[WARN] Columns missing in {fname}: {name_col}, {desc_col}")
            continue
        df_temp = df_temp[[name_col, desc_col]].dropna()
        df_temp = df_temp.rename(
            columns={name_col: "entity_name", desc_col: "description"})
        all_rows.append(df_temp)

    if not all_rows:
        return {}
    df_all = pd.concat(all_rows).drop_duplicates(subset=["entity_name"])
    entity_descriptions = dict(
        zip(df_all["entity_name"], df_all["description"]))
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entity_descriptions, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return entity_descriptions


def load_description_embeddings(entity_descriptions: dict, embedding_model: SentenceTransformer) -> dict:
    emb_json = os.path.join(ENTITY_DIR, "description_embeddings.json")
    loaded = {}
    target_device = embedding_model._target_device
    if os.path.isfile(emb_json):
        try:
            with open(emb_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, arr in raw.items():
                if isinstance(arr, list):
                    loaded[k] = torch.tensor(arr, device=target_device)
        except Exception:
            loaded = {}

    missing = [k for k in entity_descriptions.keys() if k not in loaded]
    if missing:
        for name in missing:
            emb = embedding_model.encode(
                entity_descriptions[name], convert_to_tensor=True)
            loaded[name] = emb.to(target_device)
        try:
            serializable = {k: v.cpu().tolist() for k, v in loaded.items()}
            with open(emb_json, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception:
            pass
    return loaded


def get_finbert_sentiment_score(result_list):
    positive_score = 0.0
    negative_score = 0.0
    for result in result_list:
        label = result.get("label", "").lower()
        if label == "positive":
            positive_score = result.get("score", 0.0)
        elif label == "negative":
            negative_score = result.get("score", 0.0)
    return positive_score - negative_score

# -------------------- CORE ANALYSIS --------------------


def build_spacy_with_entity_ruler(entity_descriptions: dict):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[WARN] SpaCy model 'en_core_web_sm' not found. Using blank 'en'.")
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    if "ner" in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "CUSTOM", "pattern": name}
                for name in entity_descriptions.keys()]
    ruler.add_patterns(patterns)
    return nlp


def analyze_text(
    text: str,
    nlp,
    sentiment_pipeline,
    embedding_model: SentenceTransformer,
    entity_descriptions: dict,
    description_embeddings: dict,
    sentence_cache: dict,
):
    if not isinstance(text, str) or not text.strip():
        return {}, {}
    doc = nlp(text)
    entity_scores = defaultdict(list)
    unique_pairs = set()

    for ent in doc.ents:
        if ent.text in entity_descriptions:
            unique_pairs.add((ent.text, ent.label_, ent.sent.text))

    for entity_text, entity_label, sentence in unique_pairs:
        if sentence not in sentence_cache:
            result = sentiment_pipeline(sentence)[0]
            sentiment_score = get_finbert_sentiment_score(result)
            sent_emb = embedding_model.encode(sentence, convert_to_tensor=True)
            sentence_cache[sentence] = (sentiment_score, sent_emb)
        else:
            sentiment_score, sent_emb = sentence_cache[sentence]

        desc_emb = description_embeddings[entity_text]
        if sent_emb.device != desc_emb.device:
            sent_emb = sent_emb.to(desc_emb.device)
        relevance = util.pytorch_cos_sim(sent_emb, desc_emb).item()

        entity_key = f"{entity_text} ({entity_label})"
        entity_scores[entity_key].append(
            {"sentence": sentence, "sentiment": sentiment_score, "relevance": relevance}
        )

    aggregated = {}
    for entity, items in entity_scores.items():
        if not items:
            continue
        weighted_sum = sum(i["sentiment"] * i["relevance"] for i in items)
        weight_total = sum(i["relevance"] for i in items)
        final_score = weighted_sum / weight_total if weight_total else 0.0
        avg_sent = float(np.mean([i["sentiment"] for i in items]))
        avg_rel = float(np.mean([i["relevance"] for i in items]))
        aggregated[entity] = {
            "final_bias_score": final_score,
            "avg_sentiment": avg_sent,
            "avg_relevance": avg_rel,
        }
    return aggregated, dict(entity_scores)


def run_bias_analysis(articles_df: pd.DataFrame) -> pd.DataFrame:
    entity_descriptions = load_entity_descriptions()
    if not entity_descriptions:
        print("[ERROR] No entity descriptions available. Aborting analysis.")
        return pd.DataFrame()

    print(f"[INFO] Loaded {len(entity_descriptions)} entity descriptions.")
    print("[INFO] Loading models...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        return_all_scores=True,
        device=DEVICE_INDEX
    )
    embedding_model = get_st_model()
    description_embeddings = load_description_embeddings(
        entity_descriptions, embedding_model)
    print(
        f"[INFO] Loaded {len(description_embeddings)} description embeddings on {SCORING_DEVICE}.")

    nlp = build_spacy_with_entity_ruler(entity_descriptions)
    print("[INFO] SpaCy pipeline ready.")

    out_df = articles_df.copy()
    if "text" not in out_df.columns:
        out_df["text"] = ""

    sentence_cache = {}
    aggregated_list = []
    detailed_list = []

    print("[INFO] Starting per-article analysis...")
    for _, row in out_df.iterrows():
        agg, detailed = analyze_text(
            row.get("text", ""),
            nlp,
            sentiment_pipeline,
            embedding_model,
            entity_descriptions,
            description_embeddings,
            sentence_cache,
        )
        aggregated_list.append(agg)
        detailed_list.append(detailed)

    out_df["aggregated_bias_scores"] = aggregated_list
    out_df["detailed_scores"] = detailed_list
    return out_df


def assign_bias_labels(row):
    labeled = {}
    for entity, scores in row.get("aggregated_bias_scores", {}).items():
        s = scores["final_bias_score"]
        label = "Neutral"
        if s > 0.05:
            label = "Slightly Positive"
        elif s < -0.05:
            label = "Slightly Negative"
        if s > 0.2:
            label = "Positive Bias"
        elif s < -0.2:
            label = "Negative Bias"
        labeled[entity] = {"score": s, "label": label}
    return labeled

# -------------------- MAIN --------------------


def main():
    parser = argparse.ArgumentParser(description="Run weighted bias scoring.")
    parser.add_argument(
        "--articles-json",
        default=os.path.join(BASE_DIR, "data", "links.cleaned.json"),
        help="Path to scraped articles JSON."
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(BASE_DIR, "Scoring",
                             "final_analyzed_articles_weighted.csv"),
        help="Destination CSV path."
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print per-article bias summary to console."
    )
    args = parser.parse_args()

    df_articles = load_articles(args.articles_json)
    if df_articles.empty:
        print("[ERROR] No articles to analyze.")
        return

    analyzed_df = run_bias_analysis(df_articles)
    if analyzed_df.empty:
        print("[ERROR] Analysis produced empty DataFrame.")
        return

    if args.print_summary:
        for _, r in analyzed_df.iterrows():
            print(f"\n--- ARTICLE: {r.get('title','(no title)')} ---")
            print(f"URL: {r.get('url','')}")
            scores = r.get("aggregated_bias_scores", {})
            if not scores:
                print("  No custom entities found.")
                continue
            for ent, vals in scores.items():
                bias = vals["final_bias_score"]
                label = "Neutral"
                if bias > 0.05:
                    label = "Slightly Positive"
                elif bias < -0.05:
                    label = "Slightly Negative"
                if bias > 0.2:
                    label = "Positive Bias"
                elif bias < -0.2:
                    label = "Negative Bias"
                print(f"  - {ent}: {bias:+.3f} ({label}) "
                      f"[AvgSent={vals['avg_sentiment']:.3f}, AvgRel={vals['avg_relevance']:.3f}]")

    analyzed_df["bias_labels"] = analyzed_df.apply(assign_bias_labels, axis=1)
    for col in ["aggregated_bias_scores", "detailed_scores", "bias_labels"]:
        analyzed_df[col] = analyzed_df[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False))

    out_path = os.environ.get("SCORING_OUTPUT", args.output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    analyzed_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] Saved weighted bias analysis to: {out_path}")


if __name__ == "__main__":
    main()
