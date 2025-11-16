import os
import json
import re
from typing import List, Dict, Any
import pandas as pd

# Add these imports for sentence similarity
try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(ROOT_DIR, "report")
CLUSTER_DIR = os.path.join(REPORT_DIR, "clusters")

# --- Input Files ---
CLUSTERED_CSV = os.path.join(CLUSTER_DIR, "clustered_articles.csv")
ASSIGN_JSON = os.path.join(CLUSTER_DIR, "cluster_assignments.json")
LINKS_CLEANED = os.path.join(ROOT_DIR, "data", "links.cleaned.json")

# --- Output Files ---
DEDUPE_CLUSTER_JSON = os.path.join(REPORT_DIR, "dedupli_clus.json")
GLOBAL_PROVENANCE_JSON = os.path.join(
    REPORT_DIR, "global_provenance.json")  # <-- NEW
RAW_TXT = os.path.join(REPORT_DIR, "raw.txt")

# --- Config ---
SPLIT_MIN_LEN = int(os.environ.get("DEDUP_SENT_MIN_LEN", "30"))
SIM_THRESHOLD = float(os.environ.get("DEDUP_SIM_THRESHOLD", "0.85"))
DEVICE = "cuda" if _HAS_ST and torch.cuda.is_available() else "cpu"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if len(s.strip()) >= SPLIT_MIN_LEN]


def detect_column(cols, candidates):
    return next((c for c in candidates if c in cols), None)


def _to_cluster_id(raw):
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else -1
    try:
        return int(raw) if isinstance(raw, (int, float)) or (isinstance(raw, str) and raw.isdigit()) else -1
    except (ValueError, TypeError):
        return -1


def load_cluster_articles() -> Dict[int, List[Dict[str, Any]]]:
    # Returns {cluster_id: [ {text, title, url, article_index}, ... ]}
    if os.path.exists(CLUSTERED_CSV):
        df = pd.read_csv(CLUSTERED_CSV)
        cluster_col = detect_column(
            df.columns, ["cluster_id", "viewpoint_cluster", "cluster"])
        text_col = detect_column(df.columns, ["text", "content"])
        if cluster_col and text_col:
            out: Dict[int, List[Dict[str, Any]]] = {}
            for i, row in df.iterrows():
                cid = _to_cluster_id(row[cluster_col])
                txt = row.get(text_col)
                if cid >= 0 and isinstance(txt, str) and len(txt.strip()) >= SPLIT_MIN_LEN:
                    out.setdefault(cid, []).append({
                        "text": txt.strip(),
                        "title": str(row.get("title", "")),
                        "url": str(row.get("url", "")),
                        "article_index": i
                    })
            if out:
                print(
                    f"[INFO] Loaded {sum(len(v) for v in out.values())} articles from CSV.")
                return out
    raise FileNotFoundError(
        "Could not load valid articles from clustered_articles.csv.")


def greedy_dedup_with_provenance(model: SentenceTransformer, sentence_objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not sentence_objs:
        return []

    texts = [obj["sentence"] for obj in sentence_objs]
    embeddings = model.encode(
        texts, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)

    kept_indices = []
    corpus_indices = list(range(len(texts)))

    while corpus_indices:
        main_idx = corpus_indices.pop(0)
        kept_indices.append(main_idx)

        main_embedding = embeddings[main_idx].unsqueeze(0)

        if not corpus_indices:
            break

        # Find duplicates of the current main sentence
        other_indices = torch.tensor(corpus_indices, device=embeddings.device)
        other_embeddings = embeddings[other_indices]

        similarities = util.cos_sim(main_embedding, other_embeddings)[0]

        is_duplicate = similarities > SIM_THRESHOLD
        duplicate_indices_in_others = torch.where(is_duplicate)[0].tolist()

        # Merge sources from duplicates into the main sentence object
        for dup_idx_in_others in sorted(duplicate_indices_in_others, reverse=True):
            original_dup_idx = corpus_indices.pop(dup_idx_in_others)
            # Combine sources
            sentence_objs[main_idx]["sources"].extend(
                sentence_objs[original_dup_idx]["sources"])

    # Final list of kept sentence objects
    final_objs = [sentence_objs[i] for i in kept_indices]

    # Deduplicate the sources within each final object
    for obj in final_objs:
        unique_sources = []
        seen_urls = set()
        for source in obj["sources"]:
            url = source.get("url")
            if url not in seen_urls:
                unique_sources.append(source)
                seen_urls.add(url)
        obj["sources"] = unique_sources

    return final_objs


def main():
    if not _HAS_ST:
        print(
            "[ERROR] `sentence-transformers` is not installed. Cannot perform deduplication.")
        print("Please run `pip install sentence-transformers torch`")
        return

    print("[START] Two-level dedup with provenance tracking.")
    clusters = load_cluster_articles()
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    global_sentence_objs: List[Dict[str, Any]] = []

    # Level 1: Per-cluster deduplication
    for cid, articles in clusters.items():
        cluster_sentence_objs = []
        for art in articles:
            for sent in split_sentences(art["text"]):
                cluster_sentence_objs.append({
                    "sentence": sent,
                    "sources": [{
                        "title": art["title"],
                        "url": art["url"],
                        "article_index": art["article_index"]
                    }]
                })

        print(
            f"[CLUSTER {cid}] Sentences before dedup: {len(cluster_sentence_objs)}")
        kept_in_cluster = greedy_dedup_with_provenance(
            model, cluster_sentence_objs)
        print(f"[CLUSTER {cid}] Sentences after dedup: {len(kept_in_cluster)}")
        global_sentence_objs.extend(kept_in_cluster)

    # Level 2: Global deduplication
    print(f"\n[GLOBAL] Sentences before dedup: {len(global_sentence_objs)}")
    final_kept_objs = greedy_dedup_with_provenance(model, global_sentence_objs)
    print(f"[GLOBAL] Sentences after dedup: {len(final_kept_objs)}")

    # --- Save Outputs ---
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Save the final provenance map
    with open(GLOBAL_PROVENANCE_JSON, "w", encoding="utf-8") as f:
        json.dump({"sentences": final_kept_objs},
                  f, indent=2, ensure_ascii=False)
    print(f"[DONE] Saved provenance map to {GLOBAL_PROVENANCE_JSON}")

    # 2. Save the raw text file for claim extraction
    raw_text = "\n".join([obj["sentence"] for obj in final_kept_objs])
    with open(RAW_TXT, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"[DONE] Saved raw text to {RAW_TXT} (chars: {len(raw_text)})")

    print("[COMPLETE] Deduplication with provenance finished.")


if __name__ == "__main__":
    main()
