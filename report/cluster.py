import os
import ast
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------- Config --------
ROOT_DIR = r"c:\Users\ishuv\Desktop\Capstone"
INPUT_CSV = os.path.join(
    ROOT_DIR, "Scoring", "final_analyzed_articles_weighted.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "report", "clusters")

# Clustering params
FIXED_K = 0        # 0 = auto-pick via silhouette; >0 forces k
NORMALIZE = True   # L2-normalize bias vectors before clustering
K_MIN = 2
K_MAX = 6
RANDOM_STATE = 42


def load_final(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        print(f"[ERROR] Input CSV not found: {input_path}")
        return pd.DataFrame()
    df = pd.read_csv(input_path)
    if "aggregated_bias_scores" in df.columns:
        df["aggregated_bias_scores"] = df["aggregated_bias_scores"].apply(
            lambda x: {} if pd.isna(x) or x == "" else ast.literal_eval(x)
        )
    else:
        df["aggregated_bias_scores"] = [{}] * len(df)
    return df


def build_bias_matrix(df_final: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    all_entities = set()
    for scores_dict in df_final["aggregated_bias_scores"]:
        all_entities.update(scores_dict.keys())
    entity_list = sorted(all_entities)

    rows = []
    for _, row in df_final.iterrows():
        vec = []
        scores_dict = row["aggregated_bias_scores"] or {}
        for ent in entity_list:
            vec.append(scores_dict.get(ent, {}).get("final_bias_score", 0.0))
        rows.append(vec)

    bias_df = pd.DataFrame(rows, columns=entity_list,
                           index=df_final.index).fillna(0.0)
    return bias_df, entity_list


def l2_normalize(df: pd.DataFrame) -> pd.DataFrame:
    values = df.values.astype(float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pd.DataFrame(values / norms, columns=df.columns, index=df.index)


def choose_k_auto(X: pd.DataFrame, k_min=2, k_max=6, random_state=42) -> Tuple[int, Dict[int, float]]:
    n = len(X)
    if n < 2:
        return 1, {}
    k_max = min(k_max, n)
    best_k, best_score = None, -1.0
    scores: Dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            s = silhouette_score(X, labels)
            scores[k] = s
            if s > best_score:
                best_k, best_score = k, s
    if best_k is None:
        best_k = min(max(2, k_min), k_max)
    return best_k, scores


def run_kmeans(X: pd.DataFrame, k: int, random_state=42) -> Tuple[np.ndarray, KMeans | None]:
    if k <= 1 or len(X) < 2:
        return np.zeros(len(X), dtype=int), None
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def compute_centroids(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = X.copy()
    df["cluster"] = labels
    centroids = df.groupby("cluster").mean(numeric_only=True)
    centroids.index.name = "cluster"
    return centroids.reset_index()


def save_outputs(
    df_final: pd.DataFrame,
    labels: np.ndarray,
    bias_df: pd.DataFrame,
    out_dir: str,
    centroids: pd.DataFrame,
):
    os.makedirs(out_dir, exist_ok=True)

    df_out = df_final.copy()
    df_out["viewpoint_cluster"] = labels if len(labels) else 0

    # Provide a normalized numeric cluster_id column for downstream scripts
    def _normalize_cluster(val):
        if isinstance(val, (list, tuple)) and val:
            val = val[0]
        try:
            if isinstance(val, str) and not val.isdigit():
                return -1
            return int(val)
        except Exception:
            return -1

    df_out["cluster_id"] = df_out["viewpoint_cluster"].apply(
        _normalize_cluster)

    clustered_csv = os.path.join(out_dir, "clustered_articles.csv")
    df_out.to_csv(clustered_csv, index=False)

    # Assignments per cluster (list of items)
    assignments: Dict[str, List[Dict]] = {}
    for cid in sorted(df_out["viewpoint_cluster"].unique()):
        rows = df_out[df_out["viewpoint_cluster"] == cid]
        items: List[Dict] = []
        for i, r in rows.iterrows():
            items.append({
                "index": int(i),
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "media_source": str(r.get("media_source", "")),
            })
        assignments[str(int(cid))] = items
    with open(os.path.join(out_dir, "cluster_assignments.json"), "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)

    # Centroids per entity (long form)
    centroids_long = centroids.melt(
        id_vars=["cluster"], var_name="entity", value_name="mean_final_bias_score"
    )
    centroids_long.to_csv(os.path.join(
        out_dir, "cluster_centroids.csv"), index=False)

    # Concatenated raw text per cluster (pre-merge)
    cluster_text: Dict[str, Dict] = {}
    for cid in sorted(df_out["viewpoint_cluster"].unique()):
        rows = df_out[df_out["viewpoint_cluster"] == cid]
        combined = " ".join([t for t in rows.get(
            "text", pd.Series([], dtype=str)).astype(str)])
        cluster_text[str(int(cid))] = {
            "article_count": int(len(rows)),
            "text": combined
        }
    with open(os.path.join(out_dir, "cluster_text.json"), "w", encoding="utf-8") as f:
        json.dump({"clusters": [
            {"cluster_id": int(
                k), "article_count": v["article_count"], "text": v["text"]}
            for k, v in cluster_text.items()
        ]}, f, indent=2, ensure_ascii=False)

    # Bias matrix
    bias_df.to_csv(os.path.join(out_dir, "bias_matrix.csv"), index=False)

    # Metadata
    meta = {
        "input_csv": INPUT_CSV,
        "normalize": NORMALIZE,
        "fixed_k": FIXED_K,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "random_state": RANDOM_STATE,
        "n_articles": int(len(df_final)),
        "n_entities": int(bias_df.shape[1]),
    }
    with open(os.path.join(out_dir, "cluster_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Outputs written to: {out_dir}")
    print(" - clustered_articles.csv")
    print(" - cluster_assignments.json")
    print(" - cluster_centroids.csv")
    print(" - cluster_text.json")
    print(" - bias_matrix.csv")
    print(" - cluster_metadata.json")


def main():
    df_final = load_final(INPUT_CSV)
    if df_final.empty:
        print("[ERROR] No scoring data; aborting clustering.")
        return

    bias_df, entity_list = build_bias_matrix(df_final)
    X = l2_normalize(bias_df) if NORMALIZE else bias_df

    if FIXED_K and FIXED_K > 0:
        k = min(FIXED_K, len(df_final))
        print(f"[INFO] Using fixed k={k}")
    else:
        k, scores = choose_k_auto(X, k_min=K_MIN, k_max=min(
            K_MAX, len(df_final)), random_state=RANDOM_STATE)
        print(f"[INFO] Auto-selected k={k} via silhouette. Scores: {scores}")

    labels, _ = run_kmeans(X, k)
    print(
        f"[INFO] Clustered {len(df_final)} articles into {k if k > 1 else 1} cluster(s).")

    centroids = compute_centroids(bias_df, labels if len(
        labels) else np.zeros(len(bias_df), dtype=int))
    save_outputs(df_final, labels, bias_df, OUTPUT_DIR, centroids)


if __name__ == "__main__":
    main()
