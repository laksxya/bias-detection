from transformers import pipeline
import google.generativeai as genai
from typing import List, Dict, Any
import json
import re
import os
# Env vars BEFORE transformers import
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# Optional embeddings for provenance → claim source matching
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    print("[WARN] sentence-transformers not found. References will be empty. `pip install sentence-transformers`")
try:
    import torch
except ImportError:
    torch = None
    print("[WARN] torch not found. References will be empty. `pip install torch`")


ROOT_DIR = r"c:\Users\ishuv\Desktop\Capstone"
REPORT_DIR = os.path.join(ROOT_DIR, "report")
RAW_PATH = os.path.join(REPORT_DIR, "raw.txt")
FINAL_PATH = os.path.join(REPORT_DIR, "final.txt")
CLAIMS_DEBUG_PATH = os.path.join(REPORT_DIR, "claims.debug.json")
CLAIMS_LIST_PATH = os.path.join(REPORT_DIR, "claims.numbered.txt")
PROV_PATH = os.path.join(REPORT_DIR, "global_provenance.json")

# Tuning via env
CLAIM_MODEL_NAME = os.environ.get("CLAIM_MODEL_NAME", "google/flan-t5-base")
EXTRACT_NUM_BEAMS = int(os.environ.get("EXTRACT_NUM_BEAMS", "1"))
MAX_NEW_TOKENS_CLAIMS = "512"
MAX_WINDOW_TOKENS = int(os.environ.get("EXTRACT_MAX_WINDOW_TOKENS", "448"))
WINDOW_OVERLAP = int(os.environ.get("EXTRACT_WINDOW_OVERLAP", "24"))

OVERRIDE_TOPIC_ENV = "REPORT_TOPIC"
GEMINI_MODEL = "gemini-2.5-flash-lite"
TARGET_WORDS = 700
MIN_CLAIM_LEN = 25
MAX_CLAIMS_TOTAL = int(os.environ.get("MAX_CLAIMS_TOTAL", "120"))
TOP_REF_SENTENCES = int(os.environ.get("TOP_REF_SENTENCES", "3"))


def token_windows(text: str, tokenizer, max_tokens: int = MAX_WINDOW_TOKENS, overlap: int = WINDOW_OVERLAP) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max_tokens - overlap
    windows = []
    for start in range(0, len(ids), step):
        sub_ids = ids[start:start + max_tokens]
        if not sub_ids:
            break
        windows.append(tokenizer.decode(sub_ids, skip_special_tokens=True))
        if start + max_tokens >= len(ids):
            break
    return windows


def extract_claims_from_text(text: str, extractor) -> List[str]:
    prompt = f"Summarize the key facts from this news article as bullet points: \n\n{text}"
    result = extractor(
        prompt,
        num_beams=EXTRACT_NUM_BEAMS,
        max_new_tokens=512,
        truncation=True
    )
    gen = result[0]["generated_text"]
    lines = [ln.strip() for ln in gen.splitlines()]
    claims = []
    for ln in lines:
        if re.match(r"^[-*•]\s*", ln):
            claim = re.sub(r"^[-*•]\s*", "", ln).strip()
        else:
            claim = ln
        if len(claim) >= MIN_CLAIM_LEN and any(c.isalpha() for c in claim):
            claims.append(claim)
    return claims


def build_final_prompt(query: str, claims_texts: List[str]) -> str:
    numbered = [f"[{i}] {c}" for i, c in enumerate(claims_texts, start=1)]
    return f"""You are a neutral journalist.

Write a factual {TARGET_WORDS}-word article about "{query}" using ONLY the numbered claims below.

General Rules:
1. You MUST NOT invent or introduce any facts that do not appear in the claims list. All information must be traceable to a cited claim.
2. You MAY rewrite or paraphrase claims to remove bias, emotional language, sensationalism, or loaded wording. The factual meaning must stay intact.
3. Cite every factual statement using the claim number in square brackets (e.g., [4][7]).
4. Present information in a clear chronological or logical structure. If no chronology is implied, group related claims together coherently.

Handling Bias and Imbalance:
5. If two or more claims contradict each other, report the contradiction and cite all relevant claims instead of choosing one.
6. If the claims overwhelmingly describe one side, interpretation, or perspective, include a sentence noting that the provided claims do not contain counterclaims, rebuttals, or alternative viewpoints, and cite the relevant claim numbers.
7. Remove or rephrase emotionally charged language, such as exaggeration, celebration, ridicule, or moral judgement (e.g., "tsunami", "disaster", "mastermind", "heroic", "shameful"), unless directly quoted. If quoted, clearly attribute it to the claim.

Writing Requirements:
8. Keep tone factual, concise, and neutral. Do not praise, condemn, or speculate.
9. DO NOT repeat the same actor, achievement, or narrative more than necessary. Avoid amplification bias.
10. Produce a short neutral headline (maximum 10 words), then the article body.

Numbered Claims:
{chr(10).join(numbered)}

Begin the article now:
"""


def infer_query_from_sources(raw_text: str) -> str:
    override = os.environ.get(OVERRIDE_TOPIC_ENV, "").strip()
    if override:
        return override
    first = next((ln for ln in raw_text.splitlines() if ln.strip()), "")
    return " ".join(first.split()[:8]) if first else "Topic"


def attach_provenance(claim_items: List[Dict[str, Any]], provenance_sentences: List[Dict[str, Any]]):
    if not _HAS_ST or not torch:
        print(
            "[WARN] Skipping source attachment: sentence-transformers or torch is missing.")
        return
    if not provenance_sentences:
        print("[WARN] Skipping source attachment: Provenance data is empty.")
        return

    prov_texts = [p.get("sentence", "")
                  for p in provenance_sentences if isinstance(p.get("sentence"), str)]
    if not prov_texts:
        print(
            "[WARN] Skipping source attachment: No valid sentences found in provenance data.")
        return

    print("[INFO] Attaching sources to claims via semantic search...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    prov_embs = model.encode(
        prov_texts, convert_to_tensor=True, normalize_embeddings=True)
    claim_texts = [c["text"] for c in claim_items]
    claim_embs = model.encode(
        claim_texts, convert_to_tensor=True, normalize_embeddings=True)

    for i, claim in enumerate(claim_items):
        sims = util.cos_sim(claim_embs[i].unsqueeze(0), prov_embs).flatten()
        k = min(TOP_REF_SENTENCES, len(sims))
        top_idx = torch.topk(sims, k=k).indices.tolist()
        sources = []
        for idx in top_idx:
            prov_obj = provenance_sentences[idx]
            if "sources" in prov_obj and isinstance(prov_obj["sources"], list):
                for s in prov_obj["sources"]:
                    sources.append({
                        "title": s.get("title", ""),
                        "url": s.get("url", ""),
                        "cluster_id": s.get("cluster_id"),
                        "article_index": s.get("article_index")
                    })
        uniq = []
        seen = set()
        for s in sources:
            sig = (s["url"], s["title"])
            if sig not in seen:
                seen.add(sig)
                uniq.append(s)
        claim["sources"] = uniq


def main():
    print("[START] Report generation.")
    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] Missing raw.txt at {RAW_PATH}. Run dedup first.")
        return
    raw_text = open(RAW_PATH, "r", encoding="utf-8").read()
    print(f"[INFO] raw.txt chars={len(raw_text)}")

    provenance = []
    if os.path.exists(PROV_PATH):
        try:
            provenance = json.load(
                open(PROV_PATH, "r", encoding="utf-8")).get("sentences", [])
            print(f"[INFO] Loaded provenance sentences={len(provenance)}")
        except Exception as e:
            print(f"[WARN] Failed to load or parse provenance file: {e}")
    else:
        print("[WARN] global_provenance.json not found. References will be empty.")

    device_req = os.environ.get("REPORTGEN_DEVICE", "auto").lower()
    device_index = -1
    try:
        if torch and torch.cuda.is_available():
            if device_req in ("auto", "cuda"):
                device_index = 0
                print(
                    f"[INFO] CUDA available: True | Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("[INFO] CUDA available but CPU forced by environment variable.")
        else:
            print("[INFO] CUDA unavailable; using CPU.")
    except Exception:
        print("[INFO] Torch device check failed; CPU fallback.")

    extractor = pipeline(
        "text2text-generation",
        model=CLAIM_MODEL_NAME,
        framework="pt",
        device=device_index
    )
    tokenizer = extractor.tokenizer

    claim_items: List[Dict[str, Any]] = []
    windows = token_windows(raw_text, tokenizer)
    print(f"[CLAIMS] Windows in raw: {len(windows)}")
    for w in windows:
        claims = extract_claims_from_text(w, extractor)
        for c in claims:
            claim_items.append({"text": c, "sources": []})
        if len(claim_items) >= MAX_CLAIMS_TOTAL:
            print("[WARN] Claims cap reached.")
            break

    print(f"[INFO] Raw claims extracted: {len(claim_items)}")

    attach_provenance(claim_items, provenance)

    claims_texts = [it["text"] for it in claim_items]

    with open(CLAIMS_DEBUG_PATH, "w", encoding="utf-8") as f:
        json.dump({"raw_claim_count": len(claim_items),
                   "claims": claim_items}, f, indent=2, ensure_ascii=False)
    with open(CLAIMS_LIST_PATH, "w", encoding="utf-8") as f:
        for i, it in enumerate(claim_items, start=1):
            srcs = it.get("sources", [])
            src_line = "; ".join((s.get("title") or s.get("url") or "")[
                                 :80] for s in srcs[:3])
            more = "" if len(srcs) <= 3 else f" (+{len(srcs)-3} more)"
            f.write(f"[{i}] {it['text']} -- {src_line}{more}\n")
    print(f"[DEBUG] Saved {CLAIMS_DEBUG_PATH}")
    print(f"[DEBUG] Saved {CLAIMS_LIST_PATH}")

    key = os.environ.get("GEMINI_KEY")
    if not key:
        print("[ERROR] Missing GEMINI_KEY environment variable.")
        return
    genai.configure(api_key=key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    query = infer_query_from_sources(raw_text)
    print(f"[INFO] Topic for article: {query}")
    prompt = build_final_prompt(query, claims_texts)
    print("[INFO] Generating article.")
    resp = model.generate_content(
        prompt, generation_config={"temperature": 0.0})
    article = getattr(resp, "text", str(resp)).rstrip()

    ref_lines = ["", "References:"]
    for i, c in enumerate(claim_items, start=1):
        ref_lines.append(f"[{i}]")
        sources_for_claim = c.get("sources", [])
        if not sources_for_claim:
            ref_lines.append(
                "- No specific source could be matched for this claim.")
        else:
            for s in sources_for_claim[:3]:
                title = (s.get("title") or "").strip()
                url = s.get("url") or ""
                line = f"- {title or url}"
                if title and url:
                    line += f" ({url})"
                ref_lines.append(line)
            if len(sources_for_claim) > 3:
                ref_lines.append(f"- (+{len(sources_for_claim)-3} more)")

    final_out = article + "\n\n" + "\n".join(ref_lines) + "\n"
    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        f.write(final_out)
    print(f"[DONE] Saved {FINAL_PATH} chars={len(final_out)}")


if __name__ == "__main__":
    main()
