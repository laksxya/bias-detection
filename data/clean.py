import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
import tldextract
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import dateutil.parser as dateparser

INPUT_FILE = Path(__file__).parent / "links.json"
OUTPUT_FILE = Path(__file__).parent / "links.cleaned.json"
MIN_CONTENT_CHARS = 250
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SECONDS = 1.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
}

BOILERPLATE_PATTERNS = [
    r"\bsubscribe\b", r"\bsign in\b", r"\blog ?in\b", r"\bregister\b",
    r"\bnewsletter\b", r"\bfollow us\b", r"\bdownload the app\b",
    r"\bread more\b", r"\bwatch live\b", r"\btrending\b",
    r"\ball rights reserved\b", r"\bprivacy policy\b", r"\bterms\b",
    r"\bepaper\b", r"\bclick here\b", r"\bmost read\b", r"\btop stories\b",
    r"©\s?\d{4}", r"\bcopyright\b"
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.I)


def fetch_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS,
                            timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def pick_better_url(url: str, html: str) -> str:
    # Prefer canonical or AMP if available
    try:
        soup = BeautifulSoup(html, "html.parser")
        can = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
        amp = soup.find("link", rel=lambda v: v and "amphtml" in v.lower())
        candidate = (amp.get("href") if amp and amp.get("href") else None) or (
            can.get("href") if can and can.get("href") else None)
        if candidate and candidate.startswith("http"):
            return candidate
    except Exception:
        pass
    return url


def extract_with_trafilatura(url: str, html: str) -> dict:
    try:
        # Favor recall for news; get JSON metadata
        res = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=False,
            favor_recall=True,
            output_format="json",
            url=url,
        )
        if not res:
            return {}
        data = json.loads(res)
        text = data.get("text") or ""
        title = data.get("title") or ""
        date_str = data.get("date") or data.get("published")
        publish_date = None
        if date_str:
            try:
                publish_date = dateparser.parse(date_str).date().isoformat()
            except Exception:
                publish_date = None
        return {"title": title, "text": text, "publish_date": publish_date}
    except Exception:
        return {}


def extract_with_readability(url: str, html: str) -> dict:
    try:
        doc = Document(html)
        title = (doc.short_title() or "").strip()
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "html.parser")
        # Keep paragraphs only
        parts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(parts).strip()
        return {"title": title, "text": text, "publish_date": None}
    except Exception:
        return {}


def simple_paragraph_join(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Try main/article areas first
    candidates = []
    for sel in ["article", "[role=main]", "main", "div[itemprop='articleBody']"]:
        for node in soup.select(sel):
            ps = [p.get_text(" ", strip=True) for p in node.find_all("p")]
            txt = "\n".join(ps).strip()
            if txt:
                candidates.append(txt)
    if candidates:
        return max(candidates, key=len)
    # Fallback: all paragraphs
    return "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p")).strip()


def post_clean(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = []
    for ln in lines:
        if not ln:
            continue
        # Drop boilerplate-y lines
        if BOILERPLATE_RE.search(ln):
            continue
        # Drop lines with very low alpha ratio
        alpha = sum(c.isalpha() for c in ln)
        if alpha < 0.5 * max(1, len(ln)):
            # keep even if low alpha if long and sentence-like
            pass
        # Drop very short non-sentences
        if len(ln) < 40 and not ln.endswith((".", "!", "?")):
            continue
        cleaned.append(ln)
    # Deduplicate adjacent lines
    deduped = []
    prev = None
    for ln in cleaned:
        if ln != prev:
            deduped.append(ln)
        prev = ln
    text = "\n".join(deduped)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def choose_best(*cands: dict) -> dict:
    # Score by length and punctuation density
    best = {}
    best_score = -1
    for c in cands:
        if not c or not c.get("text"):
            continue
        txt = c["text"].strip()
        length = len(txt)
        if length == 0:
            continue
        punct = sum(ch in ".!?" for ch in txt)
        score = length + 50 * punct  # simple heuristic
        if score > best_score:
            best = c
            best_score = score
    return best


def reclean_item(item: dict) -> dict:
    url = item.get("url")
    if not url:
        return item
    html = fetch_html(url)
    if not html:
        # As a last resort, clean the existing content
        existing = item.get("content") or ""
        item["content"] = post_clean(existing)
        return item

    # Prefer canonical/amp and re-fetch once if different
    best_url = pick_better_url(url, html)
    if best_url != url:
        html2 = fetch_html(best_url)
        if html2:
            url, html = best_url, html2

    # Try extractors
    t = extract_with_trafilatura(url, html)
    r = extract_with_readability(url, html)
    j = {"title": None, "text": simple_paragraph_join(
        html), "publish_date": None}

    best = choose_best(t, r, j)
    if best and best.get("text"):
        cleaned_text = post_clean(best["text"])
        if len(cleaned_text) >= MIN_CONTENT_CHARS:
            extracted = tldextract.extract(url)
            item.update({
                "url": url,
                "media_source": f"{extracted.domain}.{extracted.suffix}",
                "title": best.get("title") or item.get("title"),
                "publish_date": best.get("publish_date") or item.get("publish_date"),
                "content": cleaned_text,
            })
            # Drop fields often noisy after re-extraction
            item["keywords"] = item.get("keywords", [])
            item["summary"] = item.get("summary", "")
            return item

    # Fallback: at least clean what we already had
    existing = item.get("content") or ""
    item["content"] = post_clean(existing)
    return item


def main():
    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    cleaned = []
    for idx, item in enumerate(raw, 1):
        print(f"[{idx}/{len(raw)}] Cleaning: {item.get('url')}")
        try:
            new_item = reclean_item(dict(item))
            cleaned.append(new_item)
        except Exception as e:
            print(f"  -> Failed: {e}")
            cleaned.append(item)
        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_FILE.write_text(json.dumps(
        cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Wrote cleaned articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
