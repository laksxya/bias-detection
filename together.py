import os
import sys
import subprocess
import shutil

ROOT = r"c:\Users\ishuv\Desktop\Capstone"
PY = sys.executable  # current venv python


def run_step(desc, script_path):
    print(f"\n[STEP] {desc}")
    cmd = [PY, "-u", script_path]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        raise subprocess.CalledProcessError(r.returncode, cmd)


def safe_import_or_run(module_path, fallback_script):
    rel = os.path.relpath(module_path, ROOT)
    mod_name = rel.replace("\\", ".").replace(".py", "")
    try:
        mod = __import__(mod_name, fromlist=["main"])
        if hasattr(mod, "main"):
            print(f"[STEP] {mod_name}.main()")
            mod.main()
            return
    except Exception as e:
        print(f"[INFO] Import fallback for {mod_name}: {e}")
    run_step(mod_name, fallback_script)


def run_pipeline(topic: str):
    """
    Runs the entire data processing and report generation pipeline for a given topic.
    """
    print(f"[PIPELINE] Start full run for topic: '{topic}'")

    # Hardcode the Gemini API Key here
    GEMINI_KEY = "AIzaSyCHkTBb9p2id7l2GqzCdSwRo4QocjQxCA8"

    # Ensure env vars for downstream scripts
    os.environ["GEMINI_KEY"] = GEMINI_KEY
    os.environ["REPORT_TOPIC"] = topic
    os.environ["CLAIM_MODEL_NAME"] = "google/flan-t5-base"
    os.environ["EXTRACT_NUM_BEAMS"] = "1"
    os.environ["EXTRACT_MAX_NEW_TOKENS"] = "128"
    os.environ["EXTRACT_MAX_WINDOW_TOKENS"] = "448"
    os.environ["EXTRACT_WINDOW_OVERLAP"] = "24"
    os.environ["SCORING_DEVICE"] = "cpu"

    # 1) Data collection / cleaning
    data_dir = os.path.join(ROOT, "data")
    article_links = os.path.join(data_dir, "article_links.py")
    clean_script = os.path.join(data_dir, "clean.py")
    links_cleaned = os.path.join(data_dir, "links.cleaned.json")
    if os.path.exists(article_links):
        safe_import_or_run(article_links, article_links)
    if os.path.exists(clean_script):
        safe_import_or_run(clean_script, clean_script)
    if not os.path.exists(links_cleaned):
        raise FileNotFoundError("Missing links.cleaned.json after data step.")

    # 2) Entity mappings / NER
    ent_dir = os.path.join(ROOT, "entity_mappings")
    ner_script = os.path.join(ent_dir, "ner.py")
    ner_out = os.path.join(ent_dir, "entire.csv")
    if os.path.exists(ner_script):
        if os.path.exists(ner_out) and os.path.getmtime(ner_out) >= os.path.getmtime(links_cleaned):
            print("[STEP] entity_mappings.ner (skipped: up-to-date)")
        else:
            safe_import_or_run(ner_script, ner_script)

    # 3) Scoring
    scoring_dir = os.path.join(ROOT, "Scoring")
    score_script = os.path.join(scoring_dir, "score.py")
    if os.path.exists(score_script):
        safe_import_or_run(score_script, score_script)

    # 4) Clustering by bias
    report_dir = os.path.join(ROOT, "report")
    clusters_dir = os.path.join(report_dir, "clusters")
    shutil.rmtree(clusters_dir, ignore_errors=True)
    os.makedirs(clusters_dir, exist_ok=True)
    print("[STEP] Cleaned report/clusters (fresh cluster run).")
    cluster_script = os.path.join(report_dir, "cluster.py")
    safe_import_or_run(cluster_script, cluster_script)

    # 5) Two-level dedup
    dedup_script = os.path.join(report_dir, "deduplication.py")
    safe_import_or_run(dedup_script, dedup_script)

    # 6) Report generation
    reportgen_script = os.path.join(report_dir, "reportgen.py")
    safe_import_or_run(reportgen_script, reportgen_script)

    # 7) Copy final
    src_final = os.path.join(report_dir, "final.txt")
    dst_final = os.path.join(ROOT, "report.txt")
    if os.path.exists(src_final):
        shutil.copyfile(src_final, dst_final)
        print(f"[OUTPUT] Copied final report to {dst_final}")
    else:
        raise FileNotFoundError(
            "final.txt not found; pipeline may have failed earlier.")

    print("[PIPELINE] Complete.")


if __name__ == "__main__":
    # Default values for running this script directly
    DEFAULT_REPORT_TOPIC = "Bihar Elections 2025"
    try:
        run_pipeline(topic=DEFAULT_REPORT_TOPIC)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        sys.exit(1)
