#!/usr/bin/env python3
"""
build_subset_with_text_hdd.py  —  sample JSONL and attach cleaned DocBank text efficiently (HDD‐optimized)
"""
import os
import gzip
import random
import re
from glob import iglob
from typing import List, Dict, Optional, Tuple
import orjson as json
from concurrent.futures import ThreadPoolExecutor

# -- Configuration: tweak these paths & params --
JSON_DIR     = "pipeline_out/jsonl_docbank"  # where your 500k_train(.jsonl.gz), etc live
TXT_DIR      = r"D:\Bigger Documents\Masters\SPRINT\DocBank_500K_txt"  # your HDD‐based .txt folder
OUTPUT_DIR   = "subset_500000"
SUBSET_SIZE  = 500000
SEED         = 42

# HDD‐friendly defaults:
USE_IN_MEMORY_SAMPLING = False  # streaming reservoir sampler
MATCH_THREADS         = 4      # 4–6 threads to avoid HDD thrash

# ─────────────────────────────────────────────────────────────────────────────
def open_maybe_gzip(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode) if "b" in mode else gzip.open(path, mode, encoding="utf-8")
    return open(path, mode) if "b" in mode else open(path, mode, encoding="utf-8")

def reservoir_sample_jsonl(path: str, k: int, seed: int = 42) -> List[dict]:
    random.seed(seed)
    reservoir: List[dict] = []
    with open_maybe_gzip(path, "rt") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            if i < k:
                reservoir.append(rec)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = rec
    return reservoir

def sample_in_memory(path: str, k: int, seed: int = 42) -> List[dict]:
    random.seed(seed)
    with open_maybe_gzip(path, "rt") as f:
        lines = f.read().splitlines()
    chosen = random.sample(lines, k)
    return [json.loads(l) for l in chosen]

def build_txt_index(txt_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    exact: Dict[str, List[str]] = {}
    prefix: Dict[str, List[str]] = {}
    for fullpath in iglob(os.path.join(txt_dir, "**", "*.txt"), recursive=True):
        stem = os.path.splitext(os.path.basename(fullpath))[0].lower()
        if stem.endswith("_ori"):
            stem = stem[:-4]
        exact.setdefault(stem, []).append(fullpath)
        pkey = stem.split("_", 1)[0]
        prefix.setdefault(pkey, []).append(fullpath)
    return exact, prefix

_filename_re = re.compile(r"<filename>([^<]+)\.(?:jpe?g|png|tiff?)</filename>", re.I)

def filename_from_rec(rec: dict) -> Optional[str]:
    for key in ("image_file", "file"):
        if key in rec and rec[key]:
            base = os.path.splitext(os.path.basename(rec[key]))[0]
            return base.rstrip("_ori")
    xml = rec.get("xml", "")
    m = _filename_re.search(xml)
    if m:
        base = os.path.splitext(os.path.basename(m.group(1)))[0]
        return base.rstrip("_ori")
    return None

def clean_text(path: str) -> str:
    words: List[str] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            words.append(ln.split("\t", 1)[0])
    return " ".join(words)

def process_split(jsonl_path: str,
                  exact: Dict[str, List[str]],
                  prefix: Dict[str, List[str]],
                  k: int,
                  seed: int,
                  out_ok: str,
                  out_bad: str):
    # 1) Sample
    if USE_IN_MEMORY_SAMPLING:
        records = sample_in_memory(jsonl_path, k, seed)
    else:
        records = reservoir_sample_jsonl(jsonl_path, k, seed)

    matched = 0
    # 2) Match & clean in parallel
    with open(out_ok, "w", encoding="utf-8") as fout_ok, \
         open(out_bad, "w", encoding="utf-8") as fout_bad, \
         ThreadPoolExecutor(max_workers=MATCH_THREADS) as pool:

        def match_and_clean(rec):
            base = filename_from_rec(rec)
            if not base:
                return rec, False
            key = base.lower()
            # exact
            candidates = exact.get(key, [])
            # prefix fallback
            if not candidates:
                pkey = key.split("_", 1)[0]
                for p in prefix.get(pkey, []):
                    name = os.path.basename(p).lower()
                    if name.startswith(key) and name.endswith(".txt"):
                        candidates = [p]
                        break
            if candidates:
                rec["text"] = clean_text(candidates[0])
                return rec, True
            return rec, False

        for rec, ok in pool.map(match_and_clean, records):
            line = json.dumps(rec).decode("utf-8") + "\n"
            (fout_ok if ok else fout_bad).write(line)
            if ok:
                matched += 1

    print(f"✓ {matched}/{k} matched for {os.path.basename(jsonl_path)}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exact_idx, prefix_idx = build_txt_index(TXT_DIR)

    splits  = ["500k_train", "500k_valid", "500k_test"]
    ratios  = [0.8, 0.1, 0.1]
    counts  = [int(SUBSET_SIZE * r) for r in ratios]
    counts[2] = SUBSET_SIZE - sum(counts[:2])

    for split, k in zip(splits, counts):
        found = False
        for ext in (".jsonl.gz", ".jsonl", ".json"):
            src = os.path.join(JSON_DIR, split + ext)
            if os.path.isfile(src):
                process_split(
                    jsonl_path=src,
                    exact=exact_idx,
                    prefix=prefix_idx,
                    k=k,
                    seed=SEED,
                    out_ok=os.path.join(OUTPUT_DIR, f"{split}.with_text.jsonl"),
                    out_bad=os.path.join(OUTPUT_DIR, f"{split}.unmatched.jsonl"),
                )
                found = True
                break
        if not found:
            print(f"Error: no JSONL found for split '{split}' in {JSON_DIR}")

if __name__ == "__main__":
    main()
