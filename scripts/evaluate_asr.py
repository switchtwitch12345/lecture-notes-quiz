"""Evaluate SraVaani-1.0 on FLEURS test clips for Hindi, Kannada and Tamil.

Usage (from the project root):
    PYTHONPATH=vendor-sravaani python scripts/evaluate_asr.py
"""
import csv
import os
import random
import re
import time
import unicodedata

from transformers import AutoModel

from sravaani_flow.config import MODEL_REPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEURS = os.path.join(ROOT, "data", "fleurs")
RESULTS = os.path.join(ROOT, "results")

LANGUAGES = {"hi_in": "Hindi", "kn_in": "Kannada", "ta_in": "Tamil"}
PER_LANGUAGE = 40
SEED = 0
BATCH = 8


def normalise(text):
    """Lowercase, strip punctuation and symbols, collapse whitespace."""
    kept = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            kept.append(" ")
        else:
            kept.append(ch)
    return re.sub(r"\s+", " ", "".join(kept)).strip().lower()


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def load_test_set(lang_code):
    rows = []
    path = os.path.join(FLEURS, "%s.test.tsv" % lang_code)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rows.append({"file": parts[1], "reference": parts[3]})
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True, dtype="float32")
    model = model.to("cpu").eval()

    rng = random.Random(SEED)
    out_rows = []
    summary = []

    for code, name in LANGUAGES.items():
        rows = load_test_set(code)
        sample = rng.sample(rows, min(PER_LANGUAGE, len(rows)))
        audio_dir = os.path.join(FLEURS, code)
        paths = []
        for row in sample:
            matches = []
            for dirpath, _, files in os.walk(audio_dir):
                if row["file"] in files:
                    matches.append(os.path.join(dirpath, row["file"]))
            paths.append(matches[0])

        started = time.time()
        hyps = []
        for start in range(0, len(paths), BATCH):
            hyps.extend(model.transcribe(paths[start:start + BATCH]))
        elapsed = time.time() - started

        word_errors = word_total = char_errors = char_total = 0
        for row, hyp in zip(sample, hyps):
            ref_n = normalise(row["reference"])
            hyp_n = normalise(hyp)
            ref_w, hyp_w = ref_n.split(), hyp_n.split()
            w_err = edit_distance(ref_w, hyp_w)
            c_err = edit_distance(ref_n, hyp_n)
            word_errors += w_err
            word_total += len(ref_w)
            char_errors += c_err
            char_total += len(ref_n)
            out_rows.append({
                "language": name,
                "file": row["file"],
                "reference": row["reference"],
                "hypothesis": hyp,
                "word_errors": w_err,
                "ref_words": len(ref_w),
                "wer": round(w_err / max(len(ref_w), 1), 3),
            })

        wer = word_errors / max(word_total, 1)
        cer = char_errors / max(char_total, 1)
        summary.append((name, len(sample), wer, cer, elapsed))
        print("%-8s n=%d  WER=%.1f%%  CER=%.1f%%  time=%.0fs" % (name, len(sample), 100 * wer, 100 * cer, elapsed))

    with open(os.path.join(RESULTS, "asr_results.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    with open(os.path.join(RESULTS, "asr_summary.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "clips", "wer", "cer", "seconds"])
        for name, n, wer, cer, elapsed in summary:
            writer.writerow([name, n, round(wer, 4), round(cer, 4), round(elapsed, 1)])


if __name__ == "__main__":
    main()
