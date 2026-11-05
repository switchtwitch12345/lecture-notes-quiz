"""Compare IndicConformer (CTC and RNNT) with the SraVaani ASR on the ten synthetic lectures.

Scores each transcript against the original lecture text with the same normalisation and
edit distance as the SraVaani evaluation. Writes results/asr_indicconformer.csv.

Usage (from the project root):
    .venv/bin/python scripts/evaluate_indicconformer.py
"""
import csv
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor-sravaani"))  # evaluate_asr imports sravaani_flow

from evaluate_asr import edit_distance, normalise  # noqa: E402

MODEL = "ai4bharat/indic-conformer-600m-multilingual"
LECTURES = os.path.join(ROOT, "data", "lectures", "lectures.json")
AUDIO = os.path.join(ROOT, "data", "lectures", "audio")
OUT = os.path.join(ROOT, "results", "asr_indicconformer.csv")


def main():
    with open(LECTURES, encoding="utf-8") as fh:
        lectures = json.load(fh)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()

    rows = []
    for lecture in lectures:
        audio, sr = sf.read(os.path.join(AUDIO, lecture["id"] + ".wav"), dtype="float32")
        assert sr == 16000
        wav = torch.from_numpy(audio).unsqueeze(0)
        ref = normalise(lecture["text"])
        for decoder in ("ctc", "rnnt"):
            with torch.no_grad():
                transcript = model(wav, "hi", decoder)
            hyp = normalise(transcript)
            ref_words, hyp_words = ref.split(), hyp.split()
            rows.append({
                "lecture": lecture["id"],
                "decoder": decoder,
                "wer": round(edit_distance(ref_words, hyp_words) / max(len(ref_words), 1), 3),
                "cer": round(edit_distance(ref, hyp) / max(len(ref), 1), 3),
                "transcript": transcript,
            })
            print({k: v for k, v in rows[-1].items() if k != "transcript"}, flush=True)

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
