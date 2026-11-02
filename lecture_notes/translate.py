"""Translate an Indian-language transcript into English with IndicTrans2.

IndicTrans2 needs transformers 4.56, which conflicts with the SraVaani environment (5.x), so the
translation runs as a separate process in .venv-translate.
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "scripts", "translate_worker.py")
PYTHON = os.path.join(ROOT, ".venv-translate", "bin", "python")

LANGUAGE_CODES = {"Hindi": "hin_Deva", "Kannada": "kan_Knda", "Tamil": "tam_Taml"}


def split_sentences(text):
    """Split on Indian and Latin sentence ends."""
    return [p for p in re.split(r"(?<=[।.?!])\s+", text.strip()) if p]


def translate(text, language):
    """Return the English translation of `text`, which is written in `language`."""
    if language not in LANGUAGE_CODES:
        raise ValueError("unsupported language: %s" % language)
    sentences = split_sentences(text)
    if not sentences:
        return ""
    request = json.dumps({"sentences": sentences, "src_lang": LANGUAGE_CODES[language]})
    result = subprocess.run(
        [PYTHON, WORKER], input=request, capture_output=True, text=True, check=True
    )
    return " ".join(t.strip() for t in json.loads(result.stdout)["translations"] if t.strip())
