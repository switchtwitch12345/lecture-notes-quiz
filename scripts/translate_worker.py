"""Translate sentences with IndicTrans2 (1B). Runs in .venv-translate, which has transformers 4.56.

Reads JSON from stdin: {"sentences": [...], "src_lang": "hin_Deva"}
Writes JSON to stdout: {"translations": [...]}
"""
import json
import sys

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "ai4bharat/indictrans2-indic-en-1B"
TGT_LANG = "eng_Latn"
NUM_BEAMS = 4
BATCH_SIZE = 8


def main():
    request = json.load(sys.stdin)
    sentences = request["sentences"]
    src_lang = request["src_lang"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).eval()
    processor = IndicProcessor(inference=True)

    translations = []
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = processor.preprocess_batch(
            sentences[start:start + BATCH_SIZE], src_lang=src_lang, tgt_lang=TGT_LANG
        )
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            generated = model.generate(
                **inputs, use_cache=False, min_length=0, max_length=256, num_beams=NUM_BEAMS, num_return_sequences=1
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations.extend(processor.postprocess_batch(decoded, lang=TGT_LANG))

    json.dump({"translations": translations}, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
