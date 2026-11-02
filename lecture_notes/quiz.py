"""Notes and five-question MCQ generation with Qwen2.5-1.5B-Instruct, returned as validated JSON."""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 700
QUESTIONS = 5
OPTIONS = ["A", "B", "C", "D"]

PROMPT = """Below is a lecture transcript. Use ONLY facts stated in the transcript. Do not add names, dates or facts from outside it.

Return JSON only, in this form:
{{"notes": ["point 1", "point 2", ...],
  "questions": [
    {{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}
  ]}}

Give 4 to 6 notes and exactly 5 questions. Each question must be answerable from the transcript, and "answer" must be the letter of the correct option. Write the notes and questions in English, even if the transcript is in another language.

Transcript:
{transcript}
"""

_tokenizer = None
_model = None
_device = None


def load_model():
    global _tokenizer, _model, _device
    if _model is None:
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16 if _device == "mps" else torch.float32
        ).to(_device).eval()
    return _tokenizer, _model, _device


def _generate(transcript, feedback=None):
    tokenizer, model, device = load_model()
    messages = [{"role": "user", "content": PROMPT.format(transcript=transcript)}]
    if feedback:
        # Greedy decoding is deterministic, so a plain retry would repeat the same output.
        messages.append({"role": "assistant", "content": feedback["previous"]})
        messages.append({
            "role": "user",
            "content": "That answer was rejected: %s. Return the complete JSON again, fixed." % feedback["error"],
        })
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def _parse(raw):
    """Return (data, error). data is None if the text is not valid notes and quiz JSON."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, "no JSON object found"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, "invalid JSON: %s" % exc.msg

    notes = data.get("notes")
    questions = data.get("questions")
    if not isinstance(notes, list) or not notes or not all(isinstance(n, str) for n in notes):
        return None, "notes missing or malformed"
    if not isinstance(questions, list) or len(questions) != QUESTIONS:
        return None, "expected %d questions" % QUESTIONS
    for q in questions:
        options = q.get("options") if isinstance(q, dict) else None
        if not isinstance(q.get("question"), str) or not isinstance(options, dict):
            return None, "question missing text or options"
        if sorted(options.keys()) != OPTIONS or q.get("answer") not in OPTIONS:
            return None, "question options or answer letter invalid"
    return data, None


def generate_notes_and_quiz(transcript, attempts=3):
    """Return (data, errors). data is None if every attempt produced invalid output."""
    errors = []
    feedback = None
    for _ in range(attempts):
        raw = _generate(transcript, feedback)
        data, error = _parse(raw)
        if data is not None:
            return data, errors
        errors.append(error)
        feedback = {"previous": raw, "error": error}
    return None, errors
