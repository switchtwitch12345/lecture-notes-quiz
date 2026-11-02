"""Evaluate the full pipeline on the six synthetic Hindi lectures.

For each lecture it records:
- speech recognition word and character error rates against the original text;
- whether notes and a valid five-question quiz were produced, and how many attempts it took;
- every quiz question in results/quiz_review.csv, with blank columns for a reviewer to fill in.

Usage (from the project root):
    .venv/bin/python scripts/evaluate_pipeline.py
"""
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lecture_notes import asr, quiz, translate  # noqa: E402
from evaluate_asr import edit_distance, normalise  # noqa: E402

LECTURES = os.path.join(ROOT, "data", "lectures", "lectures.json")
AUDIO = os.path.join(ROOT, "data", "lectures", "audio")
RESULTS = os.path.join(ROOT, "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    with open(LECTURES, encoding="utf-8") as fh:
        lectures = json.load(fh)

    summary = []
    review = []
    transcripts = {}
    for lecture in lectures:
        audio_path = os.path.join(AUDIO, lecture["id"] + ".wav")
        started = time.time()
        transcript = asr.transcribe(audio_path)
        asr_seconds = time.time() - started

        ref_words = normalise(lecture["text"]).split()
        hyp_words = normalise(transcript).split()
        word_errors = edit_distance(ref_words, hyp_words)
        char_errors = edit_distance(normalise(lecture["text"]), normalise(transcript))
        wer = word_errors / max(len(ref_words), 1)
        cer = char_errors / max(len(normalise(lecture["text"])), 1)

        english = translate.translate(transcript, "Hindi")
        started = time.time()
        data, errors = quiz.generate_notes_and_quiz(english)
        quiz_seconds = time.time() - started
        valid = data is not None
        questions = data["questions"] if valid else []

        summary.append({
            "lecture": lecture["id"],
            "wer": round(wer, 3),
            "cer": round(cer, 3),
            "quiz_valid": valid,
            "failed_attempts": len(errors),
            "questions": len(questions),
            "asr_seconds": round(asr_seconds, 1),
            "quiz_seconds": round(quiz_seconds, 1),
        })
        print(summary[-1], flush=True)

        for number, q in enumerate(questions, 1):
            review.append({
                "lecture": lecture["id"],
                "number": number,
                "question": q["question"],
                "option_A": q["options"]["A"],
                "option_B": q["options"]["B"],
                "option_C": q["options"]["C"],
                "option_D": q["options"]["D"],
                "marked_answer": q["answer"],
                "reviewer_answer_correct": "",
                "reviewer_supported_by_lecture": "",
                "reviewer_notes": "",
            })

        transcripts[lecture["id"]] = {"transcript": transcript, "english": english}

    with open(os.path.join(RESULTS, "pipeline_transcripts.json"), "w", encoding="utf-8") as fh:
        json.dump(transcripts, fh, ensure_ascii=False, indent=1)

    with open(os.path.join(RESULTS, "pipeline_summary.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with open(os.path.join(RESULTS, "quiz_review.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(review[0].keys()))
        writer.writeheader()
        writer.writerows(review)


if __name__ == "__main__":
    main()
