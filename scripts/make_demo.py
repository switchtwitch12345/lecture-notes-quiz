"""Run the app's processing on one lecture recording and save the output as a Markdown demo.

Usage (from the project root):
    .venv/bin/python scripts/make_demo.py lecture_04
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import process  # noqa: E402

LECTURE = sys.argv[1] if len(sys.argv) > 1 else "lecture_04"
AUDIO = os.path.join(ROOT, "data", "lectures", "audio", LECTURE + ".wav")
OUT_DIR = os.path.join(ROOT, "results", "demo")
OUT = os.path.join(OUT_DIR, LECTURE + ".md")


def main():
    transcript, english, notes, quiz, status = process(AUDIO, "", "Hindi")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# Demo run: %s\n\n" % LECTURE)
        fh.write("Input: `data/lectures/audio/%s.wav`, language Hindi. Output of `app.process`, unedited.\n\n" % LECTURE)
        fh.write("## Transcript (SraVaani)\n\n%s\n\n" % transcript)
        fh.write("## English translation (IndicTrans2)\n\n%s\n\n" % english)
        fh.write("## Notes (Qwen2.5-1.5B)\n\n%s\n\n" % notes)
        fh.write("## Quiz (Qwen2.5-1.5B)\n\n%s\n\n" % quiz)
        fh.write("## Status\n\n%s\n" % status)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
