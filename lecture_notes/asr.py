"""Speech-to-text with SraVaani-1.0, for lectures of any length."""
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor-sravaani"))

from transformers import AutoModel  # noqa: E402

from sravaani_flow.config import MODEL_REPO, SAMPLE_RATE  # noqa: E402

# SraVaani drops the opening words of clips longer than about 10 seconds, so keep chunks short
# and cut them at quiet points.
TARGET_SECONDS = 8
SEARCH_SECONDS = 1.5
FRAME_SECONDS = 0.1
BATCH_SIZE = 8

_model = None


def load_model():
    """Load SraVaani once and reuse it. CPU only, because MPS fails on a float64 op in its loader."""
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True, dtype="float32").to("cpu").eval()
    return _model


def read_mono_16k(path):
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
    return audio


def split_chunks(audio):
    """Cut the audio near every TARGET_SECONDS, at the quietest frame within SEARCH_SECONDS."""
    frame = int(FRAME_SECONDS * SAMPLE_RATE)
    search = int(SEARCH_SECONDS * SAMPLE_RATE)
    target = TARGET_SECONDS * SAMPLE_RATE
    cuts = [0]
    while len(audio) - cuts[-1] > target + search:
        centre = cuts[-1] + target
        window = audio[centre - search:centre + search]
        energies = [np.mean(window[i:i + frame] ** 2) for i in range(0, len(window) - frame, frame)]
        quietest = int(np.argmin(energies)) * frame
        cuts.append(centre - search + quietest)
    cuts.append(len(audio))
    return [audio[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def transcribe(path):
    """Return the full transcript of an audio file as one string."""
    model = load_model()
    chunks = split_chunks(read_mono_16k(path))
    pieces = []
    for start in range(0, len(chunks), BATCH_SIZE):
        pieces.extend(model.transcribe(chunks[start:start + BATCH_SIZE]))
    return " ".join(p.strip() for p in pieces if p.strip())
