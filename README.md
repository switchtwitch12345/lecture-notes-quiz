# Multilingual Lecture-to-Notes and Quiz Generator

Upload a recording of a lecture in Hindi, Kannada or Tamil (or paste its transcript), and get:

1. the transcript,
2. an English translation,
3. 4 to 6 short English notes,
4. five multiple-choice questions with the answer marked.

Everything runs locally, apart from a one-off download of the models.

## Problem

Lecture recordings in Indian languages are hard to study from: students need notes and practice questions, and most tools only handle English. This project checks how far open Indic speech and language models can take that task, and where they fail.

## Pipeline and model choices

| Step | Model | Why |
|---|---|---|
| Speech to text | **SraVaani-1.0** (ARTPARK-IISc, MIT) | Open Indic ASR that covers Hindi, Kannada and Tamil. Used as the first step because it was the model we had to hand. The brief's IndicConformer was not tested. |
| Translation to English | **IndicTrans2 1B** (AI4Bharat, MIT) | AI4Bharat's own translation model for Indic languages. It translates medical and physics terms much better than the general-purpose NLLB model I tried first. |
| Notes and quiz | **Qwen2.5-1.5B-Instruct** (Alibaba, Apache-2.0) | Small enough to run on a 16 GB Mac. Its larger sibling Qwen2.5-3B has a research-only licence, so I did not use it. |
| Interface | Gradio | Quick upload and paste, works in a browser. |

Models I tried and dropped:
- **BharatGen Param-1 2.9B**: custom model code failed with transformers 5.x, and generation with the cache disabled stalled on this Mac.
- **NLLB-200 600M** (Meta, CC-BY-NC): translation was poor on domain terms, so IndicTrans2 replaced it.
- **Translating into Hindi** (having the model write Hindi notes): the JSON output broke, so notes and quiz are written in English.

## Setup

Requires macOS on Apple Silicon and Python 3.11.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

python3.11 -m venv .venv-translate
.venv-translate/bin/python -m pip install -r requirements-translate.txt

sh scripts/fetch_sravaani.sh
```

- `scripts/fetch_sravaani.sh` downloads the SraVaani source files at a pinned commit. The repository has no licence file, so its code is not included here.
- IndicTrans2 is a gated model. Accept its terms on Hugging Face (`ai4bharat/indictrans2-indic-en-1B`) and log in with `.venv/bin/hf auth login` before the first run.
- The first run downloads SraVaani (about 900 MB), Qwen2.5-1.5B (about 3 GB) and IndicTrans2 (about 4 GB).

## Run the app

```sh
.venv/bin/python app.py
```

Open the printed address. Choose the language, then upload a WAV file or paste a transcript. A pasted transcript is used instead of the recording, which is useful for fixing a recognition error before the quiz is made.

Processing takes about one to two minutes for a few minutes of audio on this machine. Speech recognition runs on the CPU because the Apple GPU backend fails when SraVaani loads.

## How it works

| Step | File | What it does |
|---|---|---|
| Speech recognition | `lecture_notes/asr.py` | Resamples to 16 kHz, cuts the audio into chunks of about 8 seconds at quiet points, and transcribes each chunk with SraVaani. |
| Translation | `lecture_notes/translate.py`, `scripts/translate_worker.py` | Calls IndicTrans2 in a separate process. The worker runs in `.venv-translate`, because IndicTrans2 needs transformers 4.56 and SraVaani needs 5.x. |
| Notes and quiz | `lecture_notes/quiz.py` | Asks Qwen for JSON with notes and five questions. Checks the structure, and retries with the reason for any rejection. |
| Interface | `app.py` | Gradio page. |
| Evaluation | `scripts/evaluate_pipeline.py` | Runs the full pipeline on six test lectures and writes `results/final/`. |
| Speech evaluation | `scripts/evaluate_asr.py` | Word and character error rates on FLEURS test clips. Needs the FLEURS data, which is not in the repo. |

### Why 8-second chunks

SraVaani drops the opening words of clips longer than about 10 seconds. On a 22-second test recording, the first 10 seconds of speech went missing. Short chunks cut at quiet points fixed it.

## Results

### Test data

- **Six synthetic Hindi lectures** (`data/lectures/`). I wrote the texts, checked the facts, and recorded them with macOS `say` (voice Lekha). The original text is the reference for speech recognition and for checking the quiz.
- **FLEURS test clips** (Google, CC-BY-4.0): 40 random clips per language, for speech recognition only.

### Speech recognition

| Test set | Word error rate | Character error rate |
|---|---|---|
| Lecture 1, Newton's laws | 7.1% | 2.2% |
| Lecture 2, Photosynthesis | 2.4% | 0.7% |
| Lecture 3, Water cycle | 1.4% | 0.3% |
| Lecture 4, Indian Constitution | 2.9% | 1.1% |
| Lecture 5, Human heart | 10.4% | 3.2% |
| Lecture 6, Computer functions | 6.8% | 2.2% |
| **Average on lectures** | **5.2%** | |
| FLEURS Hindi (40 clips) | 9.1% | 3.1% |
| FLEURS Kannada (40 clips) | 25.5% | 8.1% |
| FLEURS Tamil (40 clips) | 20.2% | 5.8% |

The Kannada and Tamil results are on read speech, not lecture audio, and I did not test them on lectures.

### Translation

I did not score the translation against a reference. On the heart lecture, IndicTrans2 gave "atrium" and "ventricles" correctly, where the earlier NLLB model gave "allim" and "suction chamber". Its one clear error on that lecture came from the speech recognition step ("आलिम" for "आलिंद"), which the translation then carried through.

### Notes and quiz

| | Result |
|---|---|
| Lectures that produced a valid five-question quiz | 6 of 6 |
| Lectures that needed a retry | 4 of 6 (one retry each) |
| Questions with the correct answer marked | 26 of 30 (87%) |
| Questions with an ambiguous answer | 2 of 30 |
| Questions with a wrong answer | 2 of 30 |
| Questions fully supported by the lecture text | 19 of 30 |
| Questions partly supported | 8 of 30 |
| Questions not supported by the lecture text | 3 of 30 |

The review was done by one reviewer, the author, who checked each question against the lecture text and general knowledge. The labels are in `results/final/quiz_review.csv`. This is a manual judgement, not an automatic score.

### Comparison with earlier versions

| Version | Quizzes valid | Correct answers | Supported by lecture |
|---|---|---|---|
| v1: Hindi transcript straight into Qwen | 6 of 6 | 21 of 30 (70%) | 12 of 30 |
| v2: NLLB translation, evidence check | 5 of 6 | 17 of 25 (68%) | 16 of 25 |
| **Final: IndicTrans2 translation** | **6 of 6** | **26 of 30 (87%)** | **19 of 30** |

v1 and v2 results are in `results/v1/`. v2 was dropped; its results are not kept.

## Failure analysis

**1. Wrong answers.** Two questions had the wrong answer marked:
- Lecture 5: "What happens to the blood when it leaves the right side of the heart?" marked "It mixes with oxygen-depleted blood in the lungs". The right side sends blood to the lungs without mixing it.
- Lecture 6: "How can you tell that a disk has been lost?" has no answer in the lecture, and the question makes no sense.

**2. Ambiguous questions.** Two questions had more than one defensible answer:
- Lecture 1: "Which statement correctly describes Newton's first law?" Options A and C are both true.
- Lecture 1: "If you push a box..." Friction is also a valid explanation, and the lecture does not discuss friction.

**3. Facts not in the lecture.** Eight questions were only partly supported, and three were not supported at all. Examples: the stomata detail in the water-cycle quiz, and the "runoff" term. The model often drew on general knowledge, which can be right but is not what the lecture said.

**4. Garbled terms.** Lecture 2 asked "What does fluorokhyl absorb?". The model misspelled "chlorophyll". A student could be confused by this even though the answer is right.

**5. Speech recognition errors.** The heart lecture had the most recognition errors (10.4% WER). These mostly affect spelling, but a misheard key term such as "आलिंद" (atrium) reaches the translation and the quiz.

**6. Translation of technical terms.** Translation is only as good as the transcript it receives. Errors in the Hindi transcript appear in the English text.

## Limits

- **Synthetic lectures.** The six lectures were written for this project and read by a text-to-speech voice. Real lectures are noisier, faster and less formal, so the error rates will be higher.
- **Small sample.** Six lectures and 30 questions. The brief asks for 30 to 50 cases, so this is at the minimum.
- **One reviewer.** I wrote the lectures and reviewed the quiz. A second reviewer should check the labels.
- **Small quiz model.** Qwen2.5-1.5B makes factual mistakes and sometimes produces wrong or ambiguous questions. A larger model would likely help, but its licence needs checking.
- **Kannada and Tamil** were tested only on read speech, and the pipeline was tested end to end on Hindi only.
- **Speed.** Speech recognition runs at about real time on the CPU, so a one-hour lecture takes about an hour.
- **Gated model.** IndicTrans2 needs an account that has accepted its terms on Hugging Face.

## Files

```
app.py                         Gradio interface
lecture_notes/asr.py          SraVaani speech recognition, with chunking
lecture_notes/translate.py    IndicTrans2 wrapper (runs the worker in .venv-translate)
lecture_notes/quiz.py         Qwen notes and quiz generation, with validation and retries
scripts/translate_worker.py   IndicTrans2 translation (runs in .venv-translate)
scripts/fetch_sravaani.sh     Downloads the SraVaani source files at a pinned commit
scripts/evaluate_pipeline.py  Full pipeline evaluation on the six lectures
scripts/evaluate_asr.py       Speech evaluation on FLEURS clips
data/lectures/                Lecture texts (lectures.json) and synthesised audio
results/final/                Final evaluation, including quiz_review.csv
results/v1/                   Earlier evaluation, for comparison
results/asr_*.csv             FLEURS speech recognition results
requirements.txt              Dependencies for the main environment
requirements-translate.txt    Dependencies for the translation environment
```
