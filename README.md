# Multilingual Lecture-to-Notes and Quiz Generator

Upload a recording of a lecture in Hindi, Kannada or Tamil (or paste its transcript), and get:

1. the transcript,
2. an English translation,
3. 4 to 6 short English notes,
4. five multiple-choice questions with the answer marked.

Everything runs locally, apart from a one-off download of the models.

A sample run on one test lecture is in [results/demo/lecture_04.md](results/demo/lecture_04.md).

## Problem

Lecture recordings in Indian languages are hard to study from: students need notes and practice questions, and most tools only handle English. This project checks how far open Indic speech and language models can take that task, and where they fail.

## Pipeline and model choices

| Step | Model | Why |
|---|---|---|
| Speech to text | **SraVaani-1.0** (ARTPARK-IISc, MIT) | Open Indic ASR that covers Hindi, Kannada and Tamil. It had the lowest error rate of the two models I tested on the lectures (see the comparison below). |
| Translation to English | **IndicTrans2 1B** (AI4Bharat, MIT) | AI4Bharat's own translation model for Indic languages. It translates medical and physics terms much better than the general-purpose NLLB model I tried first. |
| Notes and quiz | **Qwen2.5-1.5B-Instruct** (Alibaba, Apache-2.0) | Small enough to run on a 16 GB Mac. Its larger sibling Qwen2.5-3B has a research-only licence, so I did not use it. |
| Interface | Gradio | Quick upload and paste, works in a browser. |

Models I tried and dropped:
- **BharatGen Param-1 2.9B**: custom model code failed with transformers 5.x, and generation with the cache disabled stalled on this Mac.
- **NLLB-200 600M** (Meta, CC-BY-NC): translation was poor on domain terms, so IndicTrans2 replaced it.
- **Translating into Hindi** (having the model write Hindi notes): the JSON output broke, so notes and quiz are written in English.

### Deviations from the brief

| Brief | This project | Why |
|---|---|---|
| IndicConformer or BharatGen speech model for ASR | SraVaani-1.0 | IndicConformer was tested on the same ten lectures and had higher error rates (see Speech recognition). Its model card now recommends Indic-Transcribe, which I did not test. |
| BharatGen Param1 for the LLM | Qwen2.5-1.5B-Instruct | Param-1 did not run on this Mac (see above). |
| Real lecture recordings | Ten synthetic lectures, all read by a text-to-speech voice | The project has no real lecture recordings. Public speech data (FLEURS) is used for the speech error rates. |
| Second human reviewer | Not done | Only one reviewer, an AI model (Claude), labelled the quiz questions. See Limits. |

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

To regenerate the demo file:

```sh
.venv/bin/python scripts/make_demo.py lecture_04
```

## How it works

| Step | File | What it does |
|---|---|---|
| Speech recognition | `lecture_notes/asr.py` | Resamples to 16 kHz, cuts the audio into chunks of about 8 seconds at quiet points, and transcribes each chunk with SraVaani. |
| Translation | `lecture_notes/translate.py`, `scripts/translate_worker.py` | Calls IndicTrans2 in a separate process. The worker runs in `.venv-translate`, because IndicTrans2 needs transformers 4.56 and SraVaani needs 5.x. |
| Notes and quiz | `lecture_notes/quiz.py` | Asks Qwen for JSON with notes and five questions. Checks the structure, and retries with the reason for any rejection. |
| Interface | `app.py` | Gradio page. |
| Demo | `scripts/make_demo.py` | Runs the app's processing on one lecture and saves the output to `results/demo/`. |
| Evaluation | `scripts/evaluate_pipeline.py` | Runs the full pipeline on the ten test lectures and writes the results in `results/`. |
| Speech evaluation | `scripts/evaluate_asr.py` | Word and character error rates on FLEURS test clips. Needs the FLEURS data, which is not in the repo. |

### Why 8-second chunks

SraVaani drops the opening words of clips longer than about 10 seconds. On a 22-second test recording, the first 10 seconds of speech went missing. Short chunks cut at quiet points fixed it.

## Results

### Test data

- **Ten synthetic Hindi lectures** (`data/lectures/`). I wrote the texts, checked the facts, and recorded them with macOS `say` (voice Lekha). The original text is the reference for speech recognition and for checking the quiz.
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
| Lecture 7, Earth's layers | 15.9% | 4.2% |
| Lecture 8, Blood components | 16.4% | 5.7% |
| Lecture 9, Solar system | 5.3% | 2.6% |
| Lecture 10, Energy conservation | 1.3% | 0.3% |
| **Average on lectures** | **7.0%** | **2.2%** |
| FLEURS Hindi (40 clips) | 9.1% | 3.1% |
| FLEURS Kannada (40 clips) | 25.5% | 8.1% |
| FLEURS Tamil (40 clips) | 20.2% | 5.8% |

The Kannada and Tamil results are on read speech, not lecture audio, and I did not test them on lectures.

### SraVaani compared with IndicConformer

I also ran IndicConformer-600M (AI4Bharat, MIT, the brief's named model) on the same ten lectures, scored the same way (`scripts/evaluate_indicconformer.py`, output in `results/asr_indicconformer.csv`). IndicConformer was given each whole recording, without the chunking that SraVaani uses.

| Model | Word error rate, average | Character error rate, average | Lectures where it has the lowest WER (ties counted for each) |
|---|---|---|---|
| **SraVaani-1.0** (used) | **7.0%** | **2.2%** | **7 of 10** |
| IndicConformer, RNNT decoder | 9.3% | 2.5% | 4 of 10 |
| IndicConformer, CTC decoder | 14.2% | 6.9% | 1 of 10 |

The per-lecture numbers are in `results/asr_indicconformer.csv`. IndicConformer's CTC decoder dropped or changed many words on lectures 2 and 6, and its RNNT decoder did worse than SraVaani on lectures 7 and 8. This is one set of synthetic lectures, so it does not show which model is better in general.

### Translation

I did not score the translation against a reference. On the heart lecture, IndicTrans2 gave "atrium" and "ventricles" correctly, where the earlier NLLB model gave "allim" and "suction chamber". Its one clear error on that lecture came from the speech recognition step ("आलिम" for "आलिंद"), which the translation then carried through.

### Notes and quiz

Ten lectures, 50 questions (five per lecture). Every lecture produced a valid quiz.

| | Result |
|---|---|
| Lectures that produced a valid five-question quiz | 10 of 10 |
| Lectures that needed a retry | 6 of 10 (one retry each) |
| Questions with the correct answer marked | 38 of 50 (76%) |
| Questions with an ambiguous answer | 7 of 50 |
| Questions with a wrong answer | 5 of 50 |
| Questions fully supported by the lecture text | 38 of 50 (76%) |
| Questions partly supported | 9 of 50 |
| Questions not supported by the lecture text | 3 of 50 |

The labels are in `results/quiz_review.csv`. They were made by one reviewer, an AI model (Claude), who checked each question against the original lecture text and general knowledge. This is a manual judgement, not an automatic score, and it has not been checked by a human. Counts above come from that file.

### Comparison with earlier versions

| Version | Lectures | Quizzes valid | Correct answers | Supported by lecture |
|---|---|---|---|---|
| v1: Hindi transcript straight into Qwen | 6 | 6 of 6 | 21 of 30 (70%) | 12 of 30 |
| v2: NLLB translation, evidence check | 6 | 5 of 6 | 17 of 25 (68%) | 16 of 25 |
| v3: IndicTrans2 translation, six lectures | 6 | 6 of 6 | 26 of 30 (87%) | 19 of 30 |
| **Final: IndicTrans2 translation, ten lectures** | **10** | **10 of 10** | **38 of 50 (76%)** | **38 of 50 (76%)** |

v1 and v2 results are in `results/v1/`. The six-lecture v3 run is in `results/six_lectures/`. v2 was dropped; its results are not kept.

The v3 row (87%) comes from an earlier review of the same six lectures. When I re-labelled those six lectures for this table, I was stricter about ambiguous and unsupported answers, and they score 23 of 30 (77%). The two reviews are therefore not directly comparable. The final row uses the stricter labels throughout.

## Failure analysis

**1. Wrong answers (5 of 50).**
- Lecture 1: "If you push a box..." marks "Your push causes the box to accelerate due to its inertia". The push does cause acceleration, but "due to its inertia" is wrong.
- Lecture 5: "What happens to the blood when it leaves the right side of the heart?" marks "It mixes with oxygen-depleted blood in the lungs". The lecture says the two sides do not mix.
- Lecture 6: "How can you tell that a disk has been lost?" has no answer in the lecture, and the question makes no sense.
- Lecture 8: "What gives blood its characteristic yellow color?" marks "Hemoglobin". The lecture says plasma is yellow, and no option names it.
- Lecture 9: "Which day is referred to as the Red Planet?" marks "Tuesday". This comes from a transcript error: "मंगल" (Mars) was heard as "मंगलवार" (Tuesday), so the translation said "Tuesday is called the red planet".

**2. Ambiguous questions (7 of 50).** More than one option is defensible:
- Lecture 1: Newton's second law and the first law questions each have two defensible options.
- Lecture 6: the printer and the monitor both show results; the disk-type question has USB and floppy disks as permanent storage too.
- Lecture 7: the mantle is also the thickest layer, which is true but not in the lecture.
- Lecture 9: "revolution" and "orbit" are both correct terms.
- Lecture 10: electricity and light also convert into each other (solar cells).

**3. Facts not in the lecture (3 not supported, 9 partly).** Examples: the "stomata" detail and the "runoff" term in the water-cycle quiz, and the solar-cell location in the energy quiz (the lecture never says where solar cells are). The model often drew on general knowledge, which can be right but is not what the lecture said.

**4. Garbled terms.** Lecture 2 asked "What does fluorokhyl absorb?". The model misspelled "chlorophyll" and the translation step carried the misspelling through. A student could be confused by this even though the answer is right.

**5. Speech recognition errors reaching the quiz.** The heart lecture had "आलिम" for "आलिंद" (atrium). The solar-system lecture had "बुधवार" (Wednesday) for "बुध" (Mercury) and "मंगलवार" (Tuesday) for "मंगल" (Mars). The Mercury question was still answered correctly, but only from general knowledge, because the translated transcript said "Wednesday". The Mars question was not, and was marked wrong.

**6. Translation of technical terms.** Translation is only as good as the transcript it receives. Errors in the Hindi transcript appear in the English text, and from there in the quiz.

**7. Retries.** Six of the ten lectures needed one retry because the first output failed the structure check. The retry feeds the rejected output and the reason back to the model, because a plain retry with greedy decoding repeats the same output.

## Limits

- **Synthetic lectures.** The ten lectures were written for this project and read by a text-to-speech voice. Real lectures are noisier, faster and less formal, so the error rates will be higher.
- **Small sample.** Ten lectures and 50 questions. The brief asks for 30 to 50 cases, so this is at the top of the range.
- **One reviewer, and it is not a human.** The quiz labels were made by an AI model (Claude) and have not been checked by a human. The brief asks for a second reviewer. A human check of the 50 labels is still needed.
- **Small quiz model.** Qwen2.5-1.5B makes factual mistakes and sometimes produces wrong or ambiguous questions. A larger model would likely help, but its licence needs checking.
- **Kannada and Tamil** were tested only on read speech (FLEURS), and the pipeline was tested end to end on Hindi only.
- **Speed.** Speech recognition runs at about real time on the CPU, so a one-hour lecture takes about an hour.
- **Gated model.** IndicTrans2 needs an account that has accepted its terms on Hugging Face.
- **No demo video.** The demo is a saved run of the pipeline (`results/demo/`), not a recording of the Gradio page.

## Files

```
app.py                         Gradio interface
lecture_notes/asr.py          SraVaani speech recognition, with chunking
lecture_notes/translate.py    IndicTrans2 wrapper (runs the worker in .venv-translate)
lecture_notes/quiz.py         Qwen notes and quiz generation, with validation and retries
scripts/translate_worker.py   IndicTrans2 translation (runs in .venv-translate)
scripts/fetch_sravaani.sh     Downloads the SraVaani source files at a pinned commit
scripts/evaluate_pipeline.py  Full pipeline evaluation on the ten lectures
scripts/evaluate_asr.py       Speech evaluation on FLEURS clips
scripts/evaluate_indicconformer.py  IndicConformer comparison on the ten lectures
scripts/make_demo.py          Saves a demo run of the app for one lecture
data/lectures/                Lecture texts (lectures.json) and synthesised audio
results/pipeline_summary.csv  Per-lecture WER, CER, quiz validity and timings
results/quiz_review.csv       All 50 questions with reviewer labels
results/pipeline_transcripts.json  Transcript and English translation per lecture
results/demo/                 Demo run output
results/six_lectures/         Six-lecture evaluation, for comparison
results/v1/                   Earlier evaluation, for comparison
results/asr_*.csv             FLEURS speech recognition results
results/asr_indicconformer.csv IndicConformer results on the ten lectures
requirements.txt              Dependencies for the main environment
requirements-translate.txt    Dependencies for the translation environment
```
