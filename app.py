"""Gradio app: lecture audio (or a transcript) in, English notes and a five-question quiz out."""
import gradio as gr

from lecture_notes import asr, quiz, translate


def _format_notes(notes):
    return "\n".join("- %s" % note for note in notes)


def _format_quiz(questions):
    blocks = []
    for number, q in enumerate(questions, 1):
        lines = ["**%d. %s**" % (number, q["question"])]
        for letter in quiz.OPTIONS:
            lines.append("- %s) %s" % (letter, q["options"][letter]))
        lines.append("\n*Answer: %s*" % q["answer"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def process(audio_path, transcript_text, language):
    """Return (transcript, english, notes_md, quiz_md, status).

    A pasted transcript takes priority over the recording. The transcript is in `language`;
    the notes and quiz are made from its English translation.
    """
    transcript = (transcript_text or "").strip()
    if not transcript:
        if not audio_path:
            return "", "", "", "", "Upload a lecture recording or paste a transcript."
        transcript = asr.transcribe(audio_path)
        if not transcript:
            return "", "", "", "", "No speech was recognised in this recording. Try another file or paste a transcript."

    english = translate.translate(transcript, language)
    data, errors = quiz.generate_notes_and_quiz(english)
    if data is None:
        return transcript, english, "", "", "Could not produce a valid quiz after %d attempts: %s" % (
            len(errors), "; ".join(errors))

    status = "Done. Check the transcript and translation before you rely on the quiz; the model can make mistakes."
    return transcript, english, _format_notes(data["notes"]), _format_quiz(data["questions"]), status


with gr.Blocks(title="Lecture Notes and Quiz") as demo:
    gr.Markdown(
        "# Lecture Notes and Quiz\n"
        "Upload a Hindi, Kannada or Tamil lecture recording, or paste its transcript. "
        "You get notes and five multiple-choice questions with answers.\n\n"
        "Speech recognition uses SraVaani-1.0. Translation uses IndicTrans2 (1B). Notes and quiz use "
        "Qwen2.5-1.5B-Instruct, a small model that sometimes gets facts wrong, so check the output."
    )
    with gr.Row():
        audio = gr.Audio(sources=["upload"], type="filepath", label="Lecture recording (WAV)")
        transcript_box = gr.Textbox(
            label="Transcript (optional: paste or correct; used instead of the recording if filled in)",
            lines=8,
        )
    language = gr.Dropdown(choices=list(translate.LANGUAGE_CODES), value="Hindi", label="Language of the lecture")
    run = gr.Button("Generate notes and quiz", variant="primary")
    status = gr.Markdown()
    transcript_out = gr.Textbox(label="Transcript", lines=5, interactive=False)
    english_out = gr.Textbox(label="English translation (the notes and quiz are made from this)", lines=5, interactive=False)
    with gr.Row():
        notes_out = gr.Markdown(label="Notes")
        quiz_out = gr.Markdown(label="Quiz")

    run.click(
        process,
        inputs=[audio, transcript_box, language],
        outputs=[transcript_out, english_out, notes_out, quiz_out, status],
    )

if __name__ == "__main__":
    demo.launch()
