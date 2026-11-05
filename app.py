"""Gradio app: lecture audio (or a transcript) in, English notes and a five-question quiz out."""
import os

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


FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700;800'
    '&family=Hanken+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">'
)
STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")

with gr.Blocks(title="Lecture Notes and Quiz", fill_width=True) as demo:
    gr.HTML(
        '<header class="hero-app">'
        '<h1>Lecture in. <mark>Notes and quiz out.</mark></h1>'
        '<p class="lede">Upload a Hindi, Kannada or Tamil lecture recording, or paste its transcript. '
        'You get notes and five multiple-choice questions with answers.</p>'
        '<ul class="chips">'
        '<li>SraVaani-1.0 speech</li>'
        '<li>IndicTrans2 translation</li>'
        '<li>Qwen2.5-1.5B notes and quiz</li>'
        '</ul>'
        '</header>'
    )
    with gr.Row(equal_height=False, elem_id="panels"):
        with gr.Column(scale=5, elem_id="input-panel"):
            gr.Markdown("### Your lecture", elem_classes="panel-title")
            audio = gr.Audio(sources=["upload"], type="filepath", label="Recording (WAV)")
            transcript_box = gr.Textbox(
                label="Transcript, optional",
                info="Paste or correct it here. A filled-in transcript is used instead of the recording.",
                lines=6,
            )
            language = gr.Dropdown(choices=list(translate.LANGUAGE_CODES), value="Hindi", label="Language")
            run = gr.Button("Generate notes and quiz", variant="primary", size="lg")

        with gr.Column(scale=7, elem_id="result-panel"):
            gr.Markdown("### Result", elem_classes="panel-title")
            status = gr.Markdown(
                "Ready. Upload a recording or paste a transcript to begin.", elem_classes="status-line"
            )
            with gr.Tabs():
                with gr.Tab("Notes"):
                    notes_out = gr.Markdown(elem_classes="result")
                with gr.Tab("Quiz"):
                    quiz_out = gr.Markdown(elem_classes="result")
                with gr.Tab("English"):
                    english_out = gr.Textbox(
                        show_label=False, lines=8, interactive=False,
                        placeholder="The English translation appears here. Notes and quiz are made from it.",
                    )
                with gr.Tab("Transcript"):
                    transcript_out = gr.Textbox(
                        show_label=False, lines=8, interactive=False,
                        placeholder="The transcript in the lecture's language appears here.",
                    )

    gr.Markdown(
        "Qwen2.5-1.5B is a small model and sometimes gets facts wrong. Check the notes and questions "
        "against the lecture before you rely on them.",
        elem_classes="footnote",
    )

    run.click(
        process,
        inputs=[audio, transcript_box, language],
        outputs=[transcript_out, english_out, notes_out, quiz_out, status],
    )

if __name__ == "__main__":
    demo.launch(css_paths=[STYLE], head=FONTS)
