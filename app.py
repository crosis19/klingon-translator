import gradio as gr

from klingon_translator.model.translator import KlingonTranslator

# Load model once at startup (auto-detects models/nllb-klingon-extended,
# then falls back to base NLLB-200)
translator = KlingonTranslator()


def translate(text: str, direction: str) -> str:
    """Translate text based on selected direction."""
    if not text.strip():
        return ""
    if direction == "English → Klingon":
        return translator.to_klingon(text)
    return translator.to_english(text)


EXAMPLES = [
    ["Today is a good day to die.", "English → Klingon",
     "Heghlu'meH QaQ jajvam."],
    ["Qapla'!", "Klingon → English",
     "Success!"],
    ["Where is the bathroom?", "English → Klingon",
     "nuqDaq 'oH puchpa''e'?"],
    ["tlhIngan maH!", "Klingon → English",
     "We are Klingons!"],
    ["What do you want?", "English → Klingon",
     "nuqneH?"],
    ["Revenge is a dish best served cold.", "English → Klingon",
     "bortaS bIr jablu'DI' reH QaQqu' nay'."],
    ["qatlho'.", "Klingon → English",
     "Thank you."],
    ["I don't understand.", "English → Klingon",
     "jIyajbe'."],
]

with gr.Blocks(title="Klingon Translator") as demo:
    gr.Markdown(
        "# Klingon Translator\n"
        "English ↔ Klingon translation powered by fine-tuned NLLB-200."
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input",
                placeholder="Enter text to translate...",
                lines=3,
            )
            direction = gr.Radio(
                ["English → Klingon", "Klingon → English"],
                label="Direction",
                value="English → Klingon",
            )
            translate_btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            output = gr.Textbox(
                label="Model Output", lines=3
            )

    # Hidden textbox so the "Expected" column appears in the examples table
    expected = gr.Textbox(visible=False)

    translate_btn.click(
        fn=translate,
        inputs=[text_input, direction],
        outputs=output,
    )
    text_input.submit(
        fn=translate,
        inputs=[text_input, direction],
        outputs=output,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, direction, expected],
        outputs=output,
        fn=translate,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
