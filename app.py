import gradio as gr

from klingon_translator.model.translator import KlingonTranslator

# Load model once at startup (auto-detects models/nllb-klingon-extended,
# then falls back to base NLLB-200)
translator = KlingonTranslator()


def translate(text: str, direction: str) -> str:
    if not text.strip():
        return ""
    if direction == "English → Klingon":
        return translator.to_klingon(text)
    return translator.to_english(text)


demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Input", placeholder="Enter text to translate...", lines=3),
        gr.Radio(
            ["English → Klingon", "Klingon → English"],
            label="Direction",
            value="English → Klingon",
        ),
    ],
    outputs=gr.Textbox(label="Translation", lines=3),
    title="Klingon Translator",
    description="English ↔ Klingon translation powered by fine-tuned NLLB-200.",
    examples=[
        ["Today is a good day to die.", "English → Klingon"],
        ["Qapla'", "Klingon → English"],
        ["Where is the bathroom?", "English → Klingon"],
        ["tlhIngan maH!", "Klingon → English"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
