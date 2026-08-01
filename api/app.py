"""Gradio demo over the same predict() core: upload math image -> LaTeX + preview.

    python -m api.app                  # local (add share=True for a temporary link)
Hugging Face Spaces: set `app_file: api/app.py` in the Space README frontmatter.
"""
import io

import gradio as gr
from PIL import Image

from api.predict import predict_latex
from api.render import render_latex_png


def infer(image):
    if image is None:
        return "", None, ""
    result = predict_latex(image)
    latex = result["latex"]
    try:
        preview = Image.open(io.BytesIO(render_latex_png(latex)))
    except Exception:
        preview = None                       # rendering is best-effort
    return latex, preview, f"{result['confidence']:.1%}"


demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil", label="Handwritten or printed math"),
    outputs=[
        gr.Textbox(label="Predicted LaTeX"),
        gr.Image(label="Rendered preview"),
        gr.Textbox(label="Confidence"),
    ],
    title="scribe2tex — Handwritten Math → LaTeX",
    description="ResNet-18 + Bahdanau attention + LSTM, trained on im2latex (BLEU 91.5) "
                "and fine-tuned on MathWriting (BLEU 79.4). Upload an image of a math "
                "expression to get its LaTeX.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
