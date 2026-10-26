"""Gradio demo over the same predict() core: upload math image -> LaTeX + preview.

    python -m api.app                  # local (add share=True for a temporary link)
Hugging Face Spaces: set `app_file: api/app.py` in the Space README frontmatter.

============================================================================
DIRECTION CONTRACT  (impeccable seed key: 8ea9fce7 · form: one-bit desktop,
challenger #5 of the direction roll · chosen over assigned #7 lineprinter)
----------------------------------------------------------------------------
THESIS: A deliberately *classic* model deserves a classic machine. The demo is
  an early one-bit Macintosh desktop, not another dark-neon ML playground.
OWN-WORLD: Black & white pixels only; ordered dither for every grey; racing-
  stripe title bars with a centred name plaque and a close box; Chicago-style
  pixel type (Silkscreen) for chrome, Monaco/Courier mono for code; 1px black
  borders, hard offset shadows, square corners; invert-on-press; barber-pole
  progress; marching-ants on a fresh result.
STORY: Visitor sees a familiar desktop, drops a math image into the window,
  presses Convert, watches the LaTeX print into a Monaco field with a calibrated
  confidence bar; a Read Me window carries the honest benchmark record.
FIRST VIEWPORT: Menu bar across the top; a 50%-dither desktop; the "scribe2tex"
  window centred, input well + bold default Convert button on the left, LaTeX /
  preview / confidence stacked on the right. Primary action is the Convert button.
FORM: One-bit desktop (Susan-Kare-era Mac OS), challenger #5; seed 8ea9fce7.
FINISH: unreviewed and undocumented is unfinished; this build ends with the
  finish review, the verdict, and DESIGN.md.
============================================================================
"""
import io
import os

# Full-MathWriting checkpoint (20 epochs): ExpRate 0.445 vs 0.065 for hme100k_best on a
# 200-image MathWriting test slice. Set MODEL_PATH=checkpoints/hme100k_best.pth for photos.
os.environ.setdefault("MODEL_PATH", "checkpoints/mwfull20_best.pth")

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from api.predict import predict_latex  # noqa: E402
from api.render import render_latex_png  # noqa: E402

EXAMPLES = [
    ["api/examples/frac.png"],
    ["api/examples/sqrt.png"],
    ["api/examples/integral.png"],
]

BENCHMARKS = [
    ("1  im2latex (printed)", "91.5", "0.47"),
    ("2  MathWriting (hand)", "79.4", "0.32"),
    ("3  HME100K (photos)*", "88.3", "0.52"),
]


def _confidence_html(conf: float) -> str:
    """Classic Mac barber-pole progress bar + honest low-confidence note."""
    pct = max(0, min(100, round(conf * 100)))
    warn = (
        '<div class="mac-note">&#9888; low &mdash; try a tighter crop, darker ink</div>'
        if conf < 0.60 else
        '<div class="mac-note">&#10003; reads clean</div>'
    )
    return (
        '<div class="mac-meter-row">'
        f'  <div class="mac-meter"><div class="mac-meter-fill" style="width:{pct}%"></div></div>'
        f'  <div class="mac-meter-pct">{pct}%</div>'
        "</div>"
        f"{warn}"
    )


def infer(image):
    """image (PIL) -> (latex, preview_image_or_None, confidence_html)."""
    if image is None:
        return "", None, '<div class="mac-note">no document &mdash; drop an image above</div>'
    result = predict_latex(image)
    latex = result["latex"]
    try:
        preview = Image.open(io.BytesIO(render_latex_png(latex)))
    except Exception:
        preview = None  # rendering is best-effort
    return latex, preview, _confidence_html(result["confidence"])


# --- one-bit desktop chrome -------------------------------------------------

def _titlebar(title: str, close: bool = True) -> str:
    box = '<span class="mac-close"></span>' if close else ""
    return (
        f'<div class="mac-title">{box}'
        f'<span class="mac-title-name">{title}</span></div>'
    )


APPLE = (
    '<svg viewBox="0 0 14 16" width="13" height="15" aria-hidden="true">'
    '<path fill="currentColor" d="M9.6 0c.1 1-.3 1.9-.9 2.6-.6.7-1.6 1.2-2.5 1.1-.1-1 .4-2 1-2.6C7.8.5 8.8.1 9.6 0zM12 11.6c-.3.8-.5 1.1-.9 1.8-.6.9-1.5 2.1-2.6 2.1-.9 0-1.2-.6-2.5-.6s-1.6.6-2.5.6c-1.1 0-1.9-1-2.6-2C-.4 11.4-.6 8 .8 6.2c.7-.9 1.8-1.5 2.9-1.5 1.1 0 1.8.6 2.7.6.9 0 1.4-.6 2.7-.6.9 0 1.9.5 2.6 1.4-2.3 1.3-1.9 4.5.3 5.5z"/></svg>'
)

MENUBAR = (
    '<div class="mac-menubar">'
    f'<span class="mac-menu-apple">{APPLE}</span>'
    '<span class="mac-menu-item mac-menu-app">scribe2tex</span>'
    '<span class="mac-menu-item">File</span>'
    '<span class="mac-menu-item">Edit</span>'
    '<span class="mac-menu-item">View</span>'
    '<span class="mac-menu-item">Special</span>'
    '<span class="mac-menu-clock">image &#8594; LaTeX</span>'
    "</div>"
)

BENCH_ROWS = "".join(
    f'<tr><td>{name}</td><td class="num">{bleu}</td><td class="num">{exp}</td></tr>'
    for name, bleu, exp in BENCHMARKS
)
READ_ME = (
    '<div class="mac-readme">'
    "<p><b>scribe2tex</b> reads an image of a math expression and writes its LaTeX. "
    "The engine is a <b>from-scratch classic vision&#8594;sequence model</b> &mdash; "
    "a ResNet-18 encoder, Bahdanau attention, and an LSTM decoder with beam search &mdash; "
    "built the old way on purpose, so it can be benchmarked against the modern transformer.</p>"
    '<table class="mac-table"><thead><tr><th>Stage &amp; test set</th>'
    '<th class="num">BLEU</th><th class="num">ExpRate</th></tr></thead>'
    f"<tbody>{BENCH_ROWS}</tbody></table>"
    '<p class="mac-fine">* HME100K on a self-carved held-out slice. Best-val checkpoint.</p>'
    "<p><b>It likes a tidy document.</b> Crop close to the expression, dark ink, shoot "
    "straight on. The confidence reading is honest &mdash; it falls when the model is "
    "guessing at a smudge and rises when it can actually see the strokes.</p>"
    "</div>"
)

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&family=Courier+Prime:wght@400;700&display=swap');

:root{
  --ink:#000; --paper:#fff;
  --pix:'Silkscreen', 'Chicago', system-ui, sans-serif;
  --mono:'Courier Prime', ui-monospace, Monaco, 'Menlo', monospace;
  /* 50% checkerboard "desktop" dither */
  --desk:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='2' height='2' fill='%23000'/%3E%3Crect x='2' y='2' width='2' height='2' fill='%23000'/%3E%3C/svg%3E");
}

/* every underlying surface IS the desktop — no black can show through on scroll */
html, body, gradio-app, .app, #root, .main, .wrapper{
  background:var(--paper) var(--desk) !important; background-size:4px 4px !important;
}
/* ---- reset Gradio's own surfaces so the world shows through ---- */
.gradio-container{
  max-width:100% !important; background:var(--paper) var(--desk) !important;
  background-size:4px 4px !important; font-family:var(--pix) !important;
  color:var(--ink) !important; padding:0 !important;
  /* neutralise Gradio's colour accents — this world is 1-bit */
  --color-accent:#000 !important; --color-accent-soft:#fff !important;
  --link-text-color:#000 !important; --body-text-color:#000 !important;
}
.gradio-container *{ border-radius:0 !important; box-shadow:none !important; }
/* pure black ink everywhere; white is re-applied explicitly on inverted parts */
.gradio-container p, .gradio-container span, .gradio-container div,
.gradio-container td, .gradio-container button, .gradio-container textarea,
.gradio-container label, .gradio-container a{ color:var(--ink); }
.gradio-container svg{ color:var(--ink) !important; fill:currentColor !important; stroke:currentColor; }
.gradio-container .block, .gradio-container .form, .gradio-container .panel,
.gradio-container .gap, .gradio-container .wrap{
  background:transparent !important; border:none !important; padding:0 !important;
}
footer, .show-api, .built-with{ display:none !important; }

/* image-rendering pixelation for that bitmap feel */
.gradio-container img{ image-rendering:pixelated; }

/* ---- menu bar ---- */
.mac-menubar{
  display:flex; align-items:center; gap:0; background:var(--paper);
  border-bottom:2px solid var(--ink); height:26px; padding:0 8px;
  font-family:var(--pix); font-size:11px; position:sticky; top:0; z-index:50;
}
.mac-menu-apple{ display:flex; align-items:center; margin-right:12px; }
.mac-menu-item{ padding:0 11px; line-height:26px; cursor:default; }
.mac-menu-app{ font-weight:700; }
.mac-menu-item:hover{ background:var(--ink); color:var(--paper); }
.mac-menu-clock{ margin-left:auto; font-size:10px; letter-spacing:.5px; }

/* ---- desktop workspace ---- */
#desktop{ padding:26px 20px 60px !important; display:flex; justify-content:center; }
#desk-cols{ width:100%; max-width:940px; gap:20px !important; }

/* ---- window chrome ---- */
.mac-window{
  background:var(--paper) !important; border:2px solid var(--ink) !important;
  box-shadow:5px 5px 0 rgba(0,0,0,1) !important; overflow:hidden;
}
.mac-title{
  position:relative; height:22px; border-bottom:2px solid var(--ink);
  display:flex; align-items:center; justify-content:center;
  background:repeating-linear-gradient(var(--ink) 0 1px, var(--paper) 1px 3px);
}
.mac-title-name{
  font-family:var(--pix); font-size:11px; background:var(--paper);
  padding:0 12px; line-height:16px; white-space:nowrap;
}
.mac-close{
  position:absolute; left:7px; top:5px; width:11px; height:11px;
  border:2px solid var(--ink); background:var(--paper);
}
.mac-body{ padding:16px !important; }

/* section labels */
.mac-label{
  font-family:var(--pix); font-size:10px; text-transform:uppercase;
  letter-spacing:1px; margin:0 0 6px !important; color:var(--ink) !important;
}

/* ---- image well (input + preview) ---- */
.mac-well .image-container, .mac-well{ background:var(--paper) !important; }
.mac-well{ border:2px solid var(--ink) !important; }
.mac-well [data-testid="image"] , .mac-well .upload-container{ background:var(--paper) !important; }
.mac-well .wrap.default{ font-family:var(--pix) !important; font-size:10px !important; color:#000 !important; }

/* ---- buttons: rounded-rect Mac buttons, default = double border ---- */
button.mac-btn, .mac-btn button{
  font-family:var(--pix) !important; font-size:12px !important; color:var(--ink) !important;
  background:var(--paper) !important; border:2px solid var(--ink) !important;
  border-radius:9px !important; padding:8px 18px !important; min-height:0 !important;
  box-shadow:2px 2px 0 rgba(0,0,0,1) !important; transition:none !important;
}
.mac-btn.default button, button.mac-default{
  outline:2px solid var(--ink); outline-offset:2px; font-weight:700 !important;
}
.mac-btn button:active, button.mac-btn:active{
  background:var(--ink) !important; color:var(--paper) !important;
  box-shadow:none !important; transform:translate(2px,2px);
}

/* ---- LaTeX output field: Monaco-style code well ---- */
.mac-latex textarea{
  font-family:var(--mono) !important; font-size:15px !important; color:var(--ink) !important;
  background:var(--paper) !important; border:2px solid var(--ink) !important;
  padding:12px !important; line-height:1.5 !important;
}
.mac-latex.fresh textarea{
  /* marching ants around a just-printed result */
  border:2px dashed var(--ink) !important;
  animation:ants .6s linear infinite;
}
@keyframes ants{ to{ background-position:8px 0, -8px 100%, 0 -8px, 100% 8px; } }

/* ---- confidence meter: barber-pole progress ---- */
.mac-meter-row{ display:flex; align-items:center; gap:10px; }
.mac-meter{
  flex:1; height:16px; border:2px solid var(--ink); background:var(--paper);
  overflow:hidden;
}
.mac-meter-fill{
  height:100%; background:repeating-linear-gradient(-45deg,
    var(--ink) 0 4px, var(--paper) 4px 8px);
  border-right:2px solid var(--ink); min-width:2px;
}
.mac-meter-pct{ font-family:var(--pix); font-size:12px; min-width:38px; text-align:right; }
.mac-note{ font-family:var(--pix); font-size:10px; margin-top:6px; letter-spacing:.5px; }

/* ---- read me typography ---- */
.mac-readme{ font-family:var(--mono); font-size:13px; line-height:1.55; color:var(--ink); }
.mac-readme p{ margin:0 0 11px; }
.mac-readme b{ font-weight:700; }
.mac-fine{ font-size:11px; }
.mac-table{ width:100%; border-collapse:collapse; margin:4px 0 12px; font-family:var(--mono); font-size:12px; }
.mac-table th, .mac-table td{ border:1px solid var(--ink); padding:4px 8px; text-align:left; }
.mac-table thead th{ background:var(--ink) !important; color:var(--paper) !important; font-weight:700; }
.mac-table .num{ text-align:right; }

/* darken Gradio's upload prompt so it stays 1-bit, not washed grey */
.mac-well .wrap, .mac-well .wrap *{ color:var(--ink) !important; opacity:1 !important; }

/* ---- examples strip ---- */
.mac-examples{ margin-top:12px !important; }
.mac-examples .label, .mac-examples caption{ font-family:var(--pix) !important; font-size:10px !important; text-transform:uppercase; letter-spacing:1px; }
.mac-examples table{ border:none !important; }
.mac-examples img{ border:2px solid var(--ink) !important; background:#fff !important; }

/* headings inside HTML blocks inherit pixel font */
#desktop h1,#desktop h2,#desktop h3{ font-family:var(--pix) !important; }

/* scrollbars, lightly dithered */
.gradio-container ::-webkit-scrollbar{ width:14px; height:14px; }
.gradio-container ::-webkit-scrollbar-track{ background:var(--paper) var(--desk); background-size:4px 4px; }
.gradio-container ::-webkit-scrollbar-thumb{ background:var(--paper); border:2px solid var(--ink); }

/* Gradio's image hover-toolbar re-skinned to 1-bit (no grey chrome) */
.mac-well .icon-button-wrapper, .mac-well .icon-buttons,
.mac-well button[aria-label]{ background:var(--paper) !important; border:2px solid var(--ink) !important; }
.mac-well .icon-button-wrapper svg, .mac-well button[aria-label] svg{ color:var(--ink) !important; }

@media (max-width:760px){
  #desk-cols{ flex-direction:column !important; }
  .mac-window{ box-shadow:3px 3px 0 rgba(0,0,0,1) !important; }
  .mac-menu-clock{ display:none; }              /* free space in the narrow menu bar */
  .mac-menu-item{ padding:0 8px; }
}
@media (max-width:480px){
  .mac-menubar :nth-child(n+5){ display:none; }  /* keep Apple · app · File · Edit */
}
"""

CONTRACT_COMMENT = (
    "<!-- impeccable seed 8ea9fce7 · one-bit desktop · "
    "THESIS classic model, classic machine · FINISH review+DESIGN.md pending -->"
)

with gr.Blocks(title="scribe2tex", analytics_enabled=False) as demo:
    # Gradio 6 dropped Blocks(css=...); inject as a <style> node so it travels with
    # the Blocks everywhere (local run AND Hugging Face Spaces, which calls launch()).
    gr.HTML(f"<style>{CSS}</style>" + CONTRACT_COMMENT + MENUBAR)

    with gr.Column(elem_id="desktop"):
        with gr.Row(elem_id="desk-cols"):
            # --- tool window ---
            with gr.Column(scale=3, elem_classes="mac-window"):
                gr.HTML(_titlebar("scribe2tex"))
                with gr.Column(elem_classes="mac-body"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="mac-label">Document</div>')
                            image = gr.Image(
                                type="pil", sources=["upload", "clipboard"],
                                show_label=False, height=210, elem_classes="mac-well",
                            )
                            convert = gr.Button(
                                "Convert ⌘↵", elem_classes="mac-btn default",
                            )
                        with gr.Column(scale=1):
                            gr.HTML('<div class="mac-label">Predicted LaTeX</div>')
                            latex = gr.Textbox(
                                show_label=False, lines=3,
                                placeholder="—", elem_classes="mac-latex",
                                container=False,
                            )
                            copy = gr.Button("Copy", elem_classes="mac-btn mac-copy", scale=0)
                            gr.HTML('<div class="mac-label" style="margin-top:12px">Rendered</div>')
                            preview = gr.Image(
                                show_label=False, height=120, elem_classes="mac-well",
                                interactive=False,
                            )
                            gr.HTML('<div class="mac-label" style="margin-top:12px">Confidence</div>')
                            conf = gr.HTML(
                                '<div class="mac-note">no document yet</div>',
                            )
                    gr.Examples(
                        examples=EXAMPLES, inputs=image, outputs=[latex, preview, conf],
                        fn=infer, run_on_click=True, label="Sample documents",
                        elem_id="mac-examples-wrap",
                    )

            # --- read me window ---
            with gr.Column(scale=2, elem_classes="mac-window"):
                gr.HTML(_titlebar("Read Me"))
                with gr.Column(elem_classes="mac-body"):
                    gr.HTML(READ_ME)

    convert.click(infer, inputs=image, outputs=[latex, preview, conf])
    copy.click(None, inputs=latex, outputs=None,
               js="(t) => { if (t) navigator.clipboard.writeText(t); }")

if __name__ == "__main__":
    demo.launch()
