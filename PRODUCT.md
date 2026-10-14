# PRODUCT.md — scribe2tex

> Durable product context for design work. Facts below are drawn from the codebase
> and HANDOFF.md; lines marked (ASSUMPTION) are inferred and should be corrected if wrong.

## What it is
scribe2tex turns an image of **handwritten or printed math** into **LaTeX**. Upload a
photo or render of an expression; get back the LaTeX source, a rendered preview of that
LaTeX, and a confidence score.

## Unique mechanism
A **from-scratch, classic vision-to-sequence model**: ResNet-18 CNN encoder → **Bahdanau
attention** → LSTM decoder → beam search. Built deliberately as the "classic" architecture
so it can be **benchmarked against the modern transformer approach (pix2tex)**. It emits a
calibrated **confidence** that correctly collapses on out-of-distribution input.

## Audience & scene (ASSUMPTION, high-confidence)
Primarily **ML engineers / interviewers / recruiters** evaluating the author's portfolio —
technically literate, skeptical, short on time, viewing on a laptop. Secondary: anyone who
wants to convert a math image to LaTeX. This is a **portfolio piece whose job is to earn
credibility fast**, not a commercial product.

## What this surface (the demo) must prove
- The model actually works: upload → correct LaTeX, live.
- The engineering is real and honest: staged transfer-learning results and a confidence
  score that is *calibrated*, not decorative.
- The author has taste and rigor (this is the "impress the interviewer" surface).

## Real proof available (no invention needed)
- Benchmark results by stage: im2latex (printed) **BLEU 91.5**, MathWriting (handwritten)
  **BLEU 79.4**, HME100K (photos) **BLEU 88.3 / ExpRate 0.524**. (HME100K figures are on a
  self-carved held-out slice — label as such.)
- The live confidence score (drops to ~0.40 on OOD, ~0.94 on clean input).
- The rendered-LaTeX preview.
- The attention mechanism itself (the model literally attends over the image).

## Function & constraints (hard)
- Implemented as a **Gradio (Python) app** — `api/app.py` — that calls a shared
  `predict_latex(image) -> {latex, confidence}` core and a best-effort `render_latex_png`.
- Runs locally (`python -m api.app`) and is intended to deploy to **Hugging Face Spaces**
  (`app_file: api/app.py`). Whatever the design, it must remain a working Gradio app.
- Model weights load once (cached). `MODEL_PATH` selects the checkpoint (HME100K best is
  the most robust).

## Honest limitation (state it, don't hide it)
Raw uncropped / messy phone photos (e.g. small expression in a big frame, blue pen on
ruled paper) fail; the model wants a **tightly-cropped, dark-ink** expression. The
confidence score honestly reflects this. This is a feature of the story, not a flaw to mask.

## Brand commitments
None pinned yet. Visual world is open (to be set by the direction roll).

## Uninventable (never fabricate)
No claims beyond the real numbers above. No fake testimonials, users, or capabilities the
model lacks.
