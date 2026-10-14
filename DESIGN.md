# DESIGN.md — scribe2tex demo (`api/app.py`)

> Durable visual decisions for the Gradio demo surface. Written from the built world
> (impeccable seed `8ea9fce7`). Ground truth over intention.

## Direction
**One-bit Macintosh desktop** (Susan-Kare-era System 1–6). Thesis: a deliberately
*classic* vision→sequence model deserves a *classic machine*, not another dark-neon ML
playground. The demo is a black-and-white desktop; the model "prints" LaTeX into a window.

## Color — Drenched 1-bit
- Only **pure black `#000` (ink)** and **pure white `#fff` (paper)**. No greys as fills.
- Every "grey" is an **ordered dither**: the desktop is a 4px 50% checkerboard SVG
  (`--desk`); the racing-stripe title bar is 1px repeating lines; the confidence fill is a
  45° barber-pole.
- Gradio's color accents are neutralised (`--color-accent`, `--link-text-color`,
  `--body-text-color` → black/white) so nothing leaks colour.

## Type
- **Chrome / labels / buttons:** `Silkscreen` (Google) — pixel bitmap face, stands in for
  Chicago/Geneva. Uppercase, tracked, small.
- **Code / body / tables:** `Courier Prime` (Google) → `Monaco`/`Menlo` fallback — the
  period-correct Mac coding mono. LaTeX output and the Read Me prose live here.
- `image-rendering: pixelated` on images for the bitmap feel.

## Components (the Mac grammar)
- **Window:** white body, 2px black border, **hard 5px offset shadow** (no blur), square
  corners. Title bar = racing stripes + centred white name plaque + 11px square close box.
- **Menu bar:** sticky white strip, 2px black underline, black pixel apple + File/Edit/…;
  hover inverts. Trims to Apple·app·File·Edit ≤480px.
- **Buttons:** rounded-rect (9px), 2px border, 2px offset shadow; **default button** adds a
  2px outline; **:active inverts** to black-on-white and nudges 2px (the "push").
- **LaTeX field:** Monaco textarea, 2px border. (`.mac-latex.fresh` marching-ants class
  exists for a just-printed flourish — not yet wired to the event.)
- **Confidence:** bordered track + barber-pole fill, % in pixel type, plus an honest note
  ("reads clean" / "low — try a tighter crop").
- **Benchmark table:** black header row (white text), 1px cell borders, right-aligned nums.

## Motion
Restrained and native: button press = invert + 2px translate; menu hover = invert. No
scattered hover effects. (Marching-ants on fresh result is specced but unwired.)

## Layout / responsive
Centred 940px desktop of two windows (tool + Read Me) side by side; stacks to one column
≤760px. Everything scales; the tool remains fully usable on mobile.

## Constraints honored
Stays a working Gradio 6 app. CSS ships as a `<style>` node (Gradio 6 dropped
`Blocks(css=)`), so it applies locally **and** on Hugging Face Spaces. `predict_latex` /
`render_latex_png` cores untouched; examples + copy wired.

## Known open items (finish review, in-thread substitute)
- Marching-ants "fresh result" flourish is specced in CSS but not toggled on convert.
- Gradio's image hover-toolbar is re-skinned to 1-bit, but the fullscreen/share overlay
  variant may still show faint chrome on desktop hover.
Both minor; neither is rebuild-level.
