"""Render a predicted LaTeX string to PNG bytes (matplotlib mathtext, best-effort)."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render_latex_png(latex: str) -> bytes:
    """PNG bytes of the rendered LaTeX. Raises if mathtext can't parse the string
    (caller should treat rendering as a nice-to-have and fall back to the raw text)."""
    fig = plt.figure(figsize=(6, 2))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=24, ha="center", va="center")
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    finally:
        plt.close(fig)
    return buf.getvalue()
