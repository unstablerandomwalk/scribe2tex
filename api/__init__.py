"""scribe2tex serving layer: one predict() core, thin FastAPI + Gradio wrappers.

Register HEIF/HEIC support at import so iPhone photos (.HEIC) open via PIL — this runs
before Gradio's upload preprocessing and our own predict path both call PIL.Image.open.
"""
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # optional; PNG/JPG work without it
    pass
