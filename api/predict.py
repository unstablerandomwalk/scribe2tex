"""Shared inference core: load model + vocab once, image -> {latex, confidence}.

Checkpoint/vocab paths are env-overridable so the same code runs locally, on Colab,
and on Hugging Face Spaces:
    MODEL_PATH   (default checkpoints/finetune_best.pth)
    VOCAB_FILE   (default checkpoints/vocab.json)
"""
import os
from functools import lru_cache

import torch

from model.config import CKPT_DIR, CONFIG, VOCAB_PATH
from model.dataset import transform_image
from model.inference import beam_search
from model.model import load_model
from model.tokenizer import Vocab

MODEL_PATH = os.environ.get("MODEL_PATH", str(CKPT_DIR / "finetune_best.pth"))
VOCAB_FILE = os.environ.get("VOCAB_FILE", str(VOCAB_PATH))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load():
    """Load weights + vocab once (cached across requests)."""
    vocab = Vocab.load(VOCAB_FILE)
    model = load_model(MODEL_PATH, len(vocab), CONFIG, DEVICE)
    model.eval()
    return model, vocab


def predict_latex(image, beam_width=None) -> dict:
    """image: PIL.Image or path -> {'latex': str, 'confidence': float in [0,1]}."""
    model, vocab = _load()
    tensor, valid_hw = transform_image(image, CONFIG["image_height"], CONFIG["image_max_width"])
    latex, conf = beam_search(
        model, tensor.unsqueeze(0).to(DEVICE), vocab, valid_hw.unsqueeze(0).to(DEVICE),
        beam_width or CONFIG["beam_width"], CONFIG["max_seq_length"], CONFIG["length_penalty"],
    )
    return {"latex": latex, "confidence": round(float(conf), 4)}
