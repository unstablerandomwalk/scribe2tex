"""All hyperparameters and paths in one place."""
from pathlib import Path

# Special tokens (order fixes their ids: PAD=0 so padded loss masks cleanly)
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
SPECIALS = [PAD, SOS, EOS, UNK]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
VOCAB_PATH = CKPT_DIR / "vocab.json"

CONFIG = {
    # Model
    "encoder_dim": 256,
    "decoder_dim": 256,
    "attention_dim": 128,
    "embedding_dim": 128,
    "dropout": 0.3,

    # Training
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "finetune_lr": 1e-4,            # stage-2 fine-tune LR
    "weight_decay": 1e-5,
    "teacher_forcing_ratio": 1.0,
    "teacher_forcing_decay": 0.02,  # subtract per epoch
    "teacher_forcing_floor": 0.5,
    "grad_clip": 5.0,
    "alpha_reg": 1.0,               # doubly-stochastic attention weight
    "max_seq_length": 200,

    # Beam search (inference)
    "beam_width": 5,
    "length_penalty": 0.7,          # score = logprob / len**alpha

    # Data
    # image_height 128 = math-OCR standard: preserves stroke detail, keeps widths
    # bounded, aspect ALWAYS preserved (pad, never squish). Raise to 160/192 to
    # trade compute for a possible small gain on fine handwriting.
    "image_height": 128,
    "image_max_width": 1024,        # only aspect > 8:1 gets scaled down (still no squish)
    "min_token_freq": 2,
}
