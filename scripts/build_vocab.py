"""Build ONE vocab over ALL training manifests so the output layer never resizes
between pretrain and fine-tune.
    python scripts/build_vocab.py data/im2latex/train.jsonl data/mathwriting/train.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import CONFIG, VOCAB_PATH          # noqa: E402
from model.tokenizer import Vocab, tokenize_latex     # noqa: E402
from model.train import load_manifest                 # noqa: E402


def main(manifests):
    token_lists = (tokenize_latex(latex)
                   for m in manifests for _, latex in load_manifest(m))
    vocab = Vocab.build(token_lists, min_freq=CONFIG["min_token_freq"])
    vocab.save(VOCAB_PATH)
    print(f"vocab size {len(vocab)} -> {VOCAB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/build_vocab.py <manifest.jsonl> [more...]")
        sys.exit(1)
    main(sys.argv[1:])
