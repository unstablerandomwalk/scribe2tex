"""Plot where the decoder attends for each output token (greedy decode).

    python scripts/attention_map.py --ckpt checkpoints/mwfull20_best.pth \
        --image data/mathwriting/test/0000001.png --out docs/paper/figs/att.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import torch                                             # noqa: E402

from model.config import CONFIG, EOS_ID, SOS_ID, VOCAB_PATH  # noqa: E402
from model.dataset import transform_image                # noqa: E402
from model.model import Seq2Seq, load_model              # noqa: E402
from model.tokenizer import Vocab                        # noqa: E402

MAX_PANELS = 12


@torch.no_grad()
def decode_with_attention(model, img, valid_hw):
    feats, (fh, fw) = model.encoder(img)
    mask = Seq2Seq.build_mask(valid_hw, img.size(2), img.size(3), fh, fw, img.device)
    state = model.decoder.init_hidden(feats, mask)
    token = torch.tensor([SOS_ID])
    tokens, alphas = [], []
    for _ in range(CONFIG["max_seq_length"]):
        logits, alpha, state = model.decoder.step(token, state, feats, mask)
        token = logits.argmax(1)
        if token.item() == EOS_ID:
            break
        tokens.append(token.item())
        alphas.append(alpha.view(fh, fw))
    return tokens, alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vocab = Vocab.load(VOCAB_PATH)
    model = load_model(args.ckpt, len(vocab), CONFIG, "cpu").eval()
    img, valid_hw = transform_image(args.image, CONFIG["image_height"], CONFIG["image_max_width"])
    tokens, alphas = decode_with_attention(model, img.unsqueeze(0), valid_hw.unsqueeze(0))
    words = vocab.decode(tokens).split()
    raw = img.mean(0)                                     # (H, W), normalized grayscale
    vh, vw = valid_hw.tolist()
    step = max(1, -(-len(alphas) // MAX_PANELS))          # ceil: at most MAX_PANELS panels
    picks = list(range(0, len(alphas), step))
    cols = 3
    rows = -(-len(picks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 1.3))
    for ax in axes.flat:
        ax.axis("off")
    for ax, t in zip(axes.flat, picks):
        a = torch.nn.functional.interpolate(alphas[t][None, None], size=raw.shape,
                                            mode="bilinear", align_corners=False)[0, 0]
        ax.imshow(raw[:vh, :vw], cmap="gray")
        ax.imshow(a[:vh, :vw], cmap="jet", alpha=0.45)
        ax.set_title(words[t] if t < len(words) else "?", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("pred:", " ".join(words))


if __name__ == "__main__":
    main()
