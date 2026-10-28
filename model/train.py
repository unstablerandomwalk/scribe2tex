"""Stage-1 pretrain / stage-2 fine-tune loop: teacher forcing, grad clip, LR sched.

Run from the repo root as a module so relative imports resolve:
    python -m model.train --train data/im2latex/train.jsonl --val data/im2latex/val.jsonl
    python -m model.train --stage finetune --init-from checkpoints/pretrain_best.pth \
        --train data/mathwriting/train.jsonl --val data/mathwriting/val.jsonl --epochs 15

Best checkpoint is chosen on val cross-entropy (cheap, every epoch); BLEU/ExpRate are
reported separately by evaluate.py on the saved best checkpoint.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CKPT_DIR, CONFIG, PAD_ID, VOCAB_PATH
from .dataset import LatexImageDataset, collate_fn
from .model import Seq2Seq, arch_of, warm_start
from .tokenizer import Vocab


def load_manifest(path):
    """JSONL {"image": "...png", "latex": "..."} per line -> [(image, latex)]."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append((r["image"], r["latex"]))
    return out


def loss_fn(logits, targets, alphas, cfg):
    """Masked cross-entropy + doubly-stochastic attention regularization."""
    ce = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
    )
    alpha_reg = ((1.0 - alphas.sum(dim=1)) ** 2).mean()  # each pixel ~1 over time
    return ce + cfg["alpha_reg"] * alpha_reg


def token_accuracy(logits, targets):
    pred = logits.argmax(-1)
    mask = targets != PAD_ID
    return (pred.eq(targets) & mask).sum().item() / max(1, mask.sum().item())


def run_epoch(model, loader, opt, cfg, device, tf_ratio, train, epoch=None):
    model.train() if train else model.eval()
    tot_loss = tot_acc = n = 0
    tag = "train" if train else "val"
    desc = f"ep{epoch} {tag}" if epoch is not None else tag
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for imgs, caps, valid_hw in pbar:
        imgs, caps, valid_hw = imgs.to(device), caps.to(device), valid_hw.to(device)
        with torch.set_grad_enabled(train):
            logits, alphas = model(imgs, caps, valid_hw, tf_ratio if train else 0.0)
            targets = caps[:, 1:]                       # predict token t from prefix
            loss = loss_fn(logits, targets, alphas, cfg)
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        tot_loss += loss.item()
        tot_acc += token_accuracy(logits, targets)
        n += 1
        pbar.set_postfix(loss=f"{tot_loss / n:.3f}", acc=f"{tot_acc / n:.3f}")
    return tot_loss / max(1, n), tot_acc / max(1, n)


def make_loader(manifest, vocab, cfg, train):
    ds = LatexImageDataset(load_manifest(manifest), vocab, cfg["image_height"],
                           cfg["image_max_width"], train, cfg["max_seq_length"],
                           cfg.get("photo_aug", True))
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=train, num_workers=2,
                      collate_fn=collate_fn, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    ap.add_argument("--init-from", help="warm-start weights (stage-2)")
    ap.add_argument("--resume", help="resume same-stage checkpoint")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--tag", help="checkpoint filename prefix (default: --stage)")
    # ablation switches: turn off one improvement at a time
    ap.add_argument("--no-fine-grid", action="store_true")
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--no-photo-aug", action="store_true")
    args = ap.parse_args()

    tag = args.tag or args.stage
    cfg = {**CONFIG, "fine_grid": CONFIG["fine_grid"] and not args.no_fine_grid,
           "coverage": CONFIG["coverage"] and not args.no_coverage,
           "photo_aug": not args.no_photo_aug}
    print(f"arch: {arch_of(cfg)} photo_aug={cfg['photo_aug']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    vocab = Vocab.load(VOCAB_PATH)  # stage-1 im2latex vocab, reused for the fine-tune

    tr = make_loader(args.train, vocab, cfg, train=True)
    va = make_loader(args.val, vocab, cfg, train=False)

    model = Seq2Seq(len(vocab), cfg, pretrained=(args.stage == "pretrain")).to(device)
    lr = cfg["finetune_lr"] if args.stage == "finetune" else cfg["learning_rate"]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)

    start, best_val, tf = 0, float("inf"), cfg["teacher_forcing_ratio"]
    if args.init_from:
        new = warm_start(model, args.init_from, device)
        print(f"warm-started from {args.init_from}" + (f" (new: {new})" if new else ""))
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best_val, tf = ck["epoch"] + 1, ck["best_val"], ck["tf"]
        print(f"resumed at epoch {start} (best_val {best_val:.3f})")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or cfg["epochs"]
    for epoch in range(start, epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, tr, opt, cfg, device, tf, True, epoch)
        va_loss, va_acc = run_epoch(model, va, None, cfg, device, 0.0, False, epoch)
        sched.step(va_loss)
        tf = max(cfg["teacher_forcing_floor"], tf - cfg["teacher_forcing_decay"])
        print(f"[{tag}] ep{epoch} tf={tf:.2f} "
              f"train {tr_loss:.3f}/{tr_acc:.3f} val {va_loss:.3f}/{va_acc:.3f} "
              f"({time.time() - t0:.0f}s)")
        # decide improvement BEFORE building ck so _last.pth carries the current best_val
        improved = va_loss < best_val
        if improved:
            best_val = va_loss
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "best_val": best_val, "tf": tf, "vocab_size": len(vocab), "arch": arch_of(cfg)}
        torch.save(ck, CKPT_DIR / f"{tag}_last.pth")
        if improved:
            torch.save(ck, CKPT_DIR / f"{tag}_best.pth")
            print(f"  saved {tag}_best.pth (val_loss {best_val:.3f})")


if __name__ == "__main__":
    main()
