"""Fine-tune TAMER on scribe2tex JSONL manifests, starting from the released HME100K weights.
Tokens TAMER has never seen (from MathWriting) are added to its dictionary.

    python -m hmer.train --train data/mix_train.jsonl --val data/mix_val.jsonl \
        --init pretrained --invert --tag tamer_mix --epochs 6

Resumable mid-epoch: {tag}_last.pth is rewritten every --save-every-min minutes and after each
epoch; {tag}_best.pth is the lowest validation loss (as model/train.py does)."""
import argparse
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.config import CKPT_DIR
from model.train import load_manifest

from .data import MAX_LEN, BatchSet, batches, encode, extend_dictionary, fitted_hw, to_tokens, word2idx
from .upstream import build, loss_fn, read_checkpoint, set_vocab, struct_parses


def prepare(manifest, words, limit=None):
    """-> [(image_path, ids)] and their fitted (h, w), dropping rows TAMER cannot encode."""
    known, w2i = set(words), word2idx(words)
    samples = []
    rows = load_manifest(manifest)[:limit]
    for path, latex in rows:
        tokens = to_tokens(latex, known)
        ids = encode(tokens, w2i)
        if ids and len(ids) <= MAX_LEN and struct_parses(tokens):
            samples.append((path, ids))
    print(f"{manifest}: {len(samples)}/{len(rows)} rows usable")
    sizes = [fitted_hw(p) for p, _ in tqdm(samples, desc="image sizes", unit="img")]
    return samples, sizes


def loader(samples, batch_list, invert, workers):
    return DataLoader(BatchSet(samples, batch_list, invert), batch_size=None,
                      num_workers=workers, pin_memory=True)


def save(path, **state):
    tmp = f"{path}.tmp"          # write then rename: the HF backup thread never sees half a file
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model, data, device, amp):
    model.eval()
    total = n = 0
    for imgs, mask, ids in data:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            total += loss_fn(model, imgs.to(device), mask.to(device), ids).item() * len(ids)
        n += len(ids)
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--init", default="pretrained", help="'pretrained' or a .ckpt/.pth")
    ap.add_argument("--resume", help="{tag}_last.pth to continue from")
    ap.add_argument("--invert", action="store_true", help="white ink on black (see notebook cell 4)")
    ap.add_argument("--tag", default="tamer_mix")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.008,
                    help="Adadelta, as upstream; 0.008 = upstream LR after its first decay")
    ap.add_argument("--save-every-min", type=float, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", help="fp32 if fp16 gives NaN losses")
    ap.add_argument("--limit", type=int, help="first N rows per manifest (smoke test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not args.no_amp
    ck = read_checkpoint(args.resume or args.init)
    if args.resume:
        words, invert = ck["words"], ck["invert"]
    else:
        known = set(ck["words"])
        train_rows = load_manifest(args.train)[:args.limit]
        words = extend_dictionary(ck["words"], [to_tokens(latex, known) for _, latex in train_rows])
        invert = args.invert
        print(f"dictionary: {len(ck['words'])} pretrained + {len(words) - len(ck['words'])} new tokens")
    set_vocab(words)
    model = build(ck["arch"], words, ck["model"]).to(device)

    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epoch0, step0, best = 0, 0, math.inf
    if args.resume:
        epoch0, step0, best = ck["epoch"], ck["step"], ck["best"]
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        print(f"resumed {args.resume}: epoch {epoch0 + 1}, batch {step0}, best val {best:.4f}")

    train, train_sizes = prepare(args.train, words, args.limit)
    val, val_sizes = prepare(args.val, words, args.limit)
    train_batches = batches(train_sizes)
    val_data = loader(val, batches(val_sizes), invert, args.workers)
    last, best_path = CKPT_DIR / f"{args.tag}_last.pth", CKPT_DIR / f"{args.tag}_best.pth"
    meta = {"words": words, "arch": ck["arch"], "invert": invert}

    def save_last(epoch, step):
        save(last, model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
             epoch=epoch, step=step, best=best, **meta)

    model.train()
    for epoch in range(epoch0, args.epochs):
        order = list(range(len(train_batches)))
        random.Random(epoch).shuffle(order)          # fixed per epoch, so a resume can skip ahead
        skip = step0 if epoch == epoch0 else 0
        todo = [train_batches[i] for i in order[skip:]]
        bar = tqdm(loader(train, todo, invert, args.workers), total=len(todo),
                   desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch")
        running, saved_at = 0.0, time.time()
        for step, (imgs, mask, ids) in enumerate(bar, start=skip + 1):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model, imgs.to(device), mask.to(device), ids)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running = 0.98 * running + 0.02 * loss.item() if running else loss.item()
            bar.set_postfix(loss=f"{running:.3f}")
            if time.time() - saved_at > args.save_every_min * 60:
                save_last(epoch, step)
                saved_at = time.time()

        val_loss = validate(model, val_data, device, amp)
        if val_loss < best:
            best = val_loss
            save(best_path, model=model.state_dict(), epoch=epoch + 1, val_loss=val_loss, **meta)
        save_last(epoch + 1, 0)
        print(f"epoch {epoch + 1}: train {running:.4f} | val {val_loss:.4f} | best {best:.4f}")


if __name__ == "__main__":
    main()
