"""Download the release checkpoints into checkpoints/ (skips files already there).
GitHub release, not the Drive folder: Drive rate-limits repeated gdown pulls."""
import os
import urllib.request

from tqdm import tqdm

RELEASE = "https://github.com/unstablerandomwalk/scribe2tex/releases/download/v0.1.0/"
FILES = ["pretrain_best.pth", "finetune_best.pth", "hme100k_best.pth"]

os.makedirs("checkpoints", exist_ok=True)
for name in FILES:
    dst = os.path.join("checkpoints", name)
    if os.path.exists(dst):
        continue
    with urllib.request.urlopen(RELEASE + name) as r, open(dst + ".part", "wb") as f, \
            tqdm(total=int(r.headers.get("Content-Length", 0)), unit="B", unit_scale=True,
                 desc=name) as bar:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    os.replace(dst + ".part", dst)  # no half-written .pth if the download dies
print(sorted(os.listdir("checkpoints")))
