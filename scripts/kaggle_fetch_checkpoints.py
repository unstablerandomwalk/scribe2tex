"""Pull release checkpoints from the shared Drive folder into checkpoints/ (flat), skip
files already there, drop the hf_cache that rides along in the folder."""
import glob
import os
import shutil
import subprocess

DRIVE = "https://drive.google.com/drive/folders/1nSJGLWYajiinlNDYWbBibtLpdbL4tyr0"

if not os.path.exists("checkpoints/pretrain_best.pth"):
    subprocess.run(["gdown", "--folder", DRIVE, "-O", "checkpoints"], check=True)
for f in glob.glob("checkpoints/**/*", recursive=True):
    dst = os.path.join("checkpoints", os.path.basename(f))
    if f.endswith((".pth", "vocab.json")) and os.path.dirname(f) != "checkpoints" \
            and not os.path.exists(dst):
        shutil.move(f, dst)
for junk in ("checkpoints/ckpts", "checkpoints/hf_cache"):
    shutil.rmtree(junk, ignore_errors=True)
print(sorted(os.listdir("checkpoints")))
