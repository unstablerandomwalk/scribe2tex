"""Aspect-preserving transforms, Dataset, and collate for (image, latex) pairs.

Quality-first pipeline:
- scale to fit (height, max_width) preserving aspect ratio, then pad to fixed height
  (white). Aspect ratio is NEVER distorted.
- each sample carries its valid (h, w) so attention can mask out the padding.
"""
import io
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .config import PAD_ID
from .tokenizer import Vocab, tokenize_latex

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_POST = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _shadow(img: Image.Image) -> Image.Image:
    """Darken one side of the page with a linear gradient, like a phone's shadow."""
    w, h = img.size
    strength = random.uniform(0.3, 0.7)
    ramp = torch.linspace(1 - strength, 1, w if random.random() < 0.5 else h)
    if random.random() < 0.5:
        ramp = ramp.flip(0)
    shade = ramp.view(1, -1) if ramp.numel() == w else ramp.view(-1, 1)
    t = T.functional.pil_to_tensor(img).float() * shade
    return T.functional.to_pil_image(t.clamp(0, 255).byte())


def _jpeg(img: Image.Image) -> Image.Image:
    """Re-encode at low quality to mimic phone-camera compression artefacts."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(20, 60))
    return Image.open(io.BytesIO(buf.getvalue())).convert(img.mode)


def _pre(train: bool, photo_aug: bool = True):
    """Grayscale->3ch (keeps pretrained conv1 valid) + optional augmentation. PIL->PIL.
    Training aug mimics phone photos: tilt/perspective, uneven light, blur, JPEG.
    photo_aug=False is the lighter pre-photo aug (affine, blur, jitter) for ablations."""
    aug = []
    if train and not photo_aug:
        aug = [
            T.RandomApply([T.RandomAffine(degrees=5, shear=5, translate=(0.02, 0.02))], p=0.5),
            T.RandomApply([T.GaussianBlur(3)], p=0.3),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif train:
        aug = [
            T.RandomApply([T.RandomAffine(degrees=5, shear=5, translate=(0.02, 0.02))], p=0.5),
            T.RandomPerspective(distortion_scale=0.15, p=0.3, fill=255),
            T.RandomApply([T.Lambda(_shadow)], p=0.3),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            T.ColorJitter(brightness=0.4, contrast=0.4),
            T.RandomApply([T.Lambda(_jpeg)], p=0.3),
        ]
    return T.Compose([T.Grayscale(num_output_channels=3), *aug])


def _resize_pad(img: Image.Image, height: int, max_width: int):
    """Scale to fit (height, max_width) preserving aspect, pad bottom to fixed height.
    Returns (padded_img, valid_h, valid_w). No aspect distortion, ever."""
    w, h = img.size
    scale = min(height / h, max_width / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    if nh < height:                                   # only when width was the binding limit
        white = (255,) * len(img.getbands())
        canvas = Image.new(img.mode, (nw, height), white)
        canvas.paste(img, (0, 0))
        img = canvas
    return img, nh, nw


def transform_image(img, height: int, max_width: int, train: bool = False,
                    photo_aug: bool = True):
    """img: PIL or path -> (tensor (3,H,W), valid_hw tensor([h, w])). Shared by
    Dataset, evaluate, and the API so preprocessing never drifts between them."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = _pre(train, photo_aug)(img.convert("L"))
    img, vh, vw = _resize_pad(img, height, max_width)
    return _POST(img), torch.tensor([vh, vw], dtype=torch.long)


class LatexImageDataset(Dataset):
    """samples: list of (image_path, latex_string)."""

    def __init__(self, samples, vocab: Vocab, height, max_width, train, max_seq_length=200,
                 photo_aug=True):
        self.samples = samples
        self.vocab = vocab
        self.height = height
        self.max_width = max_width
        self.train = train
        self.max_seq_length = max_seq_length
        self.photo_aug = photo_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, latex = self.samples[idx]
        img, valid_hw = transform_image(path, self.height, self.max_width, self.train,
                                         self.photo_aug)
        ids = self.vocab.encode(tokenize_latex(latex))[: self.max_seq_length]
        return img, torch.tensor(ids, dtype=torch.long), valid_hw


def collate_fn(batch):
    """Pad images to batch-max width and captions to batch-max length. valid_hw travels
    through so attention masks the padding (quality) rather than attending to blanks."""
    imgs, caps, valid_hw = zip(*batch)
    max_w = max(i.shape[2] for i in imgs)
    imgs = torch.stack([torch.nn.functional.pad(i, (0, max_w - i.shape[2])) for i in imgs])
    max_len = max(len(c) for c in caps)
    padded = torch.full((len(caps), max_len), PAD_ID, dtype=torch.long)
    for j, c in enumerate(caps):
        padded[j, : len(c)] = c
    return imgs, padded, torch.stack(valid_hw)
