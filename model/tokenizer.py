"""LaTeX tokenizer + vocabulary. Pure Python (no torch) so it runs anywhere."""
import json
import re
from collections import Counter
from pathlib import Path

from .config import EOS, EOS_ID, PAD, SOS, SOS_ID, SPECIALS, UNK, UNK_ID

# One LaTeX command (\frac), one escaped char (\{), a brace/^/_, or a single
# printable char is one token. \\[a-zA-Z]+ must come first so \frac beats \f.
_PATTERN = re.compile(
    r"(\\[a-zA-Z]+|\\[^a-zA-Z]|\{|\}|\^|_|"
    r"[a-zA-Z0-9]|[+\-*/=<>!.,;:'\"()\[\]|]|\s+)"
)


def tokenize_latex(latex_str: str) -> list[str]:
    """Split a LaTeX string into tokens, dropping whitespace-only matches."""
    tokens = _PATTERN.findall(latex_str)
    return [t.strip() for t in tokens if t.strip()]


class Vocab:
    """Maps LaTeX tokens <-> integer ids. Build once over ALL training corpora
    (im2latex + MathWriting) so the output layer never resizes between stages."""

    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: t for t, i in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    @classmethod
    def build(cls, token_lists, min_freq: int = 2) -> "Vocab":
        """token_lists: iterable of already-tokenized captions (list[str])."""
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        stoi = {t: i for i, t in enumerate(SPECIALS)}
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq and tok not in stoi:
                stoi[tok] = len(stoi)
        return cls(stoi)

    def encode(self, tokens: list[str], add_bounds: bool = True) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokens]
        if add_bounds:
            ids = [SOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids, strip_specials: bool = True) -> str:
        """Ids -> space-separated token string (the canonical form metrics score)."""
        specials = {SOS, EOS, PAD, UNK}
        toks = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            tok = self.itos.get(i, UNK)
            if strip_specials and tok in specials:
                continue
            toks.append(tok)
        return " ".join(toks)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False, indent=0))

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))
