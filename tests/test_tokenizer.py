"""Runnable now, no torch: tokenizer + vocab behavior. `python tests/test_tokenizer.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import EOS_ID, PAD_ID, SOS_ID, UNK_ID  # noqa: E402
from model.tokenizer import Vocab, tokenize_latex          # noqa: E402


def test_tokenize_splits_commands_and_chars():
    assert tokenize_latex(r"\frac{x^2}{y} + z") == \
        ["\\frac", "{", "x", "^", "2", "}", "{", "y", "}", "+", "z"]


def test_command_not_split_into_backslash_and_letters():
    # \\[a-zA-Z]+ must win: \sqrt is ONE token, not '\','s','q','r','t'
    assert tokenize_latex(r"\sqrt{2}")[0] == "\\sqrt"


def test_specials_have_fixed_ids():
    v = Vocab.build([tokenize_latex(r"x + y")], min_freq=1)
    assert (v.stoi["<PAD>"], v.stoi["<SOS>"], v.stoi["<EOS>"], v.stoi["<UNK>"]) == \
        (PAD_ID, SOS_ID, EOS_ID, UNK_ID)


def test_min_freq_drops_rare_to_unk():
    corpus = [tokenize_latex(r"a + a"), tokenize_latex(r"a + b")]  # a:3 b:1
    v = Vocab.build(corpus, min_freq=2)
    assert "a" in v.stoi and "b" not in v.stoi
    assert v.encode(["b"], add_bounds=False) == [UNK_ID]


def test_encode_decode_roundtrip():
    toks = tokenize_latex(r"\frac{1}{2}")
    v = Vocab.build([toks], min_freq=1)
    ids = v.encode(toks)
    assert ids[0] == SOS_ID and ids[-1] == EOS_ID
    assert v.decode(ids) == r"\frac { 1 } { 2 }"   # space-joined canonical form


def test_save_load_roundtrip():
    import os
    import tempfile
    v = Vocab.build([tokenize_latex(r"x = y")], min_freq=1)
    p = os.path.join(tempfile.mkdtemp(), "vocab.json")
    v.save(p)
    assert Vocab.load(p).stoi == v.stoi


def _run():
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok: {f.__name__}")
    print(f"\n{len(fns)} tokenizer tests passed")


if __name__ == "__main__":
    _run()
