"""Pure python (needs nltk): `python tests/test_metrics.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.metrics import edit_distance, score, selective  # noqa: E402


def test_edit_distance():
    assert edit_distance(["a", "b"], ["a", "b"]) == 0
    assert edit_distance(["a", "b"], ["a", "c"]) == 1
    assert edit_distance([], ["x", "y"]) == 2


def test_score_buckets():
    refs = [["x"], ["x", "y"], ["x", "y", "z"]]
    hyps = [["x"], ["x", "q"], ["a", "b", "z"]]          # dist 0, 1, 2
    s = score(refs, hyps)
    assert (s["exprate"], s["exprate_le1"], s["exprate_le2"]) == (1 / 3, 2 / 3, 1.0)


def test_selective_keeps_most_confident():
    correct = [True, True, False, False, False]
    confs = [0.9, 0.8, 0.7, 0.2, 0.1]
    s = selective(correct, confs, coverages=(1.0, 0.4))
    assert s["exprate@100%"] == 0.4
    assert s["exprate@40%"] == 1.0                       # top 2 are both right
    assert abs(s["mean_conf_correct"] - 0.85) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
