"""Leakage tests: nothing that a model sees or is trained on comes from after the moment it is evaluated at.

Runs on the prepared data (data/ev/*.npy) and the code paths every experiment uses.
python tests/test_leakage.py            (or: pytest tests/test_leakage.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import L, UNK, events, meta, user_of_event, vocab, windows  # noqa: E402

D = 86_400
RNG = np.random.default_rng(0)
ev = events()
T = np.asarray(ev["t"])
SPLIT = np.asarray(ev["split"])
TOK = np.asarray(ev["tok"])
OFF = np.asarray(ev["offsets"])
UOE = np.asarray(user_of_event())
M = meta()


def bound_of(idx):
    """End of the event's own split (exclusive)."""
    b = np.array([M["t_val"], M["t_test"], M["tmax"] + 1])
    return b[SPLIT[idx]]


def test_sorted_by_user_then_time():
    same = UOE[1:] == UOE[:-1]
    assert (np.diff(T)[same] >= 0).all(), "events of a user are not in time order"
    assert (np.diff(UOE) >= 0).all(), "users are not contiguous"


def test_split_is_global_time_split():
    assert (SPLIT[T < M["t_val"]] == 0).all()
    assert (SPLIT[(T >= M["t_val"]) & (T < M["t_test"])] == 1).all()
    assert (SPLIT[T >= M["t_test"]] == 2).all()


def test_vocabulary_and_item_stats_from_train_only():
    voc = vocab()
    tr = TOK[(SPLIT == 0) & (TOK > UNK)]
    cnt = np.bincount(tr, minlength=len(voc["tok_pop"]))
    assert np.array_equal(cnt[2:], voc["tok_pop"][2:].astype(np.int64)), "popularity is not the train count"
    # the vocabulary = the most listened train items: nothing outside it is more popular in train than its tail
    assert voc["tok_pop"][2:].min() > 0


def test_future_labels_never_cross_their_split():
    for name, horizon in (("like", 3_600), ("cont", 1_800), ("adopt", 3 * D)):
        lab = np.asarray(ev[name])
        idx = np.flatnonzero(lab >= 0)
        assert (T[idx] + horizon < bound_of(idx)).all(), f"{name}: label horizon crosses the split boundary"
    ad = np.flatnonzero(np.asarray(ev["adopt"]) >= 0)
    assert (np.asarray(ev["first"])[ad] == 1).all() and (np.asarray(ev["organic"])[ad] == 0).all()


def test_continuation_label_recomputed():
    cont = np.asarray(ev["cont"])
    idx = RNG.choice(np.flatnonzero(cont >= 0), 20_000, replace=False)
    nxt = idx + 1
    has_next = (nxt < len(T)) & (UOE[np.minimum(nxt, len(T) - 1)] == UOE[idx])
    nxt = np.minimum(nxt, len(T) - 1)
    start_next = T[nxt] - np.round(np.asarray(ev["played"])[nxt].astype(np.float64) * np.asarray(ev["len"])[nxt]).astype(np.int64)
    expect = has_next & (start_next - T[idx] <= 1_800)
    assert np.array_equal(expect.astype(np.int8), cont[idx]), "session-continuation label differs from its definition"


def test_windows_contain_only_the_past_of_the_same_user():
    idx = RNG.choice(np.flatnonzero(TOK > UNK), 3_000, replace=False)
    w = windows(idx)
    for i, e in enumerate(idx[:3_000]):
        u0 = OFF[UOE[e]]
        lo = max(u0, e - L)
        expect = TOK[lo:e]
        got = w["tok"][i][w["mask"][i]]
        assert np.array_equal(got, expect), "window is not exactly the previous events of the same user"
    assert "like" not in w, "likes must not be part of the history (a like can come after the decision)"
    # including the event itself is explicit and only used for next-item training targets
    w2 = windows(idx[:100], length=L + 1, include_self=True)
    assert np.array_equal(w2["tok"][:, -1], TOK[idx[:100]])


def test_strict_windows_are_strictly_earlier_in_time():
    """19 % of consecutive listens share a 5-second timestamp, so their order is unknown. The conservative protocol
    (RS_STRICT=1, used for the robustness check of the test results) keeps only events with t < t(decision)."""
    idx = RNG.choice(np.flatnonzero(TOK > UNK), 3_000, replace=False)
    w = windows(idx, keys=("tok", "t"), strict=True)
    tmax_hist = np.where(w["mask"], w["t"], -1).max(1)
    assert (tmax_hist < T[idx]).all(), "strict history contains an event at or after the decision time"
    ties = (T[1:] == T[:-1]) & (UOE[1:] == UOE[:-1])
    assert ties.mean() > 0.1  # documents why the strict protocol exists


def test_training_and_rl_contexts_are_in_train():
    from policy import train_index
    from rl import rl_contexts
    tr = train_index()
    assert (SPLIT[tr] == 0).all()
    rc = rl_contexts()
    assert (SPLIT[rc] == 0).all() and (np.asarray(ev["organic"])[rc] == 0).all()


def test_rm_training_targets_never_see_themselves():
    """In rm.batch_loss the candidate at window position q may only see history positions k < q."""
    from rm import cross_feats
    tok = torch.tensor([[5, 6, 7, 8, 9]])
    played = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    art_of = torch.zeros(20, dtype=torch.long)
    cand = tok[:, 1:]  # targets are the next events
    q = torch.arange(1, 5, dtype=torch.float32)[None]
    allowed = torch.arange(5)[None, None, :] < q[:, :, None]
    f = cross_feats(cand, q, tok, played, art_of, allowed)
    assert (f[..., 0] == 0).all(), "a target event is counted in its own history"
    # a genuine repeat inside the prefix is counted
    tok2 = torch.tensor([[5, 6, 5, 8, 9]])
    f2 = cross_feats(tok2[:, 1:], q, tok2, played, art_of, allowed)
    assert f2[0, 1, 0] > 0 and f2[0, 0, 0] == 0


def test_evaluation_targets_are_future_and_same_split():
    from evaluate import eval_set
    for split, pu in ((1, 1), (2, 3)):
        idx, fs, hs = eval_set(split, pu)
        assert (SPLIT[idx] == split).all() and (TOK[idx] > UNK).all()
        for i in RNG.choice(len(idx), 300, replace=False):
            e = idx[i]
            b = OFF[UOE[e] + 1]
            sl = np.arange(e, b)
            ok = (T[sl] <= T[e] + D) & (SPLIT[sl] == split) & (TOK[sl] > UNK)
            assert np.array_equal(np.unique(TOK[sl[ok]]), fs["all"][i]), "future set is not the next 24 h"
            assert np.array_equal(hs[i], np.unique(TOK[OFF[UOE[e]]:e])), "history set includes the future"


def test_gold_item_statistics_precede_gold_training_examples():
    from gold import item_stats
    st = item_stats()
    t_stat = M["t_val"] - 30 * D
    s = (T < t_stat) & (np.asarray(ev["organic"]) == 0) & (TOK > UNK)
    n = np.bincount(TOK[s], minlength=len(st))
    assert np.allclose(st[:, 0], np.log1p(n), atol=1e-4), "gold item statistics use events after T_STAT"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as ex:
                fails += 1
                print("FAIL", name, "-", ex)
    sys.exit(1 if fails else 0)
