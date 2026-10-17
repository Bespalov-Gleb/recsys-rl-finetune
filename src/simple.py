"""Non-neural reference points under the same protocol: global popularity and the user's own most played tracks.

python src/simple.py
"""
import numpy as np

from common import L, UNK, events, vocab, windows
from evaluate import eval_set, full_eval
from rm import load_rm


def pop_recs(n):
    voc = vocab()
    top = np.argsort(-voc["tok_pop"])[:100]
    return np.tile(top, (n, 1))


def histfreq_recs(idx):
    """Most frequent tracks among the last L listens (ties -> more recent first), padded with popular tracks."""
    w = windows(idx)
    pop = pop_recs(1)[0]
    out = np.zeros((len(idx), 100), np.int64)
    for i in range(len(idx)):
        t = w["tok"][i][w["mask"][i]]
        t = t[t > UNK]
        u, last, cnt = np.unique(t[::-1], return_index=True, return_counts=True)
        order = np.lexsort((last, -cnt))
        r = list(u[order][:100])
        r += [p for p in pop if p not in set(r)][: 100 - len(r)]
        out[i] = r[:100]
    return out


if __name__ == "__main__":
    rm = load_rm("rm0")
    for split, pu in ((1, 1), (2, 3)):
        idx = eval_set(split, pu)[0]
        for name, recs in (("pop", pop_recs(len(idx))), ("histfreq", histfreq_recs(idx))):
            res = full_eval(name, recs, split=split, per_user=pu, rms=[rm])
            print(split, name, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
