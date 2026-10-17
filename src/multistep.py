"""What happens when the policy generates several items ahead, each appended to its own context?

For test contexts that are followed by >= 5 listens (known vocabulary) of the same user, compare at step k=1..5:
  teacher  context = history + the REAL next k-1 listens          (classic next-item evaluation)
  static   context = history only, the same distribution for every k (one list, no update)
  greedy   context = history + the policy's own top-1 items of the previous steps
  sample   context = history + items sampled from the policy
and measure HR@10 against the real k-th listen, plus the drift of the generated items: entropy, popularity,
repeats of earlier generated items (loops), same artist as the previous generated item, repeat of history,
and the reward of the generated item under three assumptions about the unobserved feedback on the earlier
generated items ("imagined" = the RM's own predicted played ratio, "ignored" = played 0, "real" = the RM sees
only the real history) plus the gold judge (real history only).

python src/multistep.py base grpo_full_b0.1 ...
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from common import DEV, RES, UNK, dump, events, to_dev, user_of_event, vocab, windows
from evaluate import eval_set
from policy import load_policy
from rl import append, is_new
from rm import load_rm

T = 5


def contexts(n: int = 4000) -> np.ndarray:
    ev = events()
    idx = eval_set(2, 3)[0]
    off = ev["offsets"]
    u = user_of_event()[idx]
    ok = []
    for e, uu in zip(idx, u):
        nxt = ev["tok"][e:min(e + T, off[uu + 1])]
        if len(nxt) == T and (nxt > UNK).all() and (ev["split"][e:e + T] == 2).all():
            ok.append(e)
    ok = np.array(ok)
    return ok[np.random.default_rng(0).permutation(len(ok))[:n]]


@torch.no_grad()
def run(tag: str, idx: np.ndarray, bs: int = 200) -> dict:
    from gold import gold_score
    ev = events()
    voc = vocab()
    pol = load_policy(tag).eval()
    rm = load_rm("rm0")
    art = torch.as_tensor(voc["tok_artist"]).to(DEV).long()
    lpop = torch.log1p(torch.as_tensor(voc["tok_pop"]).to(DEV))
    real = np.stack([ev["tok"][e:e + T] for e in idx])
    res = {m: {k: [] for k in ("hr10", "entropy", "logpop", "loop", "same_artist_prev", "hist_repeat",
                               "r_real", "r_imagined", "r_ignored", "gold_played", "gold_full")}
           for m in ("teacher", "static", "greedy", "sample")}
    gen_items = {m: [] for m in ("greedy", "sample")}
    for i in range(0, len(idx), bs):
        e = idx[i:i + bs]
        # last L - T events of history: history + T generated items stay within the L trained positions
        W = {k: v[:, T:] for k, v in to_dev(windows(e)).items()}
        R = torch.as_tensor(real[i:i + bs]).to(DEV).long()
        hist = W["tok"].long()
        for mode in res:
            seq = hist
            Wimg, Wign = W, W
            first_lg = None
            gen = []
            for k in range(T):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = pol.last_logits(seq) if not (mode == "static" and first_lg is not None) else first_lg
                if first_lg is None:
                    first_lg = lg
                lp = torch.log_softmax(lg.float(), -1)
                top = lg.topk(10).indices
                res[mode]["hr10"].append((top == R[:, k:k + 1]).any(1).float().cpu().numpy())
                res[mode]["entropy"].append(-(lp.exp() * lp).sum(-1).cpu().numpy())
                if mode == "teacher":
                    a = R[:, k]
                elif mode == "sample":
                    a = torch.multinomial(lp.exp(), 1)[:, 0]
                else:
                    a = top[:, 0]
                if mode in gen_items:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        P_real = torch.sigmoid(rm.score(W, a[:, None]).float())[:, 0]
                        P_img = torch.sigmoid(rm.score(Wimg, a[:, None]).float())[:, 0]
                        P_ign = torch.sigmoid(rm.score(Wign, a[:, None]).float())[:, 0]
                    res[mode]["r_real"].append(P_real[:, 0].cpu().numpy())
                    res[mode]["r_imagined"].append(P_img[:, 0].cpu().numpy())
                    res[mode]["r_ignored"].append(P_ign[:, 0].cpu().numpy())
                    res[mode]["logpop"].append(lpop[a].cpu().numpy())
                    prev = torch.stack(gen, 1) if gen else None
                    res[mode]["loop"].append(((prev == a[:, None]).any(1) if gen else torch.zeros_like(a, dtype=torch.bool)).float().cpu().numpy())
                    res[mode]["same_artist_prev"].append(((art[a] == art[gen[-1]]) & (art[a] > 0) if gen else torch.zeros_like(a, dtype=torch.bool)).float().cpu().numpy())
                    res[mode]["hist_repeat"].append((~is_new(a[:, None], hist, W["mask"])[:, 0]).float().cpu().numpy())
                    Wimg = append(Wimg, a, P_img[:, 0])
                    Wign = append(Wign, a, torch.zeros_like(P_img[:, 0]))
                    gen.append(a)
                seq = torch.cat([seq, a[:, None]], 1)
            if mode in gen_items:
                gen_items[mode].append(torch.stack(gen, 1).cpu().numpy())
    out = {}
    for mode, d in res.items():
        out[mode] = {}
        for k, v in d.items():
            if not v:
                continue
            arr = np.stack([np.concatenate(v[j::T]) for j in range(T)])  # (T, n) -- appended per batch per step
            out[mode][k] = arr.mean(1).round(5).tolist()
    for mode, g in gen_items.items():
        G = np.concatenate(g)
        P = gold_score(idx, G)
        out[mode]["gold_played"] = P[..., 0].mean(0).round(5).tolist()
        out[mode]["gold_full"] = P[..., 1].mean(0).round(5).tolist()
        out[mode]["unique_items_step"] = [int(len(np.unique(G[:, k]))) for k in range(T)]
    dump(f"multistep_{tag}", out)
    return out


if __name__ == "__main__":
    idx = contexts()
    for tag in sys.argv[1:]:
        o = run(tag, idx)
        print(tag, {m: {k: v for k, v in d.items() if k in ("hr10", "r_imagined", "r_real", "gold_played", "loop")} for m, d in o.items()})
