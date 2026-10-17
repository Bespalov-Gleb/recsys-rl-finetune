"""Reward model: predicts the user's response to a RECOMMENDED track given the listening history.

Trained only on non-organic listens (the track was put in front of the user by the service); organic listens
stay in the history as context. This follows the setup of arXiv 2607.02818 (organic events = state,
recommendation events = actions).

Heads (logits): played (soft BCE on played ratio), full (>= 50%), like (within 1 h), cont (session continues),
adopt (recommended new track is later re-listened organically / liked within 3 days; defined only for the
first listen of a track).

python src/rm.py train --tag rm0 --seed 0
python src/rm.py eval  --tag rm0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import GPT2Config, GPT2Model

from common import DEV, L, PAD, RES, UNK, cosine_lr, dump, events, meta, to_dev, vocab, windows

CKPT = RES.parent / "checkpoints"
HEADS = ["played", "full", "like", "cont", "adopt"]
N_CROSS = 8


def cross_feats(cand: torch.Tensor, q: torch.Tensor, tok: torch.Tensor, played: torch.Tensor,
                art_of: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """History x candidate features. cand (B,C) tokens; q (B,C) position of the candidate on the window's time
    axis; tok/played (B,Lw) history; allowed (B,C,Lw) which history positions are visible to that candidate."""
    Lw = tok.shape[1]
    pos = torch.arange(Lw, device=tok.device, dtype=torch.float32)
    ca, ta = art_of[cand], art_of[tok]
    eq = (cand[:, :, None] == tok[:, None, :]) & allowed
    eqa = (ca[:, :, None] == ta[:, None, :]) & allowed & (ca[:, :, None] > 0)
    eqf, eqaf = eq.float(), eqa.float()
    pl_ = played[:, None, :].float()
    n_i = eqf.sum(-1)
    n_a = eqaf.sum(-1)
    m_i = (eqf * pl_).sum(-1) / n_i.clamp(min=1)
    m_a = (eqaf * pl_).sum(-1) / n_a.clamp(min=1)
    last_i = (eqf * (pos + 1)).amax(-1)  # 0 if never
    rec_i = torch.where(last_i > 0, torch.log1p(q - last_i + 1), torch.full_like(last_i, 7.0))
    # last visible event (for training: q-1; for inference: Lw-1)
    lastpos = (allowed.float() * (pos + 1)).amax(-1).long().clamp(min=1) - 1
    same_last = (ca == torch.gather(ta, 1, lastpos.view(lastpos.shape[0], -1)).view_as(ca)) & (ca > 0)
    n_vis = allowed.float().sum(-1)
    return torch.stack([torch.log1p(n_i), m_i, (n_i > 0).float(), torch.log1p(n_a), m_a, rec_i,
                        same_last.float(), torch.log1p(n_a) - torch.log1p(n_vis)], -1)


class RewardModel(torch.nn.Module):
    def __init__(self, vocab: int, n_art: int, d: int = 128, layers: int = 2, heads: int = 4, drop: float = 0.1):
        super().__init__()
        cfg = GPT2Config(vocab_size=1, n_positions=L + 8, n_embd=d, n_layer=layers, n_head=heads,
                         resid_pdrop=drop, embd_pdrop=drop, attn_pdrop=drop)
        self.enc = GPT2Model(cfg)
        self.item = torch.nn.Embedding(vocab, d)
        self.pb = torch.nn.Embedding(12, d)   # played bucket 0..10
        self.org = torch.nn.Embedding(2, d)
        self.cand = torch.nn.Embedding(vocab, d)
        self.art = torch.nn.Embedding(n_art + 1, 32)
        voc = vocab_np()
        self.register_buffer("art_of", torch.as_tensor(voc["tok_artist"]).long())
        self.register_buffer("loglen", torch.log(torch.as_tensor(voc["tok_len"]).clamp(min=10)) - 5.3)
        self.register_buffer("logpop", torch.log1p(torch.as_tensor(voc["tok_pop"])) / 10)
        din = 3 * d + 32 + 2 + N_CROSS
        self.mlp = torch.nn.Sequential(torch.nn.Linear(din, 256), torch.nn.GELU(), torch.nn.Dropout(drop),
                                       torch.nn.Linear(256, 128), torch.nn.GELU(), torch.nn.Linear(128, len(HEADS)))

    def encode(self, tok, played, organic, mask):
        pb = (played.clamp(0, 1) * 10).round().long()
        x = self.item(tok) + self.pb(pb) + self.org(organic.long().clamp(0, 1))
        pos = (mask.long().cumsum(1) - 1).clamp(min=0)
        return self.enc(inputs_embeds=x, attention_mask=mask.long(), position_ids=pos).last_hidden_state

    def head(self, h, cand, cross):
        """h (B,C,d) or (B,1,d) broadcast, cand (B,C)."""
        e = self.cand(cand)
        h = h.expand_as(e)
        z = torch.cat([h, e, h * e, self.art(self.art_of[cand]), self.loglen[cand][..., None],
                       self.logpop[cand][..., None], cross], -1)
        return self.mlp(z)

    def score(self, w: dict, cand: torch.Tensor) -> torch.Tensor:
        """Inference: window of L past events, arbitrary candidates (B,C) -> logits (B,C,5)."""
        tok, played, organic, mask = w["tok"].long(), w["played"], w["organic"], w["mask"]
        h = self.encode(tok, played, organic, mask)[:, -1:, :]
        Lw = tok.shape[1]
        out = []
        for i in range(0, cand.shape[1], 4096):
            c = cand[:, i:i + 4096]
            allowed = mask[:, None, :].expand(-1, c.shape[1], -1)
            q = torch.full(c.shape, float(Lw), device=c.device)
            out.append(self.head(h, c, cross_feats(c, q, tok, played, self.art_of, allowed)))
        return torch.cat(out, 1)


_voc = None


def vocab_np():
    global _voc
    if _voc is None:
        _voc = vocab()
    return _voc


def load_rm(tag: str) -> RewardModel:
    ck = torch.load(CKPT / f"rm_{tag}.pt", map_location="cpu")
    m = RewardModel(**ck["kw"])
    m.load_state_dict(ck["sd"])
    return m.to(DEV).eval()


def label_tensors(e_idx: np.ndarray) -> dict:
    ev = events()
    return {"played": ev["played"][e_idx], "full": ev["full"][e_idx].astype(np.float32),
            "like": ev["like"][e_idx], "cont": ev["cont"][e_idx], "adopt": ev["adopt"][e_idx],
            "valid": (ev["organic"][e_idx] == 0) & (ev["tok"][e_idx] > UNK)}


def batch_loss(m: RewardModel, e: np.ndarray, start: int):
    """Window of L+1 events ending at e; positions >= start predict the next event from the prefix."""
    ev = events()
    w = windows(e, length=L + 1, include_self=True)
    off = ev["offsets"]
    from common import user_of_event
    u0 = off[user_of_event()[e]]
    pos_idx = e[:, None] - L + np.arange(L + 1)[None, :]  # event index of every window slot
    pos_idx = np.where(w["mask"], pos_idx, 0)
    lab = label_tensors(pos_idx[:, 1:])
    lab["valid"] &= w["mask"][:, 1:]
    lab["valid"][:, :start] = False
    W = to_dev(w)
    tok, played, organic, mask = W["tok"].long(), W["played"], W["organic"], W["mask"]
    h = m.encode(tok[:, :-1], played[:, :-1], organic[:, :-1], mask[:, :-1])[:, start:]
    cand = tok[:, 1 + start:]
    Lq = L - start
    q = torch.arange(start + 1, L + 1, device=DEV, dtype=torch.float32)[None, :].expand(len(e), -1)
    kpos = torch.arange(L, device=DEV)
    allowed = (kpos[None, None, :] < q[:, :, None]) & mask[:, None, :-1]
    cross = cross_feats(cand, q, tok[:, :-1], played[:, :-1], m.art_of, allowed)
    lg = m.head(h, cand, cross)  # (B,Lq,5)
    T = {k: torch.as_tensor(v[:, start:]).to(DEV) for k, v in lab.items()}
    valid = T["valid"]
    losses = {}
    tot = 0.0
    for i, k in enumerate(HEADS):
        y = T[k].float()
        ok = valid & (y >= 0)
        if ok.any():
            l_ = F.binary_cross_entropy_with_logits(lg[..., i][ok].float(), y[ok])
            losses[k] = l_.item()
            tot = tot + l_
    return tot, losses


@torch.no_grad()
def predict_events(m: RewardModel, idx: np.ndarray, bs: int = 256) -> np.ndarray:
    """Predictions (sigmoid) for logged events: history before event, candidate = the logged track."""
    m.eval()
    out = []
    tok_all = events()["tok"]
    for i in range(0, len(idx), bs):
        e = idx[i:i + bs]
        W = to_dev(windows(e))
        cand = torch.as_tensor(tok_all[e]).to(DEV).long()[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m.score(W, cand)[:, 0]
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out)


def heldout_idx(split: int, n: int, seed: int = 0) -> np.ndarray:
    ev = events()
    ok = np.flatnonzero((ev["split"] == split) & (ev["organic"] == 0) & (ev["tok"] > UNK))
    from common import user_of_event
    off = ev["offsets"]
    ok = ok[ok - off[user_of_event()[ok]] >= 20]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ok, min(n, len(ok)), replace=False))


def metrics_on(pred: np.ndarray, idx: np.ndarray) -> dict:
    lab = label_tensors(idx)
    res = {}
    p = pred[:, 0]
    y = lab["played"]
    res["played_mae"] = float(np.abs(p - y).mean())
    res["played_rmse"] = float(np.sqrt(((p - y) ** 2).mean()))
    res["played_mae_const"] = float(np.abs(y - y.mean()).mean())
    res["played_pearson"] = float(np.corrcoef(p, y)[0, 1])
    for i, k in enumerate(HEADS[1:], 1):
        yy = lab[k].astype(float)
        ok = yy >= 0
        if ok.sum() > 100 and len(np.unique(yy[ok])) == 2:
            res[f"{k}_auc"] = float(roc_auc_score(yy[ok], pred[ok, i]))
            res[f"{k}_rate"] = float(yy[ok].mean())
            res[f"{k}_pred_mean"] = float(pred[ok, i].mean())
    # per-user AUC for full listens (ranking quality within a user, which is what RL exploits)
    from common import user_of_event
    u = user_of_event()[idx]
    yy = lab["full"]
    aucs, wts = [], []
    for uu in np.unique(u):
        s = u == uu
        if s.sum() >= 10 and 0 < yy[s].mean() < 1:
            aucs.append(roc_auc_score(yy[s], pred[s, 1]))
            wts.append(s.sum())
    res["full_gauc"] = float(np.average(aucs, weights=wts))
    return res


def train(args):
    torch.manual_seed(args.seed)
    ev = events()
    V = meta()["vocab"] + 2
    kw = dict(vocab=V, n_art=meta()["n_artists"], d=args.d, layers=args.layers)
    m = RewardModel(**kw).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
    # windows end at a train event; bootstrap over users when --boot (ensemble diversity)
    from policy import train_index
    tr = train_index()
    rng = np.random.default_rng(args.seed)
    if args.boot:
        from common import user_of_event
        nu = len(ev["offsets"]) - 1
        keep = rng.random(nu) < 0.8
        tr = tr[keep[user_of_event()[tr]]]
    vidx = heldout_idx(1, 30000)
    log, best, t0 = [], 1e9, time.time()
    for step in range(1, args.steps + 1):
        e = rng.choice(tr, args.bs)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = batch_loss(m, e, start=L // 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == args.steps:
            met = metrics_on(predict_events(m, vidx), vidx)
            m.train()
            rec = {"step": step, **parts, **{k: met[k] for k in ("played_mae", "full_auc", "like_auc", "cont_auc", "adopt_auc", "full_gauc")},
                   "min": (time.time() - t0) / 60}
            print(rec, flush=True)
            log.append(rec)
    torch.save({"sd": m.state_dict(), "kw": kw}, CKPT / f"rm_{args.tag}.pt")
    dump(f"train_rm_{args.tag}", {"args": vars(args), "log": log})


def evaluate(args):
    m = load_rm(args.tag)
    idx = heldout_idx(2, 60000, seed=1)
    pred = predict_events(m, idx)
    res = metrics_on(pred, idx)
    print(res)
    np.save(RES / f"rm_pred_test_{args.tag}.npy", pred)
    np.save(RES / "rm_test_idx.npy", idx)
    dump(f"rm_eval_{args.tag}", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--tag", default="rm0")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", action="store_true")
    a = ap.parse_args()
    {"train": train, "eval": evaluate}[a.cmd](a)
