"""GPT-2 next-item policy: tokens are track ids (0 = PAD, 1 = UNK, 2.. = 50k tracks).

python src/policy.py train [--steps N] [--tag base]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from common import DEV, L, PAD, RES, UNK, cosine_lr, dump, events, meta, to_dev, windows

CKPT = RES.parent / "checkpoints"
CKPT.mkdir(exist_ok=True)


class Policy(torch.nn.Module):
    def __init__(self, vocab: int, d: int = 256, layers: int = 4, heads: int = 4, drop: float = 0.1):
        super().__init__()
        cfg = GPT2Config(vocab_size=vocab, n_positions=L + 8, n_embd=d, n_layer=layers, n_head=heads,
                         resid_pdrop=drop, embd_pdrop=drop, attn_pdrop=drop)
        self.gpt = GPT2Model(cfg)
        self.cfg = cfg

    def hidden(self, tok: torch.Tensor, past=None, use_cache: bool = False):
        attn = (tok != PAD).long()
        if past is not None:  # incremental decoding: attention mask must cover the cache too
            attn = torch.cat([past[1], attn], 1)
            o = self.gpt(input_ids=tok, attention_mask=attn, past_key_values=past[0], use_cache=True,
                         position_ids=past[2])
            return o.last_hidden_state, (o.past_key_values, attn)
        # left padding -> positions counted over real tokens only
        pos = (attn.cumsum(1) - 1).clamp(min=0)
        o = self.gpt(input_ids=tok, attention_mask=attn, position_ids=pos, use_cache=use_cache)
        return o.last_hidden_state, ((o.past_key_values, attn) if use_cache else None)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        lg = h @ self.gpt.wte.weight.T  # tied output embedding
        lg[..., PAD] = -1e4
        lg[..., UNK] = -1e4  # never recommend the out-of-vocabulary bucket
        return lg

    def last_logits(self, tok: torch.Tensor) -> torch.Tensor:
        h, _ = self.hidden(tok)
        return self.logits(h[:, -1]).float()


class CrossHeadPolicy(torch.nn.Module):
    """Frozen GPT-2 + a small head that re-scores the reference top-K candidates with "user x track" features.

    logit(a|s) = logit_ref(a|s) + head(cross(s, a), item(a), log pi_ref(a|s)) for a in top-K, -inf outside.
    The head sees the same history-x-candidate features as the reward model (play count of the track in the history,
    its mean played ratio, recency, the same for the artist, ...), which a dot-product output layer cannot compute.
    The last layer starts at zero, so training starts exactly at pi_ref."""

    def __init__(self, ref: "Policy", k: int = 1000, hidden: int = 64):
        super().__init__()
        from common import vocab
        from rm import N_CROSS
        self.ref, self.k = ref, k
        for p in self.ref.parameters():
            p.requires_grad_(False)
        voc = vocab()
        self.register_buffer("art_of", torch.as_tensor(voc["tok_artist"]).long())
        self.register_buffer("loglen", torch.log(torch.as_tensor(voc["tok_len"]).clamp(min=10)) - 5.3)
        self.register_buffer("logpop", torch.log1p(torch.as_tensor(voc["tok_pop"])) / 10)
        self.head = torch.nn.Sequential(torch.nn.Linear(N_CROSS + 3, hidden), torch.nn.GELU(),
                                        torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.ref.eval()
        return self

    def logits_w(self, W: dict) -> torch.Tensor:
        from rm import cross_feats
        tok, played, mask = W["tok"].long(), W["played"].float(), W["mask"]
        with torch.no_grad():
            lr = self.ref.last_logits(tok)
            cand = lr.topk(self.k).indices
            lpr = torch.log_softmax(lr, -1).gather(1, cand)
            allowed = mask[:, None, :].expand(-1, cand.shape[1], -1)
            q = torch.full(cand.shape, float(tok.shape[1]), device=tok.device)
            cr = cross_feats(cand, q, tok, played, self.art_of, allowed)
            x = torch.cat([cr, self.loglen[cand][..., None], self.logpop[cand][..., None], (lpr / 10)[..., None]], -1)
        delta = self.head(x.float())[..., 0]
        out = torch.full_like(lr, -1e4)
        return out.scatter(1, cand, lr.gather(1, cand) + delta)

    def last_logits_w(self, W: dict) -> torch.Tensor:
        return self.logits_w(W)


def load_policy(tag: str = "base"):
    ck = torch.load(CKPT / f"policy_{tag}.pt", map_location="cpu")
    if ck.get("cross"):
        m = CrossHeadPolicy(load_policy(ck["ref"]), k=ck["k"])
        m.head.load_state_dict(ck["head"])
        return m.to(DEV)
    m = Policy(**ck["kw"])
    m.load_state_dict(ck["sd"])
    return m.to(DEV)


def save_policy(m: Policy, tag: str, kw: dict) -> None:
    torch.save({"sd": m.state_dict(), "kw": kw}, CKPT / f"policy_{tag}.pt")


def train_index() -> np.ndarray:
    """Event indices usable as the LAST target of a training window: train split, >= 2 events of history."""
    ev = events()
    off = ev["offsets"]
    first = np.zeros(len(ev["tok"]), bool)
    first[off[:-1]] = True
    return np.flatnonzero((ev["split"] == 0) & ~first)


@torch.no_grad()
def val_loss(m: Policy, idx: np.ndarray) -> dict:
    m.eval()
    tot, hit, n = 0.0, 0.0, 0
    for i in range(0, len(idx), 512):
        w = to_dev(windows(idx[i:i + 512]))
        tgt = torch.as_tensor(events()["tok"][idx[i:i + 512]]).to(DEV).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m.last_logits(w["tok"].long())
        tot += F.cross_entropy(lg, tgt, reduction="sum").item()
        hit += (lg.topk(10).indices == tgt[:, None]).any(1).sum().item()
        n += len(tgt)
    m.train()
    return {"ce": tot / n, "hr10": hit / n}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ev = events()
    V = meta()["vocab"] + 2
    kw = dict(vocab=V, d=args.d, layers=args.layers, heads=args.heads, drop=args.drop)
    m = Policy(**kw).to(DEV)
    print("params", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    tr = train_index()
    rng = np.random.default_rng(args.seed)
    from common import eval_points
    vidx = eval_points(1, per_user=2)
    tok_all = ev["tok"]
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        e = rng.choice(tr, args.bs)
        w = windows(e, length=L + 1, include_self=True)  # L+1 events ending at e -> L inputs, L targets
        tok = torch.as_tensor(w["tok"]).to(DEV).long()
        inp, tgt = tok[:, :-1], tok[:, 1:].clone()
        tgt[(tgt == UNK) | (inp == PAD)] = -100
        tgt[tgt == PAD] = -100
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = m.hidden(inp)
            # loss on the last P positions only: the full (L x 50k) logit tensor does not fit into 8 GB,
            # and late positions (long context) are the ones the policy is used at
            lg = m.logits(h[:, -args.P:])
        loss = F.cross_entropy(lg.float().reshape(-1, V), tgt[:, -args.P:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps:
            vl = val_loss(m, vidx[:6000])
            rec = {"step": step, "train_ce": loss.item(), **vl, "min": (time.time() - t0) / 60}
            log.append(rec)
            print(rec, flush=True)
    save_policy(m, args.tag, kw)
    dump(f"train_policy_{args.tag}", {"args": vars(args), "log": log})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
