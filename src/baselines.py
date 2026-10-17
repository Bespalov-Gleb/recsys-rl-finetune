"""Reward-model-free alternatives that use the logged feedback directly.

posce  "Positive-CE" (the strongest simple baseline in arXiv 2607.02818): continue next-item training of the
       base policy, but only listens with played >= 50% are targets (skips stay in the context).
dpo    DPO on logged pairs: at a recommendation that was skipped (< 10% played) the rejected item is the
       skipped track and the chosen item is the next recommendation of the same user (within 10 events) that
       was listened to >= 80%; prompt = history before the skip.

python src/baselines.py posce --steps 3000
python src/baselines.py dpo --beta 0.1 --steps 1500
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DEV, L, PAD, UNK, cosine_lr, dump, events, to_dev, user_of_event, windows
from policy import CKPT, load_policy, save_policy, train_index
from rl import final_eval
from rm import load_rm


def posce(args):
    torch.manual_seed(0)
    ev = events()
    m = load_policy("base").train()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
    tr = train_index()
    rng = np.random.default_rng(0)
    played = ev["played"]
    for step in range(1, args.steps + 1):
        e = rng.choice(tr, 64)
        w = windows(e, length=L + 1, include_self=True)
        tok = torch.as_tensor(w["tok"]).to(DEV).long()
        pl_ = torch.as_tensor(w["played"]).to(DEV)
        inp, tgt = tok[:, :-1], tok[:, 1:].clone()
        tgt[(tgt <= UNK) | (inp == PAD) | (pl_[:, 1:] < 0.5)] = -100
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr, warm=100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = m.hidden(inp)
            lg = m.logits(h[:, -64:])
        loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt[:, -64:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(step, loss.item(), flush=True)
    kw = torch.load(CKPT / "policy_base.pt", map_location="cpu")["kw"]
    save_policy(m, args.tag, kw)
    final_eval(args.tag, m, load_policy("base").eval(), load_rm("rm0"), [])


def dpo_pairs(max_pairs: int = 2_000_000):
    ev = events()
    tok, pl_, org, sp = ev["tok"], ev["played"], ev["organic"], ev["split"]
    off = ev["offsets"]
    rej = np.flatnonzero((sp == 0) & (org == 0) & (tok > UNK) & (pl_ < 0.1))
    rej = rej[rej - off[user_of_event()[rej]] >= 20]
    rng = np.random.default_rng(0)
    rej = rng.choice(rej, min(len(rej), max_pairs * 2), replace=False)
    end = off[user_of_event()[rej] + 1]
    chosen = np.full(len(rej), -1)
    for d in range(1, 11):
        j = rej + d
        ok = (chosen < 0) & (j < end)
        jj = np.where(ok, j, 0)
        good = ok & (org[jj] == 0) & (tok[jj] > UNK) & (pl_[jj] >= 0.8) & (sp[jj] == 0) & (tok[jj] != tok[rej])
        chosen[good] = tok[jj[good]]
    keep = chosen >= 0
    return rej[keep][:max_pairs], chosen[keep][:max_pairs]


def dpo(args):
    torch.manual_seed(0)
    ev = events()
    rej_e, ch = dpo_pairs()
    print("pairs", len(rej_e), flush=True)
    m = load_policy("base").train()
    ref = load_policy("base").eval()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    log = []
    for step in range(1, args.steps + 1):
        b = rng.integers(0, len(rej_e), 128)
        W = to_dev(windows(rej_e[b]))
        c = torch.as_tensor(ch[b]).to(DEV).long()
        r = torch.as_tensor(ev["tok"][rej_e[b]]).to(DEV).long()
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr, warm=50)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = torch.log_softmax(m.last_logits(W["tok"].long()), -1)
            with torch.no_grad():
                lr_ = torch.log_softmax(ref.last_logits(W["tok"].long()), -1)
        margin = (lp.gather(1, c[:, None]) - lr_.gather(1, c[:, None])) - (lp.gather(1, r[:, None]) - lr_.gather(1, r[:, None]))
        loss = -F.logsigmoid(args.beta * margin).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            acc = (margin > 0).float().mean().item()
            kl = (lp.exp() * (lp - lr_)).sum(-1).mean().item()
            log.append({"step": step, "loss": loss.item(), "acc": acc, "kl": kl})
            print(log[-1], flush=True)
    kw = torch.load(CKPT / "policy_base.pt", map_location="cpu")["kw"]
    save_policy(m, args.tag, kw)
    dump(f"dpo_{args.tag}", {"args": vars(args), "log": log})
    final_eval(args.tag, m, ref, load_rm("rm0"), [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.tag = a.tag or a.cmd
    {"posce": posce, "dpo": dpo}[a.cmd](a)
