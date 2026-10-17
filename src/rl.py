"""RL fine-tuning of the GPT-2 policy against the learned reward model.

Algorithm: on-policy policy gradient with a group baseline (GRPO-style, one gradient step per batch, so no
ratio clipping is needed): for every context G items (or G sequences of T items) are sampled from the current
policy, scored by the reward model, the advantage is the reward minus the group mean, and the exact
KL(pi_theta || pi_ref) over the whole 50k vocabulary is added as a penalty (exact, not a sampled estimate --
the action space is enumerable, unlike for LLMs).

Rewards are normalised (r - mu) / sigma with mu, sigma measured under the reference policy, so that beta has
the same meaning for every reward and matches the analytic optimum pi* ~ pi_ref * exp(r / beta).

python src/rl.py --reward played --beta 0.1 --tag grpo_played_b0.1
python src/rl.py --reward full --T 5 --rm_ctx imagined --tag seq5_full
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common import DEV, L, RES, UNK, cosine_lr, dump, events, to_dev, user_of_event, vocab, windows
from policy import CKPT, load_policy, save_policy
from rm import HEADS, load_rm

D = 86_400
NORM_FILE = RES / "reward_norm.json"
H = {h: i for i, h in enumerate(HEADS)}


def len_baseline() -> np.ndarray:
    """E[played | track-length decile] over train recommendation listens, indexed by token."""
    voc = vocab()
    ev = events()
    s = np.flatnonzero((ev["split"] == 0) & (ev["organic"] == 0) & (ev["tok"] > UNK))[::7]
    ln = voc["tok_len"][ev["tok"][s]]
    edges = np.quantile(ln, np.linspace(0, 1, 11)[1:-1])
    b = np.digitize(ln, edges)
    m = np.array([ev["played"][s][b == i].mean() for i in range(10)])
    return m[np.digitize(voc["tok_len"], edges)].astype(np.float32)


class Reward:
    """Raw reward components from RM probabilities; composites are combined after normalisation."""
    COMPONENTS = ["played", "seconds", "full", "like", "cont", "adopt_new", "played_db"]
    COMPOSITES = {"lt": {"like": 1 / 3, "adopt_new": 1 / 3, "cont": 1 / 3},
                  "paper2": {"like": 1.0, "played": 0.1}}  # like + 0.1 * ratio as in arXiv 2607.02818 (raw scale)

    def __init__(self, name: str, norm: dict | None):
        self.name, self.norm = name, norm
        voc = vocab()
        self.tlen = torch.as_tensor(voc["tok_len"]).to(DEV)
        self.lb = torch.as_tensor(len_baseline()).to(DEV)

    def components(self, P: torch.Tensor, cand: torch.Tensor, new: torch.Tensor) -> dict:
        return {"played": P[..., 0], "seconds": P[..., 0] * self.tlen[cand] / 200.0, "full": P[..., 1],
                "like": P[..., 2], "cont": P[..., 3], "adopt_new": P[..., 4] * new.float(),
                "played_db": P[..., 0] - self.lb[cand]}

    def __call__(self, P, cand, new):
        c = self.components(P, cand, new)
        if self.name == "paper2":
            r = c["like"] + 0.1 * c["played"]
            mu, sd = self.norm["paper2"]
            return (r - mu) / sd
        if self.name in self.COMPOSITES:
            return sum(w * (c[k] - self.norm[k][0]) / self.norm[k][1] for k, w in self.COMPOSITES[self.name].items())
        mu, sd = self.norm[self.name]
        return (c[self.name] - mu) / sd


def is_new(cand: torch.Tensor, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ~((cand[:, :, None] == tok[:, None, :]) & mask[:, None, :]).any(-1)


class SupportMasked(torch.nn.Module):
    """Two-stage policy: candidates = top-K of the frozen reference (retrieval), ranking = the trained policy.

    Keeps the RL policy inside the support where the logs (and therefore the reward model) have data."""

    def __init__(self, policy, ref, k: int):
        super().__init__()
        self.policy, self.ref, self.k = policy, ref, k

    def mask(self, tok, lg):
        with torch.no_grad():
            top = self.ref.last_logits(tok).topk(self.k).indices
        m = torch.full_like(lg, -1e4)
        m.scatter_(1, top, 0.0)
        return lg + m

    def last_logits(self, tok):
        return self.mask(tok, self.policy.last_logits(tok))

    def train(self, mode: bool = True):  # the frozen reference always stays in eval mode
        self.policy.train(mode)
        self.ref.eval()
        return self


def rl_contexts() -> np.ndarray:
    """Decision points for RL: recommendation (non-organic) listens in the train split, >= 20 events of history."""
    ev = events()
    ok = np.flatnonzero((ev["split"] == 0) & (ev["organic"] == 0))
    return ok[ok - ev["offsets"][user_of_event()[ok]] >= 20]


@torch.no_grad()
def calibrate(rm_tag: str = "rm0", n: int = 4096, G: int = 32) -> dict:
    """mu / sigma of every reward component for items sampled from the reference policy."""
    allnorm = json.loads(NORM_FILE.read_text()) if NORM_FILE.exists() else {}
    if rm_tag in allnorm:
        return allnorm[rm_tag]
    ref, rm = load_policy("base").eval(), load_rm(rm_tag)
    R = Reward("played", None)
    rng = np.random.default_rng(123)
    e = rng.choice(rl_contexts(), n, replace=False)
    acc = {k: [] for k in Reward.COMPONENTS + ["paper2"]}
    for i in range(0, n, 128):
        W = to_dev(windows(e[i:i + 128]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = ref.last_logits(W["tok"].long())
            a = torch.multinomial(torch.softmax(lg, -1), G, replacement=True)
            P = torch.sigmoid(rm.score(W, a).float())
        c = R.components(P, a, is_new(a, W["tok"].long(), W["mask"]))
        c["paper2"] = c["like"] + 0.1 * c["played"]
        for k in acc:
            acc[k].append(c[k].flatten().cpu().numpy())
    norm = {k: (float(np.concatenate(v).mean()), float(np.concatenate(v).std() + 1e-6)) for k, v in acc.items()}
    allnorm[rm_tag] = norm
    NORM_FILE.write_text(json.dumps(allnorm, indent=1))
    return norm


@torch.no_grad()
def quick_val(policy, ref, rms, R: Reward, idx: np.ndarray, gold_eval: bool) -> dict:
    """Cheap curve point: HR@10, KL, entropy, proxy and gold reward of the top-10 list on val contexts."""
    from evaluate import policy_topk
    recs, d = policy_topk(policy, idx, K=10, ref=ref)
    hit = (recs == events()["tok"][idx][:, None]).any(1).mean()
    out = {"hr10": float(hit), **d}
    rs, raw = [], []
    for m in rms:
        res = []
        for i in range(0, len(idx), 256):
            W = to_dev(windows(idx[i:i + 256]))
            c = torch.as_tensor(recs[i:i + 256]).to(DEV).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P = torch.sigmoid(m.score(W, c).float())
            res.append(R(P, c, is_new(c, W["tok"].long(), W["mask"])).mean().item())
            raw.append(P.mean((0, 1)).cpu().numpy())
        rs.append(np.mean(res))
    out["proxy_r"] = float(rs[0])
    if len(rs) > 1:
        out["ens_r"] = float(np.mean(rs[1:]))
    out.update({f"proxy_{h}": float(v) for h, v in zip(HEADS, np.mean(raw[:len(raw) // len(rms)], 0))})
    if gold_eval:
        from gold import gold_score
        G = gold_score(idx, recs)
        Gt = torch.as_tensor(G).to(DEV).float()
        c = torch.as_tensor(recs).to(DEV).long()
        W = windows(idx)
        new = is_new(c, torch.as_tensor(W["tok"]).to(DEV).long(), torch.as_tensor(W["mask"]).to(DEV))
        out["gold_r"] = float(R(Gt, c, new).mean().item())
        out.update({f"gold_{h}": float(G[..., i].mean()) for i, h in enumerate(HEADS)})
    policy.train()
    return out


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    norm = calibrate(args.rm)
    R = Reward(args.reward, norm)
    if args.policy == "cross":
        from policy import CrossHeadPolicy
        args.topk = args.topk or 1000
        policy = CrossHeadPolicy(load_policy(args.init), k=args.topk).to(DEV).train()
    else:
        policy = load_policy(args.init).train()
    ref = load_policy("base").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rm = load_rm(args.rm)
    ens = [load_rm(t) for t in args.ens.split(",")] if args.ens else []
    for m in [rm] + ens:
        for p in m.parameters():
            p.requires_grad_(False)
    if args.freeze_emb:  # item embeddings are tied to the output layer: freeze them, train only the transformer
        policy.gpt.wte.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    ctx = rl_contexts()
    from evaluate import eval_set
    vidx = eval_set(1, 1)[0][: args.nval]
    log, t0 = [], time.time()
    kw = torch.load(CKPT / f"policy_{args.init}.pt", map_location="cpu")["kw"]

    evalpol = SupportMasked(policy, ref, args.topk) if args.topk and args.policy != "cross" else policy
    evalref = SupportMasked(ref, ref, args.topk) if args.topk else ref  # KL measured inside the same support

    def curve_point(step):
        rec = {"step": step, **quick_val(evalpol, evalref, [rm] + ens, R, vidx, gold_eval=not args.no_gold), "min": (time.time() - t0) / 60}
        log.append(rec)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

    curve_point(0)
    for step in range(1, args.steps + 1):
        e = rng.choice(ctx, args.B)
        W = to_dev(windows(e))
        tok, mask = W["tok"].long(), W["mask"]
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr, warm=50)
        if args.T == 1:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.policy == "cross":
                    lg = policy.logits_w(W).float()
                else:
                    h, _ = policy.hidden(tok)
                    lg = policy.logits(h[:, -1]).float()
                with torch.no_grad():
                    lr_ = ref.last_logits(tok)
            if args.topk:
                top = lr_.topk(args.topk).indices
                m = torch.full_like(lr_, -1e4).scatter_(1, top, 0.0)
                lg, lr_ = lg + m, lr_ + m
            lp, lpr = torch.log_softmax(lg, -1), torch.log_softmax(lr_, -1)
            if args.pg == "distill":
                # supervised upper bound: cross-entropy to the analytic optimum pi* = pi_ref * exp(r / beta) / Z
                # (r on the reference top-N, 0 elsewhere) -- can the network represent what RL is asked to find?
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    cand = lr_.topk(args.ncand).indices
                    Pc = torch.sigmoid(rm.score(W, cand).float())
                    rc = R(Pc, cand, is_new(cand, tok, mask)).float()
                tgt = torch.softmax(lr_.scatter_add(1, cand, rc / args.beta), -1)
                kl = (lp.exp() * (lp - lpr)).sum(-1)
                loss = -(tgt * lp).sum(-1).mean()
                r_mean = 0.0
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                if step % 50 == 0:
                    print(f"step {step} kl {kl.mean().item():.4f} ce {loss.item():.4f}", flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    curve_point(step)
                continue
            if args.pg == "exact":
                # exact policy gradient: E_pi[r] summed over the reference top-N candidates (no sampling noise);
                # items outside keep the neutral normalised reward 0, as in the analytic optimum
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    cand = lr_.topk(args.ncand).indices
                    Pc = torch.sigmoid(rm.score(W, cand).float())
                    rc = R(Pc, cand, is_new(cand, tok, mask)).float()
                kl = (lp.exp() * (lp - lpr)).sum(-1)
                loss = -(lp.exp().gather(1, cand) * rc).sum(1).mean() + args.beta * kl.mean()
                r_mean = (torch.softmax(lpr.gather(1, cand), 1) * rc).sum(1).mean().item()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                if step % 50 == 0:
                    print(f"step {step} kl {kl.mean().item():.4f} loss {loss.item():.4f}", flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    curve_point(step)
                continue
            with torch.no_grad():
                a = torch.multinomial(lp.exp(), args.G, replacement=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    P = torch.sigmoid(rm.score(W, a).float())
                    new = is_new(a, tok, mask)
                    r = R(P, a, new)
                    if ens and args.pess > 0:
                        rr = torch.stack([R(torch.sigmoid(m.score(W, a).float()), a, new) for m in ens])
                        r = rr.mean(0) - args.pess * rr.std(0)
                adv = r - r.mean(1, keepdim=True)
            logp = lp.gather(1, a)
            kl = (lp.exp() * (lp - lpr)).sum(-1)
            loss = -(adv * logp).mean() + args.beta * kl.mean()
            r_mean = r.mean().item()
        else:
            loss, kl, r_mean = multistep_loss(policy, ref, rm, R, W, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} r {r_mean:.4f} kl {kl.mean().item():.4f} loss {loss.item():.4f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            curve_point(step)
    if args.policy == "cross":
        torch.save({"cross": True, "head": policy.head.state_dict(), "k": policy.k, "ref": args.init},
                   CKPT / f"policy_{args.tag}.pt")
    else:
        save_policy(policy, args.tag, kw)
    dump(f"rl_{args.tag}", {"args": vars(args), "log": log})
    if not args.no_gold:
        final_eval(args.tag, evalpol, evalref, rm, ens)


def append(W: dict, a: torch.Tensor, played: torch.Tensor) -> dict:
    """Append a generated (recommended) event to an RM window (grows it by one; callers keep the total <= L)."""
    one = torch.ones_like(a, dtype=W["mask"].dtype)
    return {"tok": torch.cat([W["tok"], a[:, None].to(W["tok"].dtype)], 1),
            "played": torch.cat([W["played"], played[:, None].to(W["played"].dtype)], 1),
            "organic": torch.cat([W["organic"], torch.zeros_like(a)[:, None].to(W["organic"].dtype)], 1),
            "mask": torch.cat([W["mask"], one[:, None]], 1)}


def multistep_loss(policy, ref, rm, R, W, args):
    """Sequence-level GRPO: G sequences of T items per context, each generated item appended to the context.

    rm_ctx = real:     every item is scored by the RM against the REAL history only (independent items);
    rm_ctx = imagined: generated items enter the RM history with the RM's own predicted played ratio
                       ("the user listened as the model expects")."""
    B, G, T = W["tok"].shape[0], args.G, args.T
    # history is cut to the last L - T events so that history + T generated items never exceeds the L positions
    # both models were trained on (positions >= L have untrained embeddings)
    Wg = {k: v[:, T:].repeat_interleave(G, 0) for k, v in W.items()}
    seq = Wg["tok"].long()
    Wrm = Wg
    rewards, acts = [], []
    art_of = rm.art_of
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(T):
            lg = policy.last_logits(seq)
            a = torch.multinomial(torch.softmax(lg, -1), 1)[:, 0]
            Wscore = Wrm if args.rm_ctx == "imagined" else Wg
            P = torch.sigmoid(rm.score(Wscore, a[:, None]).float())[:, 0]
            new = is_new(a[:, None], Wscore["tok"].long(), Wscore["mask"])[:, 0]
            r = R(P[:, None], a[:, None], new[:, None])[:, 0]
            if k > 0:  # a repeat inside the generated list adds nothing new
                dup = (torch.stack(acts, 1) == a[:, None]).any(1)
                r = torch.where(dup, torch.full_like(r, args.dup_r), r)
            rewards.append(r)
            acts.append(a)
            seq = torch.cat([seq, a[:, None]], 1)
            Wrm = append(Wrm, a, P[:, 0])
    A = torch.stack(acts, 1)
    Rt = torch.stack(rewards, 1).sum(1)
    if args.div > 0:
        n_art = torch.tensor([len(set(x)) for x in art_of[A].tolist()], device=DEV, dtype=torch.float32)
        Rt = Rt + args.div * n_art
    Rg = Rt.view(B, G)
    adv = (Rg - Rg.mean(1, keepdim=True)).view(-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, _ = policy.hidden(seq[:, :-1])
        lg = policy.logits(h[:, -T:]).float()
        with torch.no_grad():
            hr, _ = ref.hidden(seq[:, :-1])
            lr_ = ref.logits(hr[:, -T:]).float()
    lp, lpr = torch.log_softmax(lg, -1), torch.log_softmax(lr_, -1)
    logp = lp.gather(2, A[:, :, None])[..., 0].sum(1)
    kl = (lp.exp() * (lp - lpr)).sum(-1).sum(1)  # sum over steps
    loss = -(adv * logp).mean() / T + args.beta * kl.mean() / T
    return loss, kl / T, Rt.mean().item()


def final_eval(tag, policy, ref, rm, ens):
    from evaluate import full_eval, policy_topk, eval_set
    for split, pu in ((1, 1), (2, 3)):
        idx = eval_set(split, pu)[0]
        recs, d = policy_topk(policy, idx, K=100, ref=ref)
        full_eval(tag, recs, split=split, per_user=pu, extra=d, rms=[rm] + ens)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="played")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--T", type=int, default=1)
    ap.add_argument("--rm_ctx", default="real")
    ap.add_argument("--div", type=float, default=0.0)
    ap.add_argument("--dup_r", type=float, default=-1.0)
    ap.add_argument("--rm", default="rm0")
    ap.add_argument("--ens", default="")
    ap.add_argument("--pess", type=float, default=0.0)
    ap.add_argument("--init", default="base")
    ap.add_argument("--nval", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_gold", action="store_true")
    ap.add_argument("--pg", default="sample", help="sample (GRPO) | exact (sum over the reference top-N)")
    ap.add_argument("--ncand", type=int, default=1000)
    ap.add_argument("--freeze_emb", action="store_true")
    ap.add_argument("--policy", default="gpt", help="gpt (fine-tune GPT-2) | cross (frozen GPT-2 + cross-feature head)")
    ap.add_argument("--topk", type=int, default=0, help="restrict the policy to the reference top-K (0 = whole catalogue)")
    ap.add_argument("--tag", required=True)
    train(ap.parse_args())
