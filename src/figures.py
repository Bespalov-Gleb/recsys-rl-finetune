"""Figures for the report (reads results/*.json, writes figures/*.png).

python src/figures.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RES

FIG = RES.parent / "figures"
FIG.mkdir(exist_ok=True)
COL = {"played": "#1f77b4", "seconds": "#ff7f0e", "full": "#2ca02c", "like": "#d62728", "played_db": "#9467bd",
       "lt": "#8c564b", "paper2": "#e377c2"}
NAMES = {"played": "доля прослушивания", "seconds": "секунды прослушивания", "full": "P(listen+)", "like": "P(like)",
         "played_db": "доля − база по длине", "lt": "долгосрочная (like+adopt+cont)", "paper2": "like + 0.1·доля"}
BETAS = ["3.0", "1.0", "0.5", "0.3", "0.2", "0.1", "0.05", "0.0"]


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def sweep_rows(split, reward, suffix=""):
    rows = [load(f"eval{split}_base")]
    rows += [load(f"eval{split}_exact_{reward}_b{b}{suffix}") for b in BETAS]
    return [r for r in rows if r]


def fig_tradeoff(split=1):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for rw in COL:
        rows = sweep_rows(split, rw)
        if len(rows) < 2:
            continue
        x = [r["NDCG@10_all"] for r in rows]
        axes[0].plot(x, [r["gold_played"] for r in rows], "o-", color=COL[rw], label=NAMES[rw], ms=3)
        axes[1].plot(x, [r["gold_full"] for r in rows], "o-", color=COL[rw], ms=3)
        axes[2].plot(x, [r["gold_like"] for r in rows], "o-", color=COL[rw], ms=3)
    for ax, t in zip(axes, ["gold: доля прослушивания топ-10", "gold: P(listen+) топ-10", "gold: P(like) топ-10"]):
        ax.set_xlabel("NDCG@10 (все прослушивания за 24 ч)")
        ax.set_title(t)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Аналитический KL-оптимум π* ∝ π_ref·exp(r/β): от base (справа) до β=0 (слева), валидация")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff_exact.png", dpi=130)


def fig_behaviour(split=1):
    keys = [("mean_len_s", "средняя длина трека, с"), ("repeat_share", "доля уже слушанных"),
            ("coverage", "coverage каталога"), ("mean_logpop", "log популярности"), ("pair_jaccard", "Jaccard списков разных польз."),
            ("Recall@10_skip", "Recall@10 по скипам")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for rw in COL:
        rows = sweep_rows(split, rw)
        if len(rows) < 2:
            continue
        xb = [0] + [1 / float(b) if float(b) > 0 else 30 for b in BETAS][: len(rows) - 1]
        for ax, (k, t) in zip(axes.ravel(), keys):
            ax.plot(xb, [r[k] for r in rows], "o-", color=COL[rw], label=NAMES[rw], ms=3)
            ax.set_title(t)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel("1/β (0 = base, 30 = чистая максимизация награды)")
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "behaviour_exact.png", dpi=130)


def fig_rl_curves(tags, fname, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for tag in tags:
        d = load(f"rl_{tag}")
        if not d:
            continue
        log = d["log"]
        x = [math.sqrt(max(r["kl_ref"], 0)) for r in log]
        axes[0].plot(x, [r["proxy_r"] for r in log], "o-", ms=3, label=tag)
        axes[1].plot(x, [r["gold_r"] for r in log], "o-", ms=3, label=tag)
        axes[2].plot(x, [r["hr10"] for r in log], "o-", ms=3, label=tag)
    for ax, t in zip(axes, ["proxy-награда топ-10 (норм.)", "gold-награда топ-10 (норм.)", "HR@10 next item"]):
        ax.set_xlabel("√KL(π‖π_ref)")
        ax.set_title(t)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)


def fig_multistep(tags):
    d = {t: load(f"multistep_{t}") for t in tags}
    d = {k: v for k, v in d.items() if v}
    if not d:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    steps = range(1, 6)
    for t, o in d.items():
        for mode, ls in (("teacher", "-"), ("static", ":"), ("greedy", "--"), ("sample", "-.")):
            axes[0].plot(steps, o[mode]["hr10"], ls, marker="o", ms=3, label=f"{t}/{mode}")
        for mode, ls in (("greedy", "--"), ("sample", "-.")):
            axes[1].plot(steps, o[mode]["r_imagined"], ls, marker="o", ms=3, label=f"{t}/{mode}: воображаемый фидбэк")
            axes[1].plot(steps, o[mode]["r_real"], ls, marker="x", ms=3, label=f"{t}/{mode}: только реальная история")
            axes[1].plot(steps, o[mode]["gold_played"], ls, marker="s", ms=3, label=f"{t}/{mode}: gold")
            axes[2].plot(steps, o[mode]["loop"], ls, marker="o", ms=3, label=f"{t}/{mode}: повтор сгенерированного")
            axes[2].plot(steps, o[mode]["same_artist_prev"], ls, marker="x", ms=3, label=f"{t}/{mode}: тот же артист")
    axes[0].set_title("HR@10 к реальному k-му прослушиванию")
    axes[1].set_title("ожидаемая доля прослушивания k-го трека")
    axes[2].set_title("зацикливание")
    for ax in axes:
        ax.set_xlabel("шаг k")
        ax.grid(alpha=.3)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "multistep.png", dpi=130)


def fig_frontier(split=1):
    """Exact optimum vs trained policies on the same axes: gold reward and HR@10 against KL(pi||pi_ref)."""
    ex = [r for r in sweep_rows(split, "played") if r.get("kl_ref") == r.get("kl_ref")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    kx = lambda r: max(r["kl_ref"], 1e-3)
    axes[0].plot([kx(r) for r in ex], [r["gold_played"] for r in ex], "o-", color="k", label="аналитический π* (β-sweep)")
    axes[1].plot([kx(r) for r in ex], [r["HR@10"] for r in ex], "o-", color="k")
    axes[2].plot([r["HR@10"] for r in ex], [r["gold_played"] for r in ex], "o-", color="k", label="π*, β-sweep")
    for ps, c in (("1.0", "#2ca02c"), ("3.0", "#d62728")):
        rows = [load(f"eval{split}_exact_played_b{b}_p{ps}") for b in ("0.3", "0.1", "0.05", "0.0")]
        rows = [r for r in rows if r]
        if rows:
            axes[2].plot([r["HR@10"] for r in rows], [r["gold_played"] for r in rows], "s--", color=c, ms=4,
                         label=f"π* с пессимизмом ансамбля (k={ps})")
    runs = {"grpo_played_b1.0": "GRPO β=1", "grpo_played_b0.3": "GRPO β=0.3", "grpo_played_b0.1": "GRPO β=0.1",
            "grpo_played_b0.3_long": "GRPO β=0.3, lr×5", "grpo_played_b0.3_long_G64": "GRPO β=0.3, lr×5, G=64",
            "epg_played_b0.3": "точный PG β=0.3", "grpo_played_b0.3_frozen": "GRPO β=0.3, эмбеддинги заморожены",
            "grpo_played_b0.1_pess1": "GRPO β=0.1 + пессимизм", "grpo_played_b0.1_top1000": "GRPO β=0.1, топ-1000",
            "distill_played_b0.3": "дистилляция π* β=0.3", "distill_played_b0.1": "дистилляция π* β=0.1",
            "posce": "Positive-CE", "dpo": "DPO β=0.1", "dpo_b1": "DPO β=1",
            "cross_played_b1.0": "cross-голова, GRPO β=1", "cross_played_b0.3": "cross-голова, GRPO β=0.3",
            "cross_played_b0.1": "cross-голова, GRPO β=0.1", "cross_distill_played_b0.3": "cross-голова, дистилляция β=0.3"}
    mk = iter("v^<>DPXhd*p8Hso*PXD")
    for tag, lab in runs.items():
        r = load(f"eval{split}_{tag}")
        if not r or "kl_ref" not in r:
            continue
        m = next(mk)
        st = dict(ms=11, mec="k", mew=1.2) if tag.startswith("cross") else dict(ms=8)
        axes[0].plot(kx(r), r["gold_played"], m, label=lab, **st)
        axes[1].plot(kx(r), r["HR@10"], m, **st)
        axes[2].plot(r["HR@10"], r["gold_played"], m, **st)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("KL(π‖π_ref), лог. шкала")
        ax.grid(alpha=.3)
    axes[0].set_title("gold: доля прослушивания топ-10")
    axes[1].set_title("HR@10 (next item)")
    axes[2].set_title("фронт: релевантность vs gold-награда")
    axes[2].set_xlabel("HR@10")
    axes[2].grid(alpha=.3)
    axes[0].legend(fontsize=6)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "frontier_rl_vs_exact.png", dpi=130)


if __name__ == "__main__":
    fig_tradeoff()
    fig_behaviour()
    fig_rl_curves([f"grpo_played_b{b}" for b in ("1.0", "0.3", "0.1", "0.0")] +
                  ["grpo_played_b0.3_long", "epg_played_b0.3", "distill_played_b0.3"], "rl_curves_played.png",
                  "Кривые обучения на доле прослушивания (точки каждые 250-500 шагов, валидация)")
    fig_multistep(["base"])
    fig_frontier()
