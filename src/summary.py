"""Markdown tables from results/eval*.json.

python src/summary.py <split> <name-glob> [columns preset]
"""
import fnmatch
import json
import sys

from common import RES

PRESETS = {
    "rel": ["HR@10", "NDCG@10", "Recall@10_all", "Recall@10_full", "Recall@10_skip", "Recall@10_like", "Recall@10_adopt",
            "MAP@10_full", "NDCG@10_full"],
    "rew": ["proxy_played", "gold_played", "proxy_full", "gold_full", "proxy_like", "gold_like", "gold_adopt", "gold_cont",
            "gold_seconds", "kl_ref"],
    "beh": ["mean_len_s", "repeat_share", "coverage", "gini", "top1_share", "pair_jaccard", "mean_logpop", "artists_per_list"],
}


def rows(split, pattern):
    out = []
    for p in sorted(RES.glob(f"eval{split}_*.json")):
        name = p.stem[len(f"eval{split}_"):]
        if fnmatch.fnmatch(name, pattern):
            out.append((name, json.loads(p.read_text())))
    return out


def table(split, pattern, cols, order=None):
    rs = rows(split, pattern)
    if order:
        rs.sort(key=lambda x: order(x[0]))
    head = "| модель | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for name, r in rs:
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append("—" if v is None or v != v else (f"{v:.1f}" if c == "mean_len_s" else f"{v:.4f}"))
        body += f"| {name} | " + " | ".join(vals) + " |\n"
    return head + body


if __name__ == "__main__":
    split, pat = int(sys.argv[1]), sys.argv[2]
    cols = PRESETS.get(sys.argv[3], sys.argv[3].split(",")) if len(sys.argv) > 3 else PRESETS["rel"]
    print(table(split, pat, cols))
