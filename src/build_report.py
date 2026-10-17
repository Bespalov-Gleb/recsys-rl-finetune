"""REPORT.md (repo root) from report/*.md parts + REPORT.html with embedded figures.

python src/build_report.py
"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["part1_intro", "draft_setup", "draft_rm", "part3_exact", "part4_rl", "part4b_cross", "part5_norm",
         "draft_longterm", "part6_multistep", "part8_test", "part8b_robust", "part8c_control", "part7_end", "part9_refs"]

md = "\n\n".join((ROOT / "report" / f"{p}.md").read_text(encoding="utf-8").strip() for p in PARTS) + "\n"
md = md.replace("](../", "](")
(ROOT / "REPORT.md").write_text(md, encoding="utf-8", newline="\n")

html = markdown.markdown(md, extensions=["tables", "fenced_code"])


def embed(m):
    p = ROOT / m.group(1)
    if p.exists() and p.suffix == ".png":
        return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
    return m.group(0)


html = re.sub(r'src="([^"]+)"', embed, html)
css = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;
line-height:1.5;color:#1b1b1b}table{border-collapse:collapse;margin:12px 0;font-size:13px;display:block;overflow-x:auto}
th,td{border:1px solid #ccc;padding:3px 7px;text-align:left}th{background:#f3f3f3}img{max-width:100%}
code{background:#f5f5f5;padding:1px 4px;border-radius:3px}pre{background:#f5f5f5;padding:10px;overflow-x:auto}
h1,h2,h3{line-height:1.25}"""
(ROOT / "REPORT.html").write_text(f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>RL-дообучение '
                                   f'рекомендателя</title><style>{css}</style></head><body>{html}</body></html>',
                                   encoding="utf-8")
print(len(md.split()), "words")
