## Литература

Подробный разбор с проверкой деталей — в [`notes/literature.md`](../notes/literature.md). Ниже — то, на что
опирается отчёт.

**Статьи задания:**
1. Petrov, Macdonald. *Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning*, 2024 —
   [arXiv:2403.04875](https://arxiv.org/abs/2403.04875). GPT-2 генерирует список по одному элементу, PPO с
   наградой NDCG + разнообразие/популярность, награда не обучается. Вывод «RL на чистую точность почти ничего не
   даёт» совпадает с нашим.
2. Matveev et al. *Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step
   Off-Policy Correction*, 2026 — [arXiv:2607.02818](https://arxiv.org/abs/2607.02818). Ближайшая работа:
   - Yambda, органика как состояние, награда like + 0,1·ratio;
   - off-policy REINFORCE, модель фидбэка как симулятор;
   - сами авторы признают цикличность оценки.

   Мы взяли у них разделение органики и рекомендаций, награду `paper2` и бейзлайн Positive-CE.

**Методы и явления:**
- Yambda (Яндекс), 2025 — [arXiv:2505.22238](https://arxiv.org/abs/2505.22238); listen+ = ≥ 50 %.
- GRPO: Shao et al., DeepSeekMath, 2024 — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
- KL-регуляризованный оптимум π_ref·exp(r/β), DPO: Rafailov et al., 2023 —
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290); Korbak et al., 2022 —
  [arXiv:2205.11275](https://arxiv.org/abs/2205.11275).
- Переоптимизация proxy vs gold: Gao, Schulman, Hilton, 2022 — [arXiv:2210.10760](https://arxiv.org/abs/2210.10760).
- Ансамбли RM и пессимизм: Coste et al., 2023 — [arXiv:2310.02743](https://arxiv.org/abs/2310.02743);
  Eisenstein et al., 2023 — [arXiv:2312.09244](https://arxiv.org/abs/2312.09244).
- Duration bias: D2Q, Zhan et al., KDD 2022 — [arXiv:2206.06003](https://arxiv.org/abs/2206.06003); OneRec-V2 —
  [arXiv:2508.20900](https://arxiv.org/abs/2508.20900).
- Preference-выравнивание генеративных рекомендателей:
  - OneRec — [arXiv:2502.18965](https://arxiv.org/abs/2502.18965), [arXiv:2506.13695](https://arxiv.org/abs/2506.13695);
  - SPRec (DPO сужает выдачу) — [arXiv:2412.09243](https://arxiv.org/abs/2412.09243);
  - ReRe — [arXiv:2510.12211](https://arxiv.org/abs/2510.12211).
- Удержание как цель RL: RLUR, Cai et al., 2023 — [arXiv:2302.01724](https://arxiv.org/abs/2302.01724).
- Off-policy REINFORCE в рекомендациях: Chen et al., 2019 — [arXiv:1812.02353](https://arxiv.org/abs/1812.02353).
- Петли обратной связи: Jiang et al., 2019 — [arXiv:1902.10730](https://arxiv.org/abs/1902.10730).
- Softmax bottleneck: Yang et al., *Breaking the Softmax Bottleneck*, ICLR 2018 —
  [arXiv:1711.03953](https://arxiv.org/abs/1711.03953).
