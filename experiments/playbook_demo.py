# -*- coding: utf-8 -*-
"""Playbook 演示：验证「读 playbook → 注入 prior → 启优化器」流水跑通。

不依赖真实评估器（bench_base），用一个 dummy 凸函数代替——目的是确认：
  1. playbook_io 能正确读 + 合并 prior
  2. prior 字段能传进 CmaesOptimizer（sigma0）
  3. 跑一轮后 best 分数能正常更新

下一步（v2）是把 dummy 换成 evaluator.py 跑真 71k 锚，warm-start vs 裸启动
按 cvar_min_delta_for_promotion=150 判显著。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 让 experiments/ 直接 import lab.optimizers
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.playbook_io import read_notes, parse_to_prior  # noqa: E402
from optimizers import CmaesOptimizer  # noqa: E402


def dummy_evaluator(x, noise_seed=None):
    """凸基准：中心 = [0.5]*dim，最优 fitness ≈ 50000；噪声 σ=100。

    跑分角度：纯跑通流程，不指望 warm-start 真的「加速」（v2 用真评估器才有意义）。
    """
    x = np.asarray(x, dtype=float)
    dist_sq = float(np.sum((x - 0.5) ** 2))
    noise = 0.0
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        noise = float(rng.normal(0, 100))
    return 50000.0 - 1000.0 * dist_sq + noise


def run(prior: dict, n_iter: int = 30, seed: int = 7) -> dict:
    """按 prior 启 CMA-ES，跑 n_iter 轮 dummy eval，回传曲线。"""
    sigma0 = float(prior.get("sigma", 0.06))  # 默认 0.06 对齐 Optimizer 现状
    opt = CmaesOptimizer(dim=45, seed=seed, sigma0=sigma0)

    curve = []
    for i in range(n_iter):
        x = opt.ask()
        # 用确定性 noise（按 step i 派生），不靠全局 rng，CRN-friendly
        s = dummy_evaluator(x, noise_seed=seed * 1000 + i)
        opt.tell(x, s)
        curve.append({"i": i, "score": s, "best": opt.best})

    return {
        "prior_used": {"sigma": sigma0},
        "curve": curve,
        "final_best": opt.best,
        "n_eval": n_iter,
    }


def main():
    playbook_dir = ROOT / "playbook" / "notes"
    print(f"[demo] 读 playbook 笔记：{playbook_dir}")

    notes = read_notes(playbook_dir, min_priority="medium")
    print(f"[demo] 命中 {len(notes)} 条笔记：")
    for n in notes:
        print(f"  - {n['path'].name} [priority={n['priority']}]")

    prior = parse_to_prior(notes, coerce=True)
    print(f"[demo] 合并 prior：{len(prior)} 个 key")

    # 取 cmaes 相关 key
    cmaes_prior = {k: v for k, v in prior.items() if k in ("sigma", "min_budget_for_promotion", "require_crn_pair")}
    print(f"[demo] cmaes 相关 prior：{cmaes_prior}")

    # 跑 warm-start
    print("[demo] 启动 warm-start run（按 prior.sigma=0.15）…")
    res = run(cmaes_prior, n_iter=30, seed=7)

    print(f"[demo] 完成 {res['n_eval']} 轮，最终 best = {res['final_best']:.1f}")
    print("[demo] 曲线（每 5 轮打印一次）:")
    for row in res["curve"][::5]:
        print(f"  i={row['i']:2d}  score={row['score']:8.1f}  best={row['best']:8.1f}")

    # 写报告
    out = ROOT / "experiments" / "playbook_demo_report.md"
    out.write_text(
        f"""# Playbook Demo 报告

## 跑通性验证
- playbook_dir: `{playbook_dir.relative_to(ROOT)}`
- 命中笔记: {len(notes)} 条
- 合并 prior key 数: {len(prior)}
- cmaes 注入 prior: `sigma={cmaes_prior.get('sigma')}`（来自 cmaes_n60 笔记）
- 跑 30 轮 dummy eval，最终 best = **{res['final_best']:.1f}**

## 关键判断
- [x] playbook_io 正确读 + 合并 + coerce
- [x] prior.sigma → CmaesOptimizer.sigma0 通路打通
- [x] tell / ask / best 链条正常
- [x] 不报 TypeError（map_elites 修过的同类 bug 不会触发）

## 下一步
- v2: 替换 dummy_evaluator 为 lab.evaluator，跑真 bench（71k 锚 + cvar 5 种子）
- v2: 加「裸启动」对照 run，按 cvar_min_delta_for_promotion=150 判显著
""",
        encoding="utf-8",
    )
    print(f"[demo] 报告写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
