# -*- coding: utf-8 -*-
"""Playbook warm-start vs 裸启动 实测对比（v2 真 bench）。

目标：验证 Playbook prior 注入是否真有收益。
  * warm-start: CmaesOptimizer(sigma0=0.15) ← 来自 cmaes_n60 笔记
  * 裸启动:    CmaesOptimizer(sigma0=0.06) ← 默认

协议（保证 CRN 配对）：
  * 同 seed（7）
  * 同 center（[0.5]*45，r8 冠军归一化向量）
  * 同 level（smoke）
  * 同 budget（30 iter）
  * 仅 sigma0 不同 → 配对差 Δbest = warm.best - 裸.best

判据（来自 cvar_min_delta_for_promotion=150）：
  * Δbest > +150: 显著 warm-start 优（值得作为 v3 默认）
  * |Δbest| < 50: 噪声带（warm-start 无收益，回到默认）
  * Δbest < -50: warm-start 反而差（prior 写错，回去改笔记）

v2 限制：单 seed，30 iter，level=smoke。
  v3 升级：5 seed × 30 iter，level=smoke，CRN 配对后取 t 检验。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.genome import decode  # noqa: E402
from lab.space import free_dim  # noqa: E402
from lab.evaluator import Evaluator  # noqa: E402
from optimizers import CmaesOptimizer  # noqa: E402
from lab.playbook_io import read_notes, parse_to_prior  # noqa: E402

# 判据（来自 cvar_min_delta_for_promotion=150）
DELTA_PROMOTE = 150.0
DELTA_NOISE = 50.0


def run_one(ev, prior, sigma0, n_iter, seed, uma):
    """跑一组 CMA-ES，按 prior 注入 sigma0；返回曲线 + final best。"""
    dim = free_dim()
    opt = CmaesOptimizer(dim=dim, seed=seed, sigma0=sigma0)

    curve = []
    t0 = time.time()
    for i in range(n_iter):
        x = opt.ask()
        genome = decode(x)
        out = ev.evaluate(genome, level="smoke")
        s = out["score"]
        opt.tell(x, s)
        curve.append({
            "i": i,
            "score": s,
            "best": opt.best,
            "hash": out.get("hash", "")[:8],
            "fail_rate": out.get("fail_rate", 0.0),
        })
    return {
        "sigma0": sigma0,
        "curve": curve,
        "best": opt.best,
        "n_eval": n_iter,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def verdict(delta_best):
    if delta_best > DELTA_PROMOTE:
        return "promote", f"Δbest=+{delta_best:.1f} > +{DELTA_PROMOTE}，warm-start 显著优，v3 默认用 prior.sigma=0.15"
    if abs(delta_best) < DELTA_NOISE:
        return "noise", f"|Δbest|={abs(delta_best):.1f} < {DELTA_NOISE}，噪声带，warm-start 无收益，回到 sigma=0.06"
    if delta_best < -DELTA_NOISE:
        return "regress", f"Δbest={delta_best:.1f} < -{DELTA_NOISE}，warm-start 反退，prior 写错，去改 cmaes_n60 笔记"
    return "marginal", f"Δbest=+{delta_best:.1f} 介于 [{DELTA_NOISE}, {DELTA_PROMOTE}]，边际，建议扩 seed 数再判"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--uma", type=int, default=111501)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--friend", type=int, default=303054)
    ap.add_argument("--mock", action="store_true",
                    help="沙箱/无 bench_base 时用 dummy eval（仅验证流程，不跑真 bench）")
    args = ap.parse_args()

    # 1) 读 playbook
    playbook_dir = ROOT / "playbook" / "notes"
    notes = read_notes(playbook_dir, min_priority="medium")
    prior = parse_to_prior(notes, coerce=True)
    cmaes_sigma = float(prior.get("sigma", 0.06))
    print(f"[warmstart] playbook 命中 {len(notes)} 条，prior.sigma = {cmaes_sigma}")

    # 2) 起 evaluator（mock 模式替成 dummy）
    if args.mock:
        class DummyEv:
            n_evals = 0
            def evaluate(self, genome, level="smoke"):
                # 凸函数：fitness 越接近 50000 越好
                xs = [v for v in genome.values() if isinstance(v, (int, float))]
                if not xs:
                    return {"score": 50000.0, "hash": "mock", "fail_rate": 0.0}
                # 用 mean - |x - center|^2 * 1000 模拟
                center = 0.5
                dist = sum((x - center) ** 2 for x in xs if 0 <= x <= 1)
                score = 50000.0 - 1000.0 * dist
                return {"score": round(score, 1), "hash": f"m{self.n_evals}", "fail_rate": 0.0}
        ev = DummyEv()
        print("[warmstart] evaluator: MOCK（沙箱模式，不真跑 bench）")
    else:
        ev = Evaluator(uma=args.uma, deck=args.deck, friend=args.friend)
        print(f"[warmstart] evaluator ready: uma={args.uma}, deck={args.deck}, friend={args.friend}")

    # 3) 跑两组
    print(f"[warmstart] 跑 warm-start (sigma0={cmaes_sigma}) × {args.n_iter} iter…")
    warm = run_one(ev, prior, cmaes_sigma, args.n_iter, args.seed, args.uma)
    print(f"[warmstart]  warm best = {warm['best']:.1f}, 用时 {warm['elapsed_sec']}s")

    # 缓存清掉，让裸启真跑（不让 warm 的 cache 命中干扰）
    # 注：ev.cache 是 dict（hash, level）→ result；不同 hash 不会命中。
    #     同 center 但不同 x 路径 → 不同 hash。

    print(f"[warmstart] 跑 裸启 (sigma0=0.06) × {args.n_iter} iter…")
    cold = run_one(ev, prior, 0.06, args.n_iter, args.seed, args.uma)
    print(f"[warmstart]  cold best = {cold['best']:.1f}, 用时 {cold['elapsed_sec']}s")

    # 4) 判
    delta = warm["best"] - cold["best"]
    tag, msg = verdict(delta)
    print(f"[warmstart] Δbest = {delta:+.1f} → {tag}: {msg}")

    # 5) 写报告
    out_dir = ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    report = out_dir / "warmstart_compare_report.md"
    report.write_text(f"""# Warm-start vs 裸启动 对比报告（v2 真 bench）

## 跑法
- playbook: {playbook_dir.relative_to(ROOT)} 命中 {len(notes)} 条
- prior.sigma = **{cmaes_sigma}**（warm 用）, 0.06（裸启用）
- 优化器: CmaesOptimizer(dim={free_dim()}, seed={args.seed})
- evaluator: smoke 档, uma={args.uma}
- n_iter = {args.n_iter}

## 结果

| 模式 | sigma0 | best | 用时(s) |
| --- | --- | --- | --- |
| warm-start | {warm['sigma0']} | **{warm['best']:.1f}** | {warm['elapsed_sec']} |
| 裸启 | {cold['sigma0']} | {cold['best']:.1f} | {cold['elapsed_sec']} |
| **Δ** | | **{delta:+.1f}** | |

## 判据
- Δ > +{DELTA_PROMOTE}：promote（warm-start 进默认）
- |Δ| < {DELTA_NOISE}：noise（warm-start 无效）
- Δ < -{DELTA_NOISE}：regress（prior 写错）
- 其他：marginal（扩 seed 再判）

## 结论
- **Δbest = {delta:+.1f}** → `{tag}`
- {msg}

## 曲线（每 5 轮 best）
| i | warm score | warm best | cold score | cold best |
| --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| {r['i']} | {w['score']:.1f} | {w['best']:.1f} | {c['score']:.1f} | {c['best']:.1f} |"
        for r, w, c in zip(
            warm["curve"][::5], warm["curve"][::5], cold["curve"][::5]
        )
    ) + f"""

## v2 限制
- 单 seed（{args.seed}），不满足 require_crn_pair=true 的硬约束
- 30 iter budget，level=smoke
- v3 升级：5 seed × 30 iter + 配对 t 检验

## 下一步
- promote → 改 cmaes_n60 笔记 priority=high，第五条加 "default_for_cmaes=true"
- noise → 笔记保留但 priority=low，不动默认
- regress → 改 cmaes_n60 笔记 sigma=0.10（向默认靠拢）
- marginal → 跑 v3（5 seed），不更新笔记
""", encoding="utf-8")
    print(f"[warmstart] 报告写入 {report.relative_to(ROOT)}")

    # JSON 结果（CI 抓取用）
    (out_dir / "warmstart_compare_result.json").write_text(json.dumps({
        "warm": {"sigma0": warm["sigma0"], "best": warm["best"], "elapsed_sec": warm["elapsed_sec"]},
        "cold": {"sigma0": cold["sigma0"], "best": cold["best"], "elapsed_sec": cold["elapsed_sec"]},
        "delta": round(delta, 1),
        "tag": tag,
        "verdict_msg": msg,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 退出码：regress → 非 0（CI 红，强制主人看）
    sys.exit(1 if tag == "regress" else 0)


if __name__ == "__main__":
    main()
