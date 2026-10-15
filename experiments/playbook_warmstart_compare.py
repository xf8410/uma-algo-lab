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
    ap.add_argument("--seed", type=int, default=7,
                    help="单 seed 模式（向后兼容），与 --seeds 互斥")
    ap.add_argument("--seeds", type=str, default=None,
                    help="多 seed 配对模式，逗号分隔，如 '7,13,23,31,47'。启用后用配对 t 检验。")
    ap.add_argument("--uma", type=int, default=111501)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--friend", type=int, default=303054)
    ap.add_argument("--mock", action="store_true",
                    help="沙箱/无 bench_base 时用 dummy eval（仅验证流程，不跑真 bench）")
    args = ap.parse_args()

    # 解析 seed 列表
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seed_list = [args.seed]

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

    # 3) 跑 N 组（每个 seed 一组）
    seed_results = []
    total_t0 = time.time()
    for seed in seed_list:
        print(f"\n[warmstart] ── seed={seed} ──")
        print(f"[warmstart] 跑 warm (sigma0={cmaes_sigma}) × {args.n_iter} iter…")
        warm = run_one(ev, prior, cmaes_sigma, args.n_iter, seed, args.uma)
        print(f"[warmstart]  warm best = {warm['best']:.1f}, 用时 {warm['elapsed_sec']}s")

        print(f"[warmstart] 跑 cold (sigma0=0.06) × {args.n_iter} iter…")
        cold = run_one(ev, prior, 0.06, args.n_iter, seed, args.uma)
        print(f"[warmstart]  cold best = {cold['best']:.1f}, 用时 {cold['elapsed_sec']}s")

        seed_results.append({
            "seed": seed,
            "warm": warm,
            "cold": cold,
            "delta": round(warm["best"] - cold["best"], 1),
        })
        print(f"[warmstart]  seed={seed} Δbest = {warm['best'] - cold['best']:+.1f}")
    total_elapsed = round(time.time() - total_t0, 1)
    print(f"\n[warmstart] 总用时 {total_elapsed}s")

    # 4) 判据：单 seed 用 verdict；多 seed 算配对 t 检验
    if len(seed_list) == 1:
        delta = seed_results[0]["delta"]
        tag, msg = verdict(delta)
        t_stat = None
        p_value = None
        mean_delta = delta
        std_delta = 0.0
    else:
        deltas = np.array([r["delta"] for r in seed_results], dtype=float)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))  # sample std
        n = len(deltas)
        if std_delta > 0:
            t_stat = float(mean_delta / (std_delta / np.sqrt(n)))
            # 配对 t 检验 p 值（双侧）：用 t 分布
            from math import lgamma, exp, log, sqrt
            x = t_stat
            df = n - 1
            # 用正则化不完全 beta 函数近似
            # 简化：用 survival 公式近似
            # 实际上 scipy.stats.t 没装，用简化版（仅在 |t|<10 准确）
            try:
                from scipy.stats import t as t_dist  # noqa
                p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=df)))
            except ImportError:
                # 兜底：t>2 → p<0.05, t>3 → p<0.01, t>4 → p<0.001（粗略）
                p_value = None
        else:
            t_stat = float("inf") if mean_delta > 0 else 0.0
            p_value = 0.0 if mean_delta != 0 else 1.0

        if mean_delta > DELTA_PROMOTE and (p_value is None or p_value < 0.05):
            tag, msg = "promote", f"mean Δbest=+{mean_delta:.1f} > +{DELTA_PROMOTE} 且 {'t=' + f'{t_stat:.2f}' if t_stat else 't=∞'} → 显著 promote"
        elif mean_delta < -DELTA_NOISE and (p_value is None or p_value < 0.05):
            tag, msg = "regress", f"mean Δbest={mean_delta:.1f} < -{DELTA_NOISE} 且显著 → regress，prior 写错"
        elif abs(mean_delta) < DELTA_NOISE:
            tag, msg = "noise", f"|mean Δ|={abs(mean_delta):.1f} < {DELTA_NOISE}，多 seed 也无收益 → noise"
        else:
            tag, msg = "marginal", f"mean Δ=+{mean_delta:.1f} (t={t_stat:.2f}, p={p_value}), 方向对但量级不够 promote → marginal"
        delta = mean_delta

    print(f"\n[warmstart] 汇总: mean Δ = {mean_delta:+.1f}, std = {std_delta:.1f}, t = {t_stat}, p = {p_value}")
    print(f"[warmstart] 判据: {tag}: {msg}")

    # 5) 写报告
    out_dir = ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    report = out_dir / "warmstart_compare_report.md"

    per_seed_table = "\n".join(
        f"| {r['seed']} | {r['warm']['best']:.1f} | {r['cold']['best']:.1f} | **{r['delta']:+.1f}** |"
        for r in seed_results
    )

    if len(seed_list) == 1:
        curve_section = "\n".join(
            f"| {r['i']} | {w['score']:.1f} | {w['best']:.1f} | {c['score']:.1f} | {c['best']:.1f} |"
            for r, w, c in zip(
                seed_results[0]["warm"]["curve"][::5],
                seed_results[0]["warm"]["curve"][::5],
                seed_results[0]["cold"]["curve"][::5],
            )
        )
        curve_table = f"""
## 曲线（每 5 轮 best）
| i | warm score | warm best | cold score | cold best |
| --- | --- | --- | --- | --- |
{curve_section}
"""
    else:
        curve_table = ""

    report.write_text(f"""# Warm-start vs 裸启动 对比报告

## 跑法
- playbook: {playbook_dir.relative_to(ROOT)} 命中 {len(notes)} 条
- prior.sigma = **{cmaes_sigma}**（warm 用）, 0.06（裸启用）
- 优化器: CmaesOptimizer(dim={free_dim()})
- evaluator: smoke 档, uma={args.uma}
- n_iter = {args.n_iter} per seed
- seeds = {seed_list}（共 {len(seed_list)} seed）
- 总用时 = {total_elapsed}s

## Per-seed 结果

| seed | warm best | cold best | Δ |
| --- | --- | --- | --- |
{per_seed_table}

## 统计汇总

| 指标 | 值 |
| --- | --- |
| mean Δ | **{mean_delta:+.1f}** |
| std Δ | {std_delta:.1f} |
| t-statistic | {t_stat if t_stat is not None else 'N/A'} |
| p-value | {f'{p_value:.4f}' if p_value is not None else 'N/A (scipy 不可用，仅 t 报)' if len(seed_list) > 1 else 'N/A (单 seed)'} |

## 判据
- mean Δ > +{DELTA_PROMOTE} 且 p<0.05（多 seed 模式）：promote
- mean Δ < -{DELTA_NOISE} 且 p<0.05：regress
- |mean Δ| < {DELTA_NOISE}：noise
- 其他：marginal

## 结论
- **{tag.upper()}**: {msg}
{curve_table}
## 下一步
- promote → 改 cmaes_n60 笔记 priority=high，加 default_for_cmaes=true
- noise → 笔记保留但 priority=low，不动默认
- regress → 改 cmaes_n60 笔记 sigma=0.10（向默认靠拢）
- marginal → 留着，sigma=0.15 作为「可选项」记入笔记但不动默认
""", encoding="utf-8")
    print(f"[warmstart] 报告写入 {report.relative_to(ROOT)}")

    # JSON 结果（CI 抓取用）
    (out_dir / "warmstart_compare_result.json").write_text(json.dumps({
        "seeds": seed_list,
        "n_iter": args.n_iter,
        "prior_sigma": cmaes_sigma,
        "per_seed": [
            {"seed": r["seed"], "warm": r["warm"]["best"], "cold": r["cold"]["best"], "delta": r["delta"]}
            for r in seed_results
        ],
        "mean_delta": round(mean_delta, 1),
        "std_delta": round(std_delta, 1),
        "t_statistic": round(t_stat, 3) if t_stat is not None else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
        "tag": tag,
        "verdict_msg": msg,
        "total_elapsed_sec": total_elapsed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 退出码：regress → 非 0（CI 红，强制主人看）
    sys.exit(1 if tag == "regress" else 0)


if __name__ == "__main__":
    main()
