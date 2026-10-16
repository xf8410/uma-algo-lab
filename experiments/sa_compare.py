# -*- coding: utf-8 -*-
"""SA 退火算法 + CMA-ES/TPE/CEM 5 seed × 30 iter 真 bench 排表。

目标：看 SA 在真评估器上是不是有竞争力。
  * SA:    t0=0.05 → t1=0.004 几何降温，1/4 维扰动
  * CMA-ES: sigma0=0.06（默认）+ 协方差自适应
  * TPE:   optuna default（独立采样）
  * CEM:   截断 30% 精英，重采样高斯

协议（CRN 配对）：
  * 同 seed 配对（warm 和 cold 用同 seed 跑全部算法）
  * 同 budget（30 iter per algo）
  * 同 level（smoke）
  * 同 center（[0.5]*45，r8 冠军归一化向量）

输出：
  * per-algo × per-seed best
  * mean / std of best
  * 配对 Δ vs CMA-ES（基准）+ 配对 t 检验
  * 排名

v1 单 seed 看方向，v3 配对看显著。
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
from optimizers import REGISTRY  # noqa: E402

# 默认 4 算法（algo-lab 第一档）
DEFAULT_ALGOS = "sa,cmaes,tpe,cem"


def run_one(name, cls, ev, args, seed):
    """跑一个算法一组，返回 best / curve / 用时。"""
    dim = free_dim()
    opt = cls(dim=dim, seed=seed)
    curve = []
    t0 = time.time()
    for i in range(args.n_iter):
        x = opt.ask()
        genome = decode(x)
        out = ev.evaluate(genome, level="smoke")
        opt.tell(x, out["score"])
        curve.append({
            "i": i,
            "score": out["score"],
            "best": opt.best,
            "hash": out.get("hash", "")[:8],
        })
    return {
        "name": name,
        "seed": seed,
        "best": opt.best,
        "curve": curve,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def paired_t(deltas):
    """5 seed 配对 t 检验；返回 (mean, std, t, p)。"""
    arr = np.array(deltas, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std > 0:
        t = mean / (std / np.sqrt(n))
        try:
            from scipy.stats import t as t_dist
            p = float(2 * (1 - t_dist.cdf(abs(t), df=n - 1)))
        except ImportError:
            p = None
    else:
        t = float("inf") if mean > 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
    return mean, std, t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default=DEFAULT_ALGOS,
                    help=f"逗号分隔，默认 {DEFAULT_ALGOS}")
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--seeds", default="7,13,23,31,47",
                    help="逗号分隔 CRN 种子")
    ap.add_argument("--uma", type=int, default=111501)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--friend", type=int, default=303054)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"[sa-cmp] algos={algos}, seeds={seed_list}, n_iter={args.n_iter}")

    # 校验所有 algo 在 REGISTRY
    for a in algos:
        if a not in REGISTRY:
            print(f"[sa-cmp] 错误：{a} 不在 REGISTRY ({list(REGISTRY.keys())})")
            sys.exit(2)

    # 起 evaluator
    if args.mock:
        class DummyEv:
            n_evals = 0
            def evaluate(self, genome, level="smoke"):
                xs = [v for v in genome.values() if isinstance(v, (int, float))]
                if not xs:
                    return {"score": 50000.0, "hash": "mock"}
                dist = sum((x - 0.5) ** 2 for x in xs if 0 <= x <= 1)
                return {"score": round(50000.0 - 1000.0 * dist, 1), "hash": f"m{self.n_evals}"}
        ev = DummyEv()
        print("[sa-cmp] evaluator: MOCK")
    else:
        ev = Evaluator(uma=args.uma, deck=args.deck, friend=args.friend)
        print(f"[sa-cmp] evaluator: uma={args.uma}")

    # 跑全网格
    all_results = {a: [] for a in algos}
    total_t0 = time.time()
    for seed in seed_list:
        print(f"\n[sa-cmp] ── seed={seed} ──")
        for a in algos:
            print(f"[sa-cmp] 跑 {a} × {args.n_iter} iter…")
            r = run_one(a, REGISTRY[a], ev, args, seed)
            all_results[a].append(r)
            print(f"[sa-cmp]  {a} best = {r['best']:.1f}, {r['elapsed_sec']}s")
    total_elapsed = round(time.time() - total_t0, 1)
    print(f"\n[sa-cmp] 总用时 {total_elapsed}s")

    # 汇总：per-algo mean / std of best
    summary = {}
    for a in algos:
        bests = [r["best"] for r in all_results[a]]
        summary[a] = {
            "mean": round(float(np.mean(bests)), 1),
            "std": round(float(np.std(bests, ddof=1)), 1),
            "min": round(float(np.min(bests)), 1),
            "max": round(float(np.max(bests)), 1),
            "bests_per_seed": bests,
        }

    # 排名（按 mean best 降序）
    ranking = sorted(summary.items(), key=lambda kv: -kv[1]["mean"])

    # 配对 vs CMA-ES（如果 cmaes 在 algos）
    paired = {}
    if "cmaes" in summary:
        cmaes_bests = summary["cmaes"]["bests_per_seed"]
        for a in algos:
            if a == "cmaes":
                continue
            deltas = [summary[a]["bests_per_seed"][i] - cmaes_bests[i] for i in range(len(seed_list))]
            mean_d, std_d, t_d, p_d = paired_t(deltas)
            paired[a] = {
                "mean_delta": round(mean_d, 1),
                "std_delta": round(std_d, 1),
                "t": round(t_d, 3) if t_d is not None else None,
                "p": round(p_d, 4) if p_d is not None else None,
            }

    # 输出
    out_dir = ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    result = {
        "algos": algos,
        "seeds": seed_list,
        "n_iter": args.n_iter,
        "uma": args.uma,
        "summary": summary,
        "ranking": [{"algo": a, **summary[a]} for a, _ in ranking],
        "paired_vs_cmaes": paired,
        "total_elapsed_sec": total_elapsed,
    }
    (out_dir / "sa_compare_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[sa-cmp] 排名（按 mean best）:")
    for i, (a, _) in enumerate(ranking, 1):
        s = summary[a]
        print(f"  {i}. {a:8s}  mean={s['mean']:9.1f}  std={s['std']:6.1f}  range=[{s['min']:.1f}, {s['max']:.1f}]")
    if paired:
        print(f"\n[sa-cmp] 配对 Δ vs CMA-ES（p<0.05 显著）:")
        for a, p in paired.items():
            sig = "✓ sig" if (p["p"] is not None and p["p"] < 0.05) else "  n.s."
            print(f"  {a:8s}  Δ={p['mean_delta']:+7.1f}  std={p['std_delta']:5.1f}  t={p['t']}  p={p['p']}  {sig}")

    # 写大白话 markdown 报告
    report = out_dir / "sa_compare_report.md"
    ranking_table = "\n".join(
        f"| {i} | **{a}** | {s['mean']:.1f} | {s['std']:.1f} | [{s['min']:.1f}, {s['max']:.1f}] |"
        for i, (a, s) in enumerate(ranking, 1)
    )
    paired_table = ""
    if paired:
        paired_table = "\n## 配对 Δ vs CMA-ES（5 seed 配对 t 检验）\n\n| algo | mean Δ | std Δ | t | p | 显著? |\n| --- | --- | --- | --- | --- | --- |\n"
        for a, p in paired.items():
            sig = "✓ p<0.05" if (p["p"] is not None and p["p"] < 0.05) else "n.s."
            paired_table += f"| {a} | **{p['mean_delta']:+.1f}** | {p['std_delta']:.1f} | {p['t']} | {p['p']} | {sig} |\n"

    report.write_text(f"""# SA 退火算法 4 算法排表（5 seed × 30 iter 真 bench）

## 跑法
- algos: {algos}
- seeds: {seed_list}（共 {len(seed_list)} seed）
- n_iter: {args.n_iter} per (algo × seed)
- evaluator: smoke 档, uma={args.uma}
- 总用时: {total_elapsed}s
- 总 evals: {len(algos) * len(seed_list) * args.n_iter}

## 排名（按 mean best 降序）

| 排名 | algo | mean best | std | 范围 |
| --- | --- | --- | --- | --- |
{ranking_table}
{paired_table}
## 大白话解读

{"SA 赢了吗？" if "sa" in [a for a, _ in ranking[:1]] else "SA 没排第一" + "——" + ranking[0][0] + " 第一（mean=" + f"{ranking[0][1]['mean']:.1f}" + "）"}

- 配对 Δ 跟 CMA-ES 比：数字大 = 该算法跑得比 CMA-ES 高
- p<0.05 显著 = 不是瞎蒙，方向稳
- std 小 = 多个 seed 跑出来差不多，鲁棒

## 下一步
- 配对 t 检验显著（p<0.05）且 mean Δ>+100：进 algo-selection.md 第二档
- SA 排第一 + 显著：写 sa_opt 实战笔记，prior 默认 t1=0.004
- SA 没显著：笔记保持 priority=low，SA 作为多样性补充保留
""", encoding="utf-8")
    print(f"\n[sa-cmp] 报告写入 {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
