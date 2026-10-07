# -*- coding: utf-8 -*-
"""CVaR 适应度赛马：把 fitness 从均值切换为下尾均值，验证风险敏感目标能否挤出分。

动机：GA/CMA-ES 默认优化均值（mean），但用户痛点是 holdout 波动大+摆烂诱因。
CVaR_alpha（Conditional Value at Risk）= 最差 alpha 比例种子的 per_seed 均值——
只看均值会选方差大的策略；CVaR 强迫"倒霉时候也得及格"。

协议（与 mean 模式严格对照，保证 CRN）：
  * 同 warm-start 起点（GA r8 冠军）
  * 同预算（同 optimizer 同 budget 次评估）
  * 仅 fitness 计算切换：mean -> 9 (alpha=0.2)
  * 换种子验收：用同一组 VAL_SEEDS=(9001,9002,9003) 复评 best 基因组

输出：
  results/cvar_<ts>_<optimize>_<mode>.json
  summary.txt：每 optimizer 的 mean 模式 vs CVaR 模式对比

跑法（CI）：
  python experiments/cvar_compare.py --optimizers cmaes,cem --budget 20 --uma 111501
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.genome import decode, genome_to_toml, toml_to_vector
from lab.space import free_dim
from lab.evaluator import (
    Evaluator, FAIL_PENALTY, SCREEN_BUILDS, SMOKE_RUNS,
    CVAR_SEEDS, CVAR_DEFAULT_ALPHA,
)
from optimizers import REGISTRY

# 复用 compare.py 的换种子验收口径，CVaR 与 mean 模式完全对齐：
VAL_SEEDS = (9001, 9002, 9003)
VAL_THRESHOLD = 300.0


def _evaluate_for_mode(ev, genome, mode):
    """按 mode 分发评估调用：mean=smoke 1 seed, cvar=5 seeds 下尾均值。

    返回 dict（带 score / per_seed / fail_rate），异常时回退到大负分保流程不断。
    """
    if mode == "mean":
        return ev.evaluate(genome, level="smoke")
    if mode == "cvar":
        return ev.evaluate_cvar(genome, alpha=CVAR_DEFAULT_ALPHA)
    raise ValueError("未知 mode: " + str(mode))


def _val_eval_both_modes(ev, genome):
    """best 基因组用 3 个独立种子（VAL_SEEDS）复评一遍，把两种 fitness 都报出来。

    目的：cvar-only 选出的 best 在 mean 维度可能"看起来没涨"，但 holdout/val 上
    mean 分不掉才是真赢。CVaR 模式的 val 也要报 mean（双口径对表）。
    """
    if genome is None:
        return {"val_score": None, "val_cvar": None, "val_fail_rate": None}
    toml_text = genome_to_toml(genome)
    val_means, val_frs, val_per_seed = [], [], []
    for s in VAL_SEEDS:
        m, fr = ev._run_bench(toml_text, s, SMOKE_RUNS, builds=SCREEN_BUILDS)
        val_means.append(m)
        val_frs.append(fr)
        val_per_seed.append(round(m - FAIL_PENALTY * fr, 1))
    mean_score = sum(val_means) / len(val_means) - FAIL_PENALTY * (sum(val_frs) / len(val_frs))
    # CVaR val：用 val_means 的下尾均值（与训练 CVaR 同 alpha）
    sorted_pairs = sorted(zip(val_per_seed, val_frs), key=lambda t: t[0])
    n_tail = max(1, int(round(len(val_per_seed) * CVAR_DEFAULT_ALPHA)))
    cvar_val = sum(s for s, _ in sorted_pairs[:n_tail]) / n_tail
    return {
        "val_score": round(mean_score, 1),
        "val_cvar": round(cvar_val, 1),
        "val_fail_rate": round(sum(val_frs) / len(val_frs), 4),
        "val_per_seed": val_per_seed,
    }


def run_one(name, cls, args, mode, center_vec=None):
    opt = cls(free_dim(), seed=args.seed, center=center_vec)
    ev = Evaluator(args.uma, deck=args.deck, friend=args.friend,
                   cache_path=str(Path(args.out) / f"cache_{name}_{mode}.json"))
    budget = args.budget
    n = 0
    best_genome, best_score = None, -float("inf")
    fail_count = 0
    while n < budget:
        try:
            x = opt.ask()
        except RuntimeError as e:
            print(f"[{name}/{mode}] ask 停止: {e}")
            break
        genome = decode(x)
        try:
            res = _evaluate_for_mode(ev, genome, mode)
        except Exception as e:
            fail_count += 1
            worst = min((h["score"] for h in opt.history), default=40000.0)
            res = {"score": worst - 5000}
            print(f"[{name}/{mode}] 评估失败（fail {fail_count}/{n+1}）: {str(e)[:80]}")
        opt.tell(x, res["score"])
        n += 1
        if res["score"] > best_score:
            best_score = res["score"]
            best_genome = genome
        if n % 5 == 0 or n == budget:
            print(f"[{name}/{mode}] eval {n}/{budget} best={best_score:.1f}", flush=True)

    val = _val_eval_both_modes(ev, best_genome)
    out = {
        "optimizer": name, "mode": mode, "uma": args.uma,
        "budget": budget, "evals_done": n, "eval_fail_count": fail_count,
        "best_score": round(best_score, 1),
        "best_genome": best_genome,
        "best_genome_toml": genome_to_toml(best_genome) if best_genome else None,
        "history": getattr(opt, "history", []),
        "val": val,
        "alpha": CVAR_DEFAULT_ALPHA, "cvar_seeds": list(CVAR_SEEDS),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="cmaes,cem",
                    help="对比优化器（默认 cmaes,cem；可加 tpe,sa）")
    ap.add_argument("--budget", type=int, default=20,
                    help="每优化器评估次数上限（CVaR 模式 5x cost）")
    ap.add_argument("--uma", type=int, default=111501)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--friend", type=int, default=303054)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--center-toml", default="seeds/center_r8.toml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--modes", default="mean,cvar",
                    help="对比 fitness 模式：mean / cvar / 逗号分隔")
    args = ap.parse_args()

    center_vec = None
    if args.center_toml and Path(args.center_toml).exists():
        center_vec = toml_to_vector(Path(args.center_toml).read_text(encoding="utf-8"))
        print(f"warm-start: {args.center_toml} -> {len(center_vec)} 维向量")
    else:
        print(f"[warn] center_toml {args.center_toml} 不存在，空间中心 0.5 启动")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%m%d_%H%M")
    summary = {}
    for name in args.optimizers.split(","):
        name = name.strip()
        if name not in REGISTRY:
            print(f"未知优化器 {name}，跳过（可用: {list(REGISTRY)}）")
            continue
        for mode in args.modes.split(","):
            mode = mode.strip()
            if mode not in ("mean", "cvar"):
                print(f"未知 mode {mode}，跳过")
                continue
            print(f"\n===== [{name}] mode={mode} =====")
            res = run_one(name, REGISTRY[name], args, mode, center_vec)
            path = outdir / f"cvar_{name}_{mode}_{ts}.json"
            path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{name}/{mode}] 完成 -> {path}  best={res['best_score']}")
            key = f"{name}/{mode}"
            v = res["val"]
            summary[key] = {
                "best": res["best_score"],
                "val_mean": v["val_score"] if v["val_score"] is not None else None,
                "val_cvar": v["val_cvar"] if v.get("val_cvar") is not None else None,
                "val_fail": v["val_fail_rate"],
            }

    print("\n==== CVaR vs Mean 赛马小结（val 用 3 独立种子复评）====")
    print(f"{'optimizer/mode':24s} {'best(train)':>12s} {'val_mean':>12s} {'val_cvar':>12s} {'val_fail':>8s}")
    for k, v in sorted(summary.items()):
        v_mean = v["val_mean"] if v["val_mean"] is not None else float("nan")
        v_cvar = v["val_cvar"] if v["val_cvar"] is not None else float("nan")
        print(f"  {k:22s} {v['best']:>12.1f} {v_mean:>12.1f} {v_cvar:>12.1f} "
              f"{v['val_fail']:>8.4f}")


if __name__ == "__main__":
    main()
