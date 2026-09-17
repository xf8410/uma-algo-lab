# -*- coding: utf-8 -*-
"""赛马实验入口：同预算、同评估协议下对比多个优化器。

用法（CI）：
  python experiments/compare.py --optimizers cmaes,tpe,sa,cem --budget 120 \
      --uma 111501 --runs-rounds 3 --out results/
预算语义：每个优化器最多 budget 次 bench 调用（smoke 档每档 3 次 bench）。
结果写 results/<name>_<ts>.json：{best_score, best_genome, history, evals}。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.genome import decode, genome_to_toml, toml_to_vector
from lab.space import free_dim
from lab.evaluator import FAIL_PENALTY, SCREEN_BUILDS, SMOKE_RUNS
from optimizers import REGISTRY

# 换种子验收（良性判据，用户规则 2026-09-17）：best 基因组用与训练（seed42）无关的
# 独立种子复评，分数接近才说明改进是泛化的；大幅掉分 = 吃种子噪声/过拟合，剔除。
VAL_SEEDS = (9001, 9002, 9003)
VAL_THRESHOLD = 300.0  # |Δ| 超过视为种子敏感（参考阈值，随实测校准）


def run_one(name: str, cls, args, center_vec=None) -> dict:
    from lab.evaluator import Evaluator
    opt = cls(free_dim(), seed=args.seed, center=center_vec)
    ev = Evaluator(args.uma, deck=args.deck, friend=args.friend,
                   cache_path=str(Path(args.out) / f"cache_{name}.json"))
    budget = args.budget
    n = 0
    best_genome = None
    best_score = -float("inf")
    while n < budget:
        try:
            x = opt.ask()
        except RuntimeError as e:
            print(f"[{name}] ask 停止: {e}")
            break
        genome = decode(x)
        try:
            res = ev.evaluate(genome, level=args.level)
        except Exception as e:
            print(f"[{name}] 评估失败（罚分跳过）: {e}")
            res = {"score": best_score - 5000 if best_score > -float("inf") else 50000.0}
        opt.tell(x, res["score"])
        n += 1
        if res["score"] > best_score:
            best_score = res["score"]
            best_genome = genome
        if n % 5 == 0 or n == budget:
            print(f"[{name}] eval {n}/{budget} best={best_score:.1f} "
                  f"(cache hit 率见 cache json)", flush=True)

    # 换种子验收：best 基因组用 3 个独立种子复评（不参与搜索、不占预算）。
    # val_score 接近 best_score → 改进泛化（良性）；大幅掉分 → 种子噪声/过拟合。
    val_score, val_fail = None, None
    if best_genome is not None:
        val_means, val_frs = [], []
        toml_text = genome_to_toml(best_genome)
        for s in VAL_SEEDS:
            m, fr = ev._run_bench(toml_text, s, SMOKE_RUNS, builds=SCREEN_BUILDS)
            val_means.append(m)
            val_frs.append(fr)
        val_score = round(sum(val_means) / len(val_means)
                          - FAIL_PENALTY * (sum(val_frs) / len(val_frs)), 1)
        val_fail = round(sum(val_frs) / len(val_frs), 4)
        delta = round(val_score - best_score, 1)
        verdict = "良性" if abs(delta) <= VAL_THRESHOLD else "种子敏感"
        print(f"[{name}] 换种子验收: 训练={best_score:.1f} → 验证={val_score} "
              f"(Δ={delta:+.1f} {verdict}, seeds={VAL_SEEDS})", flush=True)

    out = {
        "optimizer": name, "uma": args.uma, "level": args.level,
        "budget": budget, "evals_done": n,
        "best_score": round(best_score, 1),
        "best_genome": best_genome,
        "best_genome_toml": genome_to_toml(best_genome) if best_genome else None,
        "history": getattr(opt, "history", []),
        "val_seeds": list(VAL_SEEDS), "val_score": val_score, "val_fail_rate": val_fail,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="cmaes,tpe,sa,cem")
    ap.add_argument("--budget", type=int, default=60, help="每优化器最多评估次数")
    ap.add_argument("--uma", type=int, required=True)
    ap.add_argument("--deck", default=None, help="精确卡组 idrank 串")
    ap.add_argument("--friend", type=int, default=303054)
    ap.add_argument("--level", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--center-toml", default=None,
                    help="warm-start 基因组 TOML（如 seeds/center_r5.toml）；缺省=空间中心 0.5")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    center_vec = None
    if args.center_toml:
        center_vec = toml_to_vector(Path(args.center_toml).read_text(encoding="utf-8"))
        print(f"warm-start: {args.center_toml} → {len(center_vec)} 维向量")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%m%d_%H%M")
    summary = {}
    for name in args.optimizers.split(","):
        name = name.strip()
        if name not in REGISTRY:
            print(f"未知优化器 {name}，跳过（可用: {list(REGISTRY)}）")
            continue
        res = run_one(name, REGISTRY[name], args, center_vec)
        path = outdir / f"{name}_{ts}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        summary[name] = {"best": res["best_score"], "val": res["val_score"]}
        print(f"[{name}] 完成 → {path}  best={res['best_score']}")
    print("\n==== 赛马小结（同预算 + 换种子验收）====")
    for k, v in sorted(summary.items(), key=lambda kv: -(kv[1]["val"] or kv[1]["best"])):
        val = v["val"]
        if val is not None:
            verdict = "良性" if abs(val - v["best"]) <= VAL_THRESHOLD else "种子敏感"
            print(f"  {k:8s} 训练={v['best']}  验证={val} (Δ={val - v['best']:+.1f} {verdict})")
        else:
            print(f"  {k:8s} 训练={v['best']}  验证=无候选")


if __name__ == "__main__":
    main()
