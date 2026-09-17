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

from lab.genome import decode, genome_to_toml
from lab.space import free_dim
from optimizers import REGISTRY


def run_one(name: str, cls, args) -> dict:
    from lab.evaluator import Evaluator
    opt = cls(free_dim(), seed=args.seed)
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
    out = {
        "optimizer": name, "uma": args.uma, "level": args.level,
        "budget": budget, "evals_done": n,
        "best_score": round(best_score, 1),
        "best_genome": best_genome,
        "best_genome_toml": genome_to_toml(best_genome) if best_genome else None,
        "history": getattr(opt, "history", []),
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
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%m%d_%H%M")
    summary = {}
    for name in args.optimizers.split(","):
        name = name.strip()
        if name not in REGISTRY:
            print(f"未知优化器 {name}，跳过（可用: {list(REGISTRY)}）")
            continue
        res = run_one(name, REGISTRY[name], args)
        path = outdir / f"{name}_{ts}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        summary[name] = res["best_score"]
        print(f"[{name}] 完成 → {path}  best={res['best_score']}")
    print("\n==== 赛马小结（同预算 best_score）====")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
