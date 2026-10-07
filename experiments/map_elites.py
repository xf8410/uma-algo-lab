# -*- coding: utf-8 -*-
"""MAP-Elites 单跑：跑 N eval 后报 archive grid + 各 cell 冠军 + 总冠军。

与 compare.py 区别：
  * compare.py 只报全 archive 最大 fitness（与 GA 同口径对表）
  * 本脚本额外报 archive 填充率 / 5x5 grid 分布 / 行为区域冠军
    → 反映 MAP-Elites 的核心价值：多样性覆盖

用法（CI）：
  python experiments/map_elites.py --budget 60 --uma 111501
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
)
from optimizers import REGISTRY

# 换种子验收（复用 compare.py 口径）
VAL_SEEDS = (9001, 9002, 9003)
VAL_THRESHOLD = 300.0


def run_one(args, center_vec=None) -> dict:
    name = "map_elites"
    cls = REGISTRY[name]
    opt = cls(free_dim(), seed=args.seed, center=center_vec)
    ev = Evaluator(args.uma, deck=args.deck, friend=args.friend,
                   cache_path=str(Path(args.out) / f"cache_{name}.json"))
    budget = args.budget
    n = 0
    best_genome, best_score = None, -float("inf")
    fail_count = 0
    while n < budget:
        try:
            x = opt.ask()
        except RuntimeError as e:
            print(f"[{name}] ask 停止: {e}")
            break
        genome = decode(x)
        try:
            res = ev.evaluate(genome, level="smoke")
        except Exception as e:
            fail_count += 1
            worst = min((h["score"] for h in opt.history), default=40000.0)
            res = {"score": worst - 5000}
            print(f"[{name}] 评估失败（fail {fail_count}/{n+1}）: {str(e)[:80]}")
        opt.tell(x, res["score"])
        n += 1
        if res["score"] > best_score:
            best_score = res["score"]
            best_genome = genome
        if n % 5 == 0 or n == budget:
            print(f"[{name}] eval {n}/{budget} best={best_score:.1f} "
                  f"archive={opt.coverage:.0%}", flush=True)

    # 换种子验收（与 compare.py 同口径）
    val_score, val_fail = None, None
    if best_genome is not None:
        toml_text = genome_to_toml(best_genome)
        val_means, val_frs = [], []
        for s in VAL_SEEDS:
            m, fr = ev._run_bench(toml_text, s, SMOKE_RUNS, builds=SCREEN_BUILDS)
            val_means.append(m)
            val_frs.append(fr)
        val_score = round(sum(val_means) / len(val_means)
                         - FAIL_PENALTY * (sum(val_frs) / len(val_frs)), 1)
        val_fail = round(sum(val_frs) / len(val_frs), 4)
        delta = round(val_score - best_score, 1)
        verdict = "良性" if abs(delta) <= VAL_THRESHOLD else "种子敏感"
        print(f"[{name}] 换种子验收: 训练={best_score:.1f} -> 验证={val_score} "
              f"(Δ={delta:+.1f} {verdict})", flush=True)

    archive = opt.archive_dump()
    # 序列化 grid：5x5 列表 (None 表示空 cell)
    grid_pretty = []
    for r, row in enumerate(archive["grid"]):
        grid_pretty.append([f"{v:.1f}" if v is not None else "—" for v in row])
    out = {
        "optimizer": name, "uma": args.uma, "level": "smoke",
        "budget": budget, "evals_done": n, "eval_fail_count": fail_count,
        "best_score": round(best_score, 1),
        "best_genome": best_genome,
        "best_genome_toml": genome_to_toml(best_genome) if best_genome else None,
        "history": getattr(opt, "history", []),
        "val_seeds": list(VAL_SEEDS), "val_score": val_score, "val_fail_rate": val_fail,
        "archive": archive, "grid_pretty": grid_pretty,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60,
                    help="MAP-Elites 评估次数（与 CMA-ES 同口径可比）")
    ap.add_argument("--uma", type=int, default=111501)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--friend", type=int, default=303054)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--center-toml", default="seeds/center_r8.toml")
    ap.add_argument("--out", default="results")
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
    res = run_one(args, center_vec)
    path = outdir / f"map_elites_{ts}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[{res['optimizer']}] 完成 -> {path}  best={res['best_score']}")
    print(f"\n==== archive 摘要 ====")
    a = res["archive"]
    print(f"filled_cells = {a['filled_cells']}/{a['n_bins']**2}  coverage = {a['coverage']:.0%}")
    print(f"best_score   = {a['best_score']:.1f}")
    print(f"ask/tell     = {a['n_asked']}  explore={a['explore_count']}  mutate={a['mutate_count']}")
    print(f"\n行为格子（行=vital_rest 0=少休息/4=多休息；列=pt_tradeoff 0=低/4=高）")
    print(f"  bh2\\bh1  {'  '.join(f'  bh1={i}  ' for i in range(a['n_bins']))}")
    for c in range(a["n_bins"]):
        row = res["grid_pretty"][c]
        cells = '  '.join(f'{v:>8s}' if v == "—" else f'{float(v):>8.1f}' for v in row)
        print(f"  bh2={c}     {cells}")


if __name__ == "__main__":
    main()
