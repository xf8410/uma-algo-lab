#!/usr/bin/env python3
"""--seed-genome 回验汇总 v2：从 ga_logs/run_*/ga_detail.csv 提取注入个体。

v2 修复（原版 no data 根因）：ga_detail.csv 的 level 列实际取值是 s/f/h
（初筛/精评/holdout），不是 "full"；且 genome_hash 列直接给个体身份，无需指纹。

口径: 精评(f) fitness —— 个体×build 种子池均值, 按 (run,build) 去重。
目标: e52237aeb5b9b2b7=r8champ, e9843ff378fb8f95=tpe_best
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import mean, stdev

R8 = "e52237aeb5b9b2b7"
TPE = "e9843ff378fb8f95"
NAMES = {R8: "r8champ", TPE: "tpe_best"}

root = sys.argv[1] if len(sys.argv) > 1 else "ga_logs"
runs = sorted(glob.glob(os.path.join(root, "run_*")))
if not runs:
    print("no data")
    sys.exit(1)

per = defaultdict(dict)      # run -> name -> {build: fitness}
for r in runs:
    path = os.path.join(r, "ga_detail.csv")
    if not os.path.exists(path):
        continue
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            h = row.get("genome_hash", "")
            if h not in NAMES:
                continue
            if row.get("level") != "f":
                continue
            try:
                fit = float(row["fitness"])
            except (KeyError, ValueError):
                continue
            per[os.path.basename(r)][NAMES[h]][row["build"]] = fit

if not per:
    print("no data")
    sys.exit(1)

print(f"总轮数={len(runs)} 双方在场轮数={sum(1 for v in per.values() if len(v) == 2)}")
rows = []
allfits = {n: [] for n in NAMES.values()}
for name in NAMES.values():
    vals = [fv for v in per.values() for fv in v.get(name, {}).values()]
    if vals:
        allfits[name] = vals
        print(f"[{name}] build观测数={len(vals)} 精评fitness均值={mean(vals):.1f} ±{stdev(vals):.1f}")
        rows.append((name, len(vals), f"{mean(vals):.2f}", f"{stdev(vals):.2f}"))

diffs, wins = [], 0
for run, v in sorted(per.items()):
    if len(v) < 2:
        continue
    fa = list(v.get("r8champ", {}).values())
    fb = list(v.get("tpe_best", {}).values())
    if fa and fb:
        d = mean(fa) - mean(fb)
        diffs.append(d)
        if d > 0:
            wins += 1
if diffs:
    sd = stdev(diffs) if len(diffs) > 1 else 0.0
    print(f"轮级配对差(r8-tpe): n={len(diffs)} 均值={mean(diffs):+.1f} ±{sd:.1f} 胜率={wins}/{len(diffs)}")
    rows.append(("pair_r8_minus_tpe", len(diffs), f"{mean(diffs):+.2f}", f"{wins}/{len(diffs)}"))

out = os.path.join(root, "seed_eval_summary.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "build_obs", "f_fitness_mean", "f_fitness_std"])
    for r in rows:
        w.writerow(r)
print("summary:", out)
