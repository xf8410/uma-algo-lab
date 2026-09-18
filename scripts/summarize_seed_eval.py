#!/usr/bin/env python3
"""--seed-genome 回验汇总：从 ga_logs/run_*/ga_detail.csv 提取注入个体。

判定：注入个体 31 列参数行（指纹）跨轮恒定；随机个体参数每轮几乎必变。
输出：每个注入个体 Full 级均值±std、同轮配对差（seed0-seed1）、胜率。
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import mean, stdev

root = sys.argv[1] if len(sys.argv) > 1 else "ga_logs"
runs = sorted(glob.glob(os.path.join(root, "run_*")))
per = defaultdict(list)  # (run, fingerprint) -> [full fitness]
for r in runs:
    path = os.path.join(r, "ga_detail.csv")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 34:
                continue
            try:
                fit = float(row[2])
            except ValueError:
                continue  # 表头或异常行
            if row[0].lower().startswith("full"):
                fp = "|".join(row[3:34])
                per[(os.path.basename(r), fp)].append(fit)

if not per:
    print("no data")
    sys.exit(1)

fp_rounds = defaultdict(set)
fp_fit = defaultdict(list)
for (run, fp), fits in per.items():
    fp_rounds[fp].add(run)
    fp_fit[fp].extend(fits)

all_rounds = {os.path.basename(r) for r in runs}
threshold = max(2, int(0.8 * len(all_rounds)))
injected = [fp for fp, rs in fp_rounds.items() if len(rs) >= threshold]
injected.sort(key=lambda fp: -mean(fp_fit[fp]))

print(f"总轮数={len(all_rounds)} 判定阈值=出现>={threshold}轮 注入个体数={len(injected)}")
rows = []
for i, fp in enumerate(injected):
    m = mean(fp_fit[fp])
    s = stdev(fp_fit[fp]) if len(fp_fit[fp]) > 1 else 0.0
    rows.append((i, len(fp_rounds[fp]), m, s))
    print(f"[seed#{i}] 出现{len(fp_rounds[fp])}轮 Full均值={m:.1f} ±{s:.1f} 局数={len(fp_fit[fp])}")

diffs = []
wins = 0
if len(injected) >= 2:
    a, b = injected[0], injected[1]
    for run in sorted(all_rounds):
        fa = per.get((run, a))
        fb = per.get((run, b))
        if fa and fb:
            d = mean(fa) - mean(fb)
            diffs.append(d)
            if d > 0:
                wins += 1
    if diffs:
        sd = stdev(diffs) if len(diffs) > 1 else 0.0
        print(f"配对差(seed0-seed1): n={len(diffs)} 均值={mean(diffs):+.1f} ±{sd:.1f} 胜率={wins}/{len(diffs)}")

out = os.path.join(root, "seed_eval_summary.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed_idx", "rounds", "full_mean", "full_std"])
    for r in rows:
        w.writerow(r)
    if diffs:
        w.writerow(["pair0_minus_1", len(diffs), f"{mean(diffs):+.2f}", f"{wins}/{len(diffs)}"])
print("summary:", out)
