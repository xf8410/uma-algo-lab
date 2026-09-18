#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA 口径回验：r8champ vs tpe_best 逐轮配对分析
输入: eval_runs/ 目录（artifact 解压后, run_*/ga_detail.csv, 9213 轮）
输出: pair_report.md + pair_report.json

配对逻辑（CRN 严格同种子）:
  同一轮 run_N 内, 同 level(s/f/h) + 同 build + 同 seed 的两行 score 相减。
  目标个体:
    e52237aeb5b9b2b7 = R8 冠军 (GA loop 老账 fit max 71415.6 那一位)
    e9843ff378fb8f95 = TPE best  (uma-algo-lab TPE 512发冠军 64315)
CSV 列: level,genome_hash,fitness,build,seed,score,rank,...
"""
import csv, glob, json, math, os, statistics, sys, collections

R8 = "e52237aeb5b9b2b7"
TPE = "e9843ff378fb8f95"
NAMES = {R8: "r8champ", TPE: "tpe_best"}
LEVELS = ("f", "h", "s")

def mean(xs): return sum(xs) / len(xs)
def ci95(d):
    if len(d) < 2: return 0.0
    return 1.96 * statistics.pstdev(d) / math.sqrt(len(d))

def main(root, out_md, out_json):
    # 个体级: hash -> level -> [score...]
    ind = {R8: collections.defaultdict(list), TPE: collections.defaultdict(list)}
    # 轮级 f 均值 (对齐日志"最优(精评)"口径)
    per_run = {}          # run -> {hash: mean_f_score}
    # 严格 CRN 配对: (level, build, seed) -> [d...], d = score_r8 - score_tpe
    pair = collections.defaultdict(list)
    runs_seen = set()
    files = sorted(glob.glob(os.path.join(root, "run_*", "ga_detail.csv")))
    for fp in files:
        run = fp.split(os.sep)[-2]
        rows = {R8: [], TPE: []}
        with open(fp, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                h = row.get("genome_hash", "")
                if h not in NAMES: continue
                try: sc = float(row["score"])
                except Exception: continue
                lv, bd, sd = row["level"], row["build"], row["seed"]
                rows[h].append((lv, bd, sd, sc))
                ind[h][lv].append(sc)
        runs_seen.add(run)
        hit = [h for h in (R8, TPE) if rows[h]]
        if len(hit) == 2:
            by = {h: collections.defaultdict(list) for h in (R8, TPE)}
            for h in (R8, TPE):
                for lv, bd, sd, sc in rows[h]:
                    by[h][(lv, bd, sd)].append(sc)
            for k in set(by[R8]) & set(by[TPE]):
                for a, b in zip(by[R8][k], by[TPE][k]):
                    pair[k].append(a - b)
            mf = {h: (lambda xs: mean(xs) if xs else None)([s for (lv, bd, sd, s) in rows[h] if lv == "f"]) for h in (R8, TPE)}
            if mf[R8] is not None and mf[TPE] is not None:
                per_run[run] = mf
    # ---------- 汇总 ----------
    rep = {"runs_total": len(files), "runs_with_pair": len(per_run),
           "individual": {}, "pair_crn": {}, "per_run_pair": {}}
    for h in (R8, TPE):
        rep["individual"][NAMES[h]] = {
            lv: {"n": len(v), "mean": round(mean(v), 1) if v else None,
                 "std": round(statistics.pstdev(v), 1) if len(v) > 1 else None}
            for lv in LEVELS if ind[h][lv]}
    for lv in LEVELS:
        ds = [d for (l, b, s), v in pair.items() if l == lv for d in v]
        if not ds: continue
        wins = sum(1 for d in ds if d > 0)
        rep["pair_crn"][lv] = {"n": len(ds), "mean_d": round(mean(ds), 1),
                               "std": round(statistics.pstdev(ds), 1) if len(ds) > 1 else 0,
                               "ci95": round(ci95(ds), 1),
                               "r8_win_rate": round(wins / len(ds) * 100, 1)}
    if per_run:
        drun = [m[R8] - m[TPE] for m in per_run.values()]
        rep["per_run_pair"] = {
            "n": len(drun), "mean_d": round(mean(drun), 1),
            "ci95": round(ci95(drun), 1),
            "r8_win_rate": round(sum(1 for d in drun if d > 0) / len(drun) * 100, 1),
            "m_r8_mean": round(mean([m[R8] for m in per_run.values()]), 1),
            "m_tpe_mean": round(mean([m[TPE] for m in per_run.values()]), 1)}
    json.dump(rep, open(out_json, "w"), ensure_ascii=False, indent=1)
    # ---------- 报告 ----------
    L = ["# GA 口径回验: R8 冠军 vs TPE best 逐轮配对", "",
         f"- 数据: {len(files)} 轮 ga_detail.csv (artifact 10543408803, GA 老考场复刻 uma=106301 等)",
         f"- 配对口径: 同轮同 level+build+seed 的 score 直接相减 (CRN 同种子, 零噪声平移)", ""]
    L.append("## 个体绝对分 (本回验考场)")
    L.append("| 个体 | 层 | n | 均值 | std |")
    L.append("|---|---|---|---|---|")
    for h in (R8, TPE):
        for lv, st in rep["individual"][NAMES[h]].items():
            L.append(f"| {NAMES[h]} | {lv} | {st['n']} | {st['mean']} | {st['std']} |")
    L += ["", "## 严格 CRN 配对差 (d = r8 - tpe, 正数=r8 更好)"]
    L.append("| 层 | n | 均值差 | 95%CI | r8 胜率 |")
    L.append("|---|---|---|---|---|")
    for lv, st in rep["pair_crn"].items():
        L.append(f"| {lv} | {st['n']} | {st['mean_d']} | ±{st['ci95']} | {st['r8_win_rate']}% |")
    if rep["per_run_pair"]:
        p = rep["per_run_pair"]
        L += ["", "## 轮级配对 (GA 实战视角, 每轮各算精评均值再相减)",
              f"- 配对轮数 n={p['n']}, r8 胜率 {p['r8_win_rate']}%",
              f"- 均值差 {p['mean_d']} ±{p['ci95']}",
              f"- r8 精评均值 {p['m_r8_mean']}, tpe 精评均值 {p['m_tpe_mean']}", ""]
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "eval_runs"
    main(root, "pair_report.md", "pair_report.json")
