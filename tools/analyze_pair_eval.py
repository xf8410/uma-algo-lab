#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA 口径回验：r8champ vs tpe_best 逐轮配对分析（v2, 双口径 CRN）

CSV 真实列: level,genome_hash,fitness,build,seed,score,rank,...
  - level: s=初筛 / f=精评 / h=holdout
  - fitness: 个体×build 的种子池均值（同 build 多行重复, 需按 (run,build) 去重）
  - score: 单局比赛分（每 seed 一行, 局级 CRN 可配）

口径:
  A. build 级 CRN: 同 (run, build) 双方 fitness 差 —— 种子池相同(42), 对齐 71k 老账口径
  B. 局级 CRN:   同 (run, build, seed) 双方 score 差 —— 最细粒度
  C. 轮级:       同 run 双方精评(f) fitness 均值差 —— GA 实战视角
  D. 71k 折算:   71415.6 - (mean_r8_fit - mean_tpe_fit)
"""
import csv, glob, json, math, os, statistics, sys, collections

R8 = "e52237aeb5b9b2b7"   # R8 冠军 (老账 fit max 71415.6)
TPE = "e9843ff378fb8f95"  # TPE best (lab 512发 64315)
NAMES = {R8: "r8champ", TPE: "tpe_best"}
LEVELS = ("f", "h", "s")

def mean(xs): return sum(xs) / len(xs)
def ci95(d):
    if len(d) < 2: return 0.0
    return 1.96 * statistics.pstdev(d) / math.sqrt(len(d))

def main(root, out_md, out_json):
    # fitness: (run,hash,build) -> fit ; score: hash -> level -> [s...]
    fit_map = {R8: {}, TPE: {}}
    sc_map = {R8: collections.defaultdict(list), TPE: collections.defaultdict(list)}
    run_fitmean = {R8: {}, TPE: {}}  # run -> mean fitness of f-level builds
    files = sorted(glob.glob(os.path.join(root, "run_*", "ga_detail.csv")))
    for fp in files:
        run = fp.split(os.sep)[-2]
        seen_b = {R8: {}, TPE: {}}
        with open(fp, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                h = row.get("genome_hash", "")
                if h not in NAMES: continue
                try:
                    fit = float(row["fitness"]); sc = float(row["score"])
                except Exception: continue
                lv, bd, sd = row["level"], row["build"], row["seed"]
                sc_map[h][lv].append(sc)
                if lv == "f":
                    seen_b[h][bd] = fit          # fitness 同 build 重复, 覆盖即可
                # 局级配对缓存
                if h == R8:
                    pass
        for h in (R8, TPE):
            for bd, fv in seen_b[h].items():
                fit_map[h][(run, bd)] = fv
            fvals = list(seen_b[h].values())
            if fvals:
                run_fitmean[h][run] = mean(fvals)
    # ---- A. build 级 CRN (fitness) ----
    keys_r8 = {k for k in fit_map[R8] if k[0] in run_fitmean[TPE]}
    keys_tpe = set(fit_map[TPE])
    common = keys_r8 & keys_tpe
    d_fit = [fit_map[R8][k] - fit_map[TPE][k] for k in common]
    # ---- B. 局级 CRN (score): 需再扫一遍按 (build,seed) 配对 ----
    # 为省内存, 局级配对在第二轮扫描做
    d_sc_by_level = collections.defaultdict(list)
    for fp in files:
        rows = {R8: [], TPE: []}
        with open(fp, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                h = row.get("genome_hash", "")
                if h in NAMES:
                    try: rows[h].append((row["level"], row["build"], row["seed"], float(row["score"])))
                    except Exception: pass
        if len(rows[R8]) and len(rows[TPE]):
            by = {h: collections.defaultdict(list) for h in (R8, TPE)}
            for h in (R8, TPE):
                for lv, bd, sd, sc in rows[h]:
                    by[h][(lv, bd, sd)].append(sc)
            for k in set(by[R8]) & set(by[TPE]):
                for a, b in zip(by[R8][k], by[TPE][k]):
                    d_sc_by_level[k[0]].append(a - b)
    # ---- C. 轮级 ----
    common_runs = set(run_fitmean[R8]) & set(run_fitmean[TPE])
    d_run = [run_fitmean[R8][r] - run_fitmean[TPE][r] for r in common_runs]
    # ---- 汇总 ----
    rep = {"runs_total": len(files), "runs_with_both": len(common_runs)}
    rep["fitness_abs"] = {}
    for h in (R8, TPE):
        rep["fitness_abs"][NAMES[h]] = {
            lv: {"n_build_obs": len(sc_map[h][lv])} for lv in LEVELS}
    # fitness 绝对值: 按 (run,build) 去重后的均值
    for h in (R8, TPE):
        allf = list(fit_map[h].values())
        rep["fitness_abs"][NAMES[h]]["all_builds_mean"] = round(mean(allf), 1) if allf else None
    def pack(ds):
        if not ds: return None
        w = sum(1 for x in ds if x > 0)
        return {"n": len(ds), "mean_d": round(mean(ds), 1),
                "std": round(statistics.pstdev(ds), 1) if len(ds) > 1 else 0,
                "ci95": round(ci95(ds), 1),
                "r8_win_rate": round(w / len(ds) * 100, 1)}
    rep["A_build_crn_fitness"] = pack(d_fit)
    rep["B_score_crn_by_level"] = {lv: pack(v) for lv, v in sorted(d_sc_by_level.items())}
    rep["C_run_pair"] = pack(d_run) | {"m_r8": round(mean([run_fitmean[R8][r] for r in common_runs]), 1),
                                       "m_tpe": round(mean([run_fitmean[TPE][r] for r in common_runs]), 1)} if d_run else None
    a = rep["A_build_crn_fitness"]
    if a:
        fr = mean([v for (r, b), v in fit_map[R8].items()])
        ft = mean([v for (r, b), v in fit_map[TPE].items()])
        rep["D_71k_translate"] = {"r8_baseline": 71415.6, "m_r8_fit": round(fr, 1),
                                  "m_tpe_fit": round(ft, 1), "diff": round(fr - ft, 1),
                                  "tpe_at_71k_scale": round(71415.6 - (fr - ft), 1)}
    json.dump(rep, open(out_json, "w"), ensure_ascii=False, indent=1)
    # ---- 报告 ----
    L = ["# GA 口径回验: R8 冠军 vs TPE best（GA 老考场复刻, artifact 10543408803）", "",
         f"- 轮数: {len(files)}, 双方同场轮数: {rep['runs_with_both']}",
         "- CRN 口径: 同轮同 build（fitness, 种子池同） / 同轮同 build+seed（score, 单局同种子）", ""]
    if a: L += [f"## A. build 级 CRN 配对（fitness, 71k 同口径）",
                f"- n={a['n']} 对, 均值差 {a['mean_d']} ±{a['ci95']} (95%CI), std {a['std']}, r8 胜率 {a['r8_win_rate']}%"]
    for lv, st in rep["B_score_crn_by_level"].items():
        if st: L.append(f"- B 局级[{lv}]: n={st['n']}, 差 {st['mean_d']} ±{st['ci95']}, r8 胜率 {st['r8_win_rate']}%")
    c = rep.get("C_run_pair")
    if c: L += ["", "## C. 轮级配对（每轮精评均值相减, GA 实战视角）",
                f"- n={c['n']} 轮, 差 {c['mean_d']} ±{c['ci95']}, r8 胜率 {c['r8_win_rate']}%",
                f"- r8 轮均 {c['m_r8']}, tpe 轮均 {c['m_tpe']}"]
    d = rep.get("D_71k_translate")
    if d: L += ["", "## D. 71k 口径折算",
                f"- r8 老账基准 {d['r8_baseline']}",
                f"- 本考场 r8 fitness 均值 {d['m_r8_fit']}, tpe 均值 {d['m_tpe_fit']}, 差 {d['diff']}",
                f"**=> TPE best 折算到 r8 老账口径 ≈ {d['tpe_at_71k_scale']}**"]
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "eval_runs"
    main(root, "pair_report.md", "pair_report.json")
