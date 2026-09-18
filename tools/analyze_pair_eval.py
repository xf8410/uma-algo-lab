#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA 口径回验 v3：r8champ vs tpe_best 逐轮配对分析（身份定位实锤版）

已实锤的结构事实（run_1/run_2 人工验证）:
  1. artifact run_N ↔ 日志第 N 条 "最优（精评）fitness=X 基因组哈希 H"（顺序一一对应）
  2. ga_detail.csv 的 s 块按评估顺序排: 第1个哈希=个体0=r8champ注入, 第2个=个体1=tpe_best注入,
     第3/4个=随机个体（每轮 CSV 哈希是轮内实例哈希, 跨轮不同, 不能直接用日志哈希匹配）
  3. f 块: 4 个体 × 各自 build × 60 runs(seed=42) 全有精评; h 块: 轮内最优 × 1 build × 40 runs(seed=43)
  4. 轮最优个体 f 均值 round1 == 日志 fitness（校验锚）

口径:
  A. 绝对分: F_r8 / F_tpe（f 层 fitness, 无条件全轮集合）+ 71k 老账折算
  B. 轮级 CRN: 同轮个体0 vs 个体1 且 f build 相同 → d=F_r8-F_tpe（同 build 同种子池, 最严格）
  C. 局级 CRN: s 层同 (build,seed) score 配对（20 runs/轮）
  D. h 层: 按轮最优身份分组（r8 赢轮 / tpe 赢轮 / 随机赢轮）
"""
import csv, glob, json, math, os, statistics, sys, collections

R8_TAG, TPE_TAG = "r8champ", "tpe_best"

def mean(xs): return sum(xs) / len(xs)
def ci95(d):
    if len(d) < 2: return 0.0
    return 1.96 * statistics.pstdev(d) / math.sqrt(len(d))
def pct(xs, p):
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(len(xs) * p)))
    return xs[k]

def main(root, seq_path, out_md, out_json):
    seq = []
    with open(seq_path) as fh:
        for l in fh:
            f, h = l.strip().split(",")
            seq.append((float(f), h))
    nA = len(seq)
    # 收集
    F = {R8_TAG: [], TPE_TAG: []}            # f 层 fitness（每轮个体0/1 各一值）
    Fb = {R8_TAG: [], TPE_TAG: []}           # (fitness, build)
    d_pair = []                               # B: 轮级配对差（同 build）
    d_pair_any = []                           # B2: 不限 build 的轮级配对差
    d_score = []                              # C: 局级配对差（s 层）
    h_by = {R8_TAG: [], TPE_TAG: [], "rand": []}
    ok_verify = 0; bad_verify = 0
    runs_scanned = 0
    files = sorted(glob.glob(os.path.join(root, "run_*", "ga_detail.csv")),
                   key=lambda p: int(p.split(os.sep)[-2].split("_")[1]))
    for idx, fp in enumerate(files):
        if idx >= nA: break
        rows_s, rows_f, rows_h = [], [], []
        with open(fp, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.reader(fh):
                if not row or row[0] == "level": continue
                lv = row[0]
                if lv == "s": rows_s.append(row)
                elif lv == "f": rows_f.append(row)
                elif lv == "h": rows_h.append(row)
        if not rows_s: continue
        # s 块个体顺序（首现序）→ 个体0/1
        order, seen = [], set()
        for r in rows_s:
            if r[1] not in seen:
                seen.add(r[1]); order.append(r[1])
        if len(order) < 2: continue
        h0, h1 = order[0], order[1]
        runs_scanned += 1
        # f 块: 个体0/1 的 fitness 均值与 build
        fby = collections.defaultdict(list)
        fbd = {}
        for r in rows_f:
            fby[r[1]].append(float(r[2])); fbd[r[1]] = r[3]
        if h0 in fby and h1 in fby:
            f0 = mean(fby[h0]); f1 = mean(fby[h1])
            F[R8_TAG].append(f0); F[TPE_TAG].append(f1)
            Fb[R8_TAG].append((f0, fbd[h0])); Fb[TPE_TAG].append((f1, fbd[h1]))
            if fbd[h0] == fbd[h1]:
                d_pair.append(f0 - f1)
            d_pair_any.append(f0 - f1)
            # 校验: 轮最优 = f 块最大个体? 其均值 round1 == 日志?
            best_h = max(fby, key=lambda h: mean(fby[h]))
            if abs(round(mean(fby[best_h]), 1) - seq[idx][0]) < 0.05:
                ok_verify += 1
            else:
                bad_verify += 1
        # s 局级 CRN: 个体0 vs 个体1, 同 (build, seed)
        s0 = collections.defaultdict(list); s1 = collections.defaultdict(list)
        for r in rows_s:
            if r[1] == h0: s0[(r[3], r[4])].append(float(r[5]))
            elif r[1] == h1: s1[(r[3], r[4])].append(float(r[5]))
        for k in set(s0) & set(s1):
            for a, b in zip(s0[k], s1[k]):
                d_score.append(a - b)
        # h 层归属: 轮最优身份
        if rows_h and rows_f:
            hb = collections.defaultdict(list)
            for r in rows_h:
                hb[r[1]].append(float(r[2]))
            for h, v in hb.items():
                if h not in fby: continue
                if abs(round(mean(fby[h]), 1) - seq[idx][0]) < 0.05:
                    tag = seq[idx][1]
                    key = TPE_TAG if tag == "e9843ff378fb8f95" else (R8_TAG if tag == "e52237aeb5b9b2b7" else "rand")
                    h_by[key].extend(v)
    # 汇总
    rep = {"runs_seq": nA, "runs_scanned": runs_scanned,
           "verify_ok": ok_verify, "verify_bad": bad_verify}
    rep["A_abs"] = {}
    for tag in (R8_TAG, TPE_TAG):
        v = F[tag]
        if v:
            rep["A_abs"][tag] = {"n": len(v), "mean": round(mean(v), 1),
                                 "std": round(statistics.pstdev(v), 1),
                                 "p10": round(pct(v, .10), 1), "p50": round(pct(v, .50), 1),
                                 "p90": round(pct(v, .90), 1), "max": round(max(v), 1)}
    m_r8, m_tpe = rep["A_abs"][R8_TAG]["mean"], rep["A_abs"][TPE_TAG]["mean"]
    rep["A_71k"] = {"r8_baseline": 71415.6, "diff_r8_minus_tpe": round(m_r8 - m_tpe, 1),
                    "tpe_at_71k": round(71415.6 - (m_r8 - m_tpe), 1)}
    def pack(ds):
        if not ds: return None
        w = sum(1 for x in ds if x > 0)
        return {"n": len(ds), "mean_d": round(mean(ds), 1),
                "std": round(statistics.pstdev(ds), 1) if len(ds) > 1 else 0,
                "ci95": round(ci95(ds), 1), "r8_win_rate": round(w / len(ds) * 100, 1)}
    rep["B_round_crn"] = pack(d_pair)
    rep["B_round_crn_anybuild"] = pack(d_pair_any)
    rep["C_score_crn"] = pack(d_score)
    rep["D_holdout"] = {k: {"n": len(v), "mean": round(mean(v), 1) if v else None,
                            "std": round(statistics.pstdev(v), 1) if len(v) > 1 else None}
                        for k, v in h_by.items() if v}
    json.dump(rep, open(out_json, "w"), ensure_ascii=False, indent=1)
    L = ["# GA 口径回验 v3: R8 冠军 vs TPE best（老考场复刻, 9213 轮）", "",
         f"- 扫描 {runs_scanned} 轮 / 序列 {nA} 轮; 校验(轮最优 f 均值==日志fitness) OK {ok_verify} / BAD {bad_verify}",
         "- 身份: s 块评估序个体0=r8champ注入, 个体1=tpe_best注入（run_1/run_2 人工实锤）", ""]
    for tag in (R8_TAG, TPE_TAG):
        a = rep["A_abs"].get(tag)
        if a: L.append(f"- **{tag}**: n={a['n']} 精评fitness {a['mean']} ±{a['std']} (P10 {a['p10']} / P50 {a['p50']} / P90 {a['p90']}, max {a['max']})")
    if "A_71k" in rep:
        L += ["", f"## 71k 口径折算", f"- r8 老账基准 71415.6；本考场 r8={m_r8}, tpe={m_tpe}, 差 {rep['A_71k']['diff_r8_minus_tpe']}",
              f"- **=> TPE best 折算老账口径 ≈ {rep['A_71k']['tpe_at_71k']}**"]
    b = rep["B_round_crn"]
    if b: L += ["", f"## B. 轮级 CRN（同轮同 f build, 同种子池）", f"- n={b['n']}, d(r8-tpe)={b['mean_d']} ±{b['ci95']}, std {b['std']}, r8 胜率 {b['r8_win_rate']}%"]
    b2 = rep["B_round_crn_anybuild"]
    if b2: L.append(f"- 不限 build: n={b2['n']}, d={b2['mean_d']} ±{b2['ci95']}, r8 胜率 {b2['r8_win_rate']}%")
    c = rep["C_score_crn"]
    if c: L += ["", f"## C. 局级 CRN（s 层同 build+seed 单局配对）", f"- n={c['n']}, d={c['mean_d']} ±{c['ci95']}, r8 胜率 {c['r8_win_rate']}%"]
    d = rep.get("D_holdout")
    if d: L += ["", "## D. holdout（seed=43, 轮最优分组）", "| 轮最优身份 | n | 均值 | std |", "|---|---|---|---|"]
    if d:
        for k, v in d.items(): L.append(f"| {k} | {v['n']} | {v['mean']} | {v['std']} |")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "eval_runs"
    seqp = sys.argv[2] if len(sys.argv) > 2 else "data/eval_seq_35332465533.csv"
    main(root, seqp, "pair_report.md", "pair_report.json")
