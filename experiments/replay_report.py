# -*- coding: utf-8 -*-
"""最优基因组回放 → 育成过程报告（逐回合决策日志）。

用法：
  python experiments/replay_report.py --genome best_genome.toml --uma 111501 \
      --seed 1001 --runs 1 --out replay/
依赖 UMA_BENCH_BIN（编译好的 bench_base，需含 --genome-file 能力）。
产出：replay/bench_base_decision_<uma>_<seed>.csv + 汇总 markdown。
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True, help="基因组覆盖层 TOML 路径")
    ap.add_argument("--uma", type=int, required=True)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--friend", type=int, default=303054)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="replay")
    args = ap.parse_args()

    bench = os.environ.get("UMA_BENCH_BIN", "bench_base")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [bench, "--uma", str(args.uma), "--trainer", "handwritten", "--log",
           "--runs", str(args.runs), "--seed", str(args.seed),
           "--genome-file", args.genome, "--out", str(outdir)]
    if args.deck:
        cmd += ["--deck", args.deck]
    else:
        cmd += ["--friend", str(args.friend)]
    print("回放:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    # 汇总最近一份决策日志为 markdown 摘要
    logs = sorted(outdir.glob("bench_base_decision_*.csv"))
    if not logs:
        print("未找到决策日志文件")
        return
    latest = logs[-1]
    rows = list(csv.DictReader(open(latest, newline="", encoding="utf-8")))
    md = [f"# 育成过程报告（回放）", "",
          f"- 基因组: `{args.genome}`", f"- 马娘: {args.uma}，seed {args.seed}，局数 {args.runs}",
          f"- 决策日志: `{latest.name}`（{len(rows)} 行）", "",
          "| 回合 | 阶段 | 动作 | 得分 |", "|---|---|---|---|"]
    for r in rows[:400]:
        md.append("| {turn} | {phase} | {action} | {score} |".format(
            turn=r.get("turn", ""), phase=r.get("phase", ""),
            action=(r.get("action") or r.get("choice") or "")[:60],
            score=r.get("score_after", r.get("vital", ""))))
    (outdir / "育成过程报告.md").write_text("\n".join(md), encoding="utf-8")
    print(f"报告已写: {outdir/'育成过程报告.md'}")


if __name__ == "__main__":
    main()
