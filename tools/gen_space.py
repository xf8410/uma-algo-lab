# -*- coding: utf-8 -*-
"""从模拟器仓库重新生成 lab/space.py（换 center 基准时用）。

用法：python tools/gen_space.py <umaai-rs 路径> <genomes_r5.json> <genomes_r6.json>
"""
import json
import re
import sys
from pathlib import Path

RS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("umaai-rs")
G5 = sys.argv[2] if len(sys.argv) > 2 else "genomes_r5.json"
G6 = sys.argv[3] if len(sys.argv) > 3 else "genomes_r6.json"

src = (RS / "crates/umasim/src/trainer/local_ramen_trainer.rs").read_text(encoding="utf-8", errors="replace")
i = src.find("pub struct ParamOverride"); j = src.find("}", i)
fields = re.findall(r"pub (\w+)\s*:\s*(?:Option<(\w+)>|\[(?:Option<(\w+)>); 3\])", src[i:j])
types = {n: (a or f"arr{b}") for n, a, b in fields}
assert len(types) == 70, f"字段数 {len(types)} != 70，Rust 结构体可能改版，需同步本脚本与 genome.py"

best = None
for f in (G5, G6):
    for g in json.load(open(f)):
        if best is None or g["fitness"] > best["fitness"]:
            best = g
print("center:", best["round"], best["uma"], best["fitness"], best["hash"])
print("提示：请以本输出为准更新 lab/space.py 头注释；字段结构变化时同步 genome.py。")
# 生成逻辑与首版一致，见仓库历史 v0.1.0；此处只做校验与提醒，避免双源维护。
