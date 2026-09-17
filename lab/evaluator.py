# -*- coding: utf-8 -*-
"""评估总线：bench CLI 黑盒调用 + CRN 种子纪律 + (hash, level) 缓存。

评估协议与 Rust GA 同源（2026-09-17 对齐 + 局数规则升级）：
- smoke 档：3 build（GA select_screen_builds 速/耐/智最多）× 100 局 × base_seed 42
- full 档：7 build × 100 局 × seed 42 + holdout 7 build × 100 局 × seed 43
- 种子分层：训练 42（所有个体共用=CRN）/ 验收 9001-9003（换种子）/ holdout 43
- 局数规则：每 build 100 局（用户规则 2026-09-17）——小样本极端值会误杀好算法
- fitness = mean(score) - 800 * race_fail_rate（与 Rust GA 同式）
注意：GA 冠军 71415.6 是「3 build×20 局」口径，与 100 局口径不直接可比，
基准以本次实测 center 分为准。
"""
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BENCH_BIN = os.environ.get("UMA_BENCH_BIN", "bench_base")

SMOKE_SEEDS = (42,)
SMOKE_RUNS = 100
FULL_SEEDS = (42,)
FULL_RUNS = 100
HOLDOUT_SEEDS = (43,)
HOLDOUT_RUNS = 100
FAIL_PENALTY = 800.0
# GA 初筛 select_screen_builds：速/耐/智卡最多的 3 个 build（平局取声明序靠前）
SCREEN_BUILDS = "speed,stamina,power_wisdom"


class EvaluateError(RuntimeError):
    pass


class Evaluator:
    def __init__(self, uma: int, deck: Optional[str] = None, friend: int = 303054,
                 cache_path: Optional[str] = None, bench_bin: Optional[str] = None):
        self.uma = int(uma)
        self.deck = deck
        self.friend = int(friend)
        self.bench_bin = bench_bin or BENCH_BIN
        self.cache_path = cache_path
        self.cache: Dict[Tuple[str, str], Dict] = {}
        self.n_evals = 0
        if cache_path and Path(cache_path).exists():
            self._load_cache()

    def _load_cache(self) -> None:
        try:
            raw = json.loads(Path(self.cache_path).read_text())
            self.cache = {tuple(k.split("|", 1)): v for k, v in raw.items()}
        except Exception:
            self.cache = {}

    def _save_cache(self) -> None:
        if self.cache_path:
            Path(self.cache_path).write_text(
                json.dumps({"|".join(k): v for k, v in self.cache.items()}, ensure_ascii=False)
            )

    def _run_bench(self, genome_toml: str, seed: int, runs: int,
                   builds: Optional[str] = None) -> Tuple[float, float]:
        """跑一组 bench，返回 (mean_score, race_fail_rate)"""
        with tempfile.TemporaryDirectory(prefix="umalab_") as td:
            gpath = Path(td) / "genome.toml"
            gpath.write_text(genome_toml, encoding="utf-8")
            cmd = [self.bench_bin,
                   "--uma", str(self.uma), "--trainer", "handwritten",
                   "--runs", str(runs), "--seed", str(seed),
                   "--genome-file", str(gpath), "--out", td]
            if builds:
                cmd += ["--builds", builds]
            if self.deck:
                cmd += ["--deck", self.deck]
            else:
                cmd += ["--friend", str(self.friend)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                raise EvaluateError(f"bench 失败({proc.returncode}): {proc.stderr[-800:]}")
            res = Path(td) / "bench_base_results.csv"
            if not res.exists():
                raise EvaluateError(f"bench 未产出 results.csv: {proc.stdout[-400:]}")
            scores, fails = [], []
            with open(res, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    scores.append(float(row["score"]))
                    race_ok = row.get("race_gate_ok", "")
                    if race_ok not in ("", "1", "true", "True"):
                        fails.append(1.0)
                    else:
                        fails.append(0.0)
            if not scores:
                raise EvaluateError("results.csv 无数据行")
            mean = sum(scores) / len(scores)
            fail_rate = sum(fails) / len(fails)
            return mean, fail_rate

    def evaluate(self, genome: Dict, level: str = "smoke") -> Dict:
        """评估一个 genome dict；CRN 同种子 + 缓存。返回 {score, holdout, level, ...}"""
        from .genome import genome_hash, genome_to_toml
        gh = genome_hash(genome)
        key = (gh, level)
        if key in self.cache:
            return dict(self.cache[key], cached=True)
        toml_text = genome_to_toml(genome)
        seeds = FULL_SEEDS if level == "full" else SMOKE_SEEDS
        runs = FULL_RUNS if level == "full" else SMOKE_RUNS
        # GA 口径：smoke 只跑初筛 3 build；full/holdout 跑全部 preset builds
        builds = None if level == "full" else SCREEN_BUILDS
        means, fail_rates = [], []
        for s in seeds:
            m, fr = self._run_bench(toml_text, s, runs, builds=builds)
            means.append(m)
            fail_rates.append(fr)
            self.n_evals += 1
        score = sum(means) / len(means) - FAIL_PENALTY * (sum(fail_rates) / len(fail_rates))
        out = {"hash": gh, "level": level, "score": round(score, 1),
               "per_seed": [round(m, 1) for m in means],
               "fail_rate": round(sum(fail_rates) / len(fail_rates), 4)}
        if level == "full":
            hm, hfr = self._run_bench(toml_text, HOLDOUT_SEEDS[0], HOLDOUT_RUNS)
            out["holdout"] = round(hm, 1)
            self.n_evals += 1
        self.cache[key] = out
        self._save_cache()
        return out
