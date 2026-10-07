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

# CVaR 评估用多种子：与训练种子 42 错开、互质，保证 CRN 同分布但样本外异。
# 5 种子 × 100 局 × 3 build ≈ 1500 局/次评估（vs smoke 1 seed × 100 × 3 = 300 局/5x），
# alpha 默认 0.2 → ceil(5*0.2)=1，取最差 1 种子作 fitness。算法预算 20 eval ≈ ~50 分钟。
CVAR_SEEDS: Tuple[int, ...] = (101, 103, 107, 109, 113)
CVAR_RUNS = 100
CVAR_DEFAULT_ALPHA = 0.2


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
                    # bench_base results.csv 的自选比赛达标列是 free_race_ok（2026-09-18 修：
                    # 原误读不存在的 race_gate_ok，恒空 → 失败判别失效，fail_rate 恒 0）
                    race_ok = str(row.get("free_race_ok", "1")).strip().lower()
                    if race_ok not in ("", "1", "true"):
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

    def _multi_seed_means(self, toml_text: str, seeds: Tuple[int, ...], runs: int,
                          builds: Optional[str] = None) -> List[Tuple[float, float]]:
        """跑多种子 → [(mean, fail_rate), ...]；不写缓存（供 CVaR 内部用）"""
        results: List[Tuple[float, float]] = []
        for s in seeds:
            m, fr = self._run_bench(toml_text, s, runs, builds)
            results.append((m, fr))
            self.n_evals += 1
        return results

    def evaluate_cvar(self, genome: Dict, alpha: float = CVAR_DEFAULT_ALPHA,
                      seeds: Tuple[int, ...] = CVAR_SEEDS,
                      runs_per_seed: int = CVAR_RUNS) -> Dict:
        """CVaR 适应度评估（多种子下尾均值）。

        协议：
          1. 用多种子（默认 5，互质、与训练 seed 42 错开）跑同一 genome
          2. 每种子 per_seed = mean(runs_per_seed 局) - 800*fail_rate
          3. CVaR_α = 下尾 ceil(N*alpha) 种子的 per_seed 均值（风险敏感目标）
          4. 缓存键 (hash, "cvar{alpha}_{runs}")；不动原 evaluate()

        返回 dict 结构与 evaluate() 对齐：
          {score: cvar, per_seed: [...], fail_rate: 总均值, alpha, n_tail, cvar_seeds, level}

        用法示例：
          res = ev.evaluate_cvar(genome, alpha=0.2)   # 最差 1 个 seed 的分数
          res = ev.evaluate_cvar(genome, alpha=0.4) # 最差 2 个 seed 的分数均值
        """
        from .genome import genome_hash, genome_to_toml
        gh = genome_hash(genome)
        level = f"cvar{alpha}_{runs_per_seed}_{len(seeds)}s"
        key = (gh, level)
        if key in self.cache:
            return dict(self.cache[key], cached=True)

        toml_text = genome_to_toml(genome)
        results = self._multi_seed_means(toml_text, seeds, runs_per_seed,
                                         builds=SCREEN_BUILDS)
        # per_seed 分数（含 fail penalty，与 GA 同式）
        per_seed_scores = [m - FAIL_PENALTY * fr for m, fr in results]
        sorted_pairs = sorted(zip(per_seed_scores,
                                  [fr for _, fr in results]),
                              key=lambda t: t[0])
        n_tail = max(1, int(round(len(per_seed_scores) * alpha)))
        tail = sorted_pairs[:n_tail]
        cvar_score = sum(s for s, _ in tail) / len(tail)
        out = {
            "hash": gh, "level": level, "score": round(cvar_score, 1),
            "per_seed": [round(m, 1) for m, _ in results],
            "fail_rate": round(sum(fr for _, fr in results) / len(results), 4),
            "alpha": alpha, "n_tail": n_tail, "cvar_seeds": list(seeds),
        }
        self.cache[key] = out
        self._save_cache()
        return out
