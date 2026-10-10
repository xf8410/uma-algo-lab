# -*- coding: utf-8 -*-
"""MAP-Elites：Quality-Diversity 算法，把"单一最优解"换成"行为格子里的多个最优"。

为什么入选：GA 收敛到 71415.6 后卡死在单一基因组。但用户痛点是「摆烂诱因」——
同基因组在不同代码状态下表现差很大。MAP-Elites 维护一个 NxN grid，
每格存该行为区域的最优 genome，搜索时同时优化「质量」（fitness）和
「多样性」（行为描述符覆盖）。

行为描述符（bd，predictor 全称 Behavior Descriptor）：
  bh1 = vital_rest 归一化值 [0,1] → 决定「休息偏多/偏少」
  bh2 = pt_tradeoff 归一化值 [0,1] → 决定「PT 转换偏好激进/保守」
两个维度都直接决定育成节奏，是搜索空间卡死的主要嫌疑轴。

archive grid：5x5=25 cell，每 cell 留最优 (score, x)。
ask 策略：
  warmup 前 N 次：从 center + noise 全探索（填满 archive）
  之后：70% 从 archive 随机 cell 加 mutation / 30% 从 center 探索
比单点优化的关键好处：搜索预算用完后，能报「各行为区域冠军」，而不是单一最优点。
"""
from typing import List, Optional

import numpy as np

from .base import Optimizer


class MapElitesOptimizer(Optimizer):
    name = "map_elites"

    def __init__(self, dim: int, seed: int = 7, center: Optional[List[float]] = None,
                 n_bins: int = 5, mutation_rate: float = 0.12, sigma_explore: float = 0.18,
                 warmup: int = 8):
        super().__init__(dim, seed, center)
        self.rng = np.random.default_rng(seed)
        self.n_bins = n_bins
        self.mutation_rate = mutation_rate
        self.sigma_explore = sigma_explore
        self.warmup = warmup
        self.archive: dict = {}           # (c1, c2) -> (score, x_list)
        self.bh_indices = self._find_behavior_dims()
        self._n_asked = 0
        self._explore_count = 0
        self._mutate_count = 0

    def _find_behavior_dims(self):
        """找 vital_rest / pt_tradeoff 在 free_layout 中的扁平索引。
        推迟 import 避免循环依赖（base.py 已在 optimizers/ 顶部）。"""
        from lab.space import ALL_FIELDS
        from lab.genome import free_layout
        layout = free_layout()
        bh1, bh2 = None, None
        for i, (f, slot) in enumerate(layout):
            if slot is not None:
                continue
            if f.name == "vital_rest":
                bh1 = i
            elif f.name == "pt_tradeoff":
                bh2 = i
        if bh1 is None or bh2 is None:
            raise RuntimeError("找不到 vital_rest / pt_tradeoff 的扁平索引")
        return bh1, bh2

    def ask(self) -> List[float]:
        warmup_phase = self._n_asked < self.warmup
        sparse = len(self.archive) < self.n_bins ** 2 * 0.3
        if warmup_phase or sparse or self.rng.random() < 0.3 or not self.archive:
            # 探索：从 center + 大 noise（覆盖大范围行为格子）
            x = np.asarray(self.center, dtype=float) + self.rng.normal(
                0, self.sigma_explore, self.dim)
            self._explore_count += 1
        else:
            # 突变：从 archive 随机 cell 加小 noise（精修该行为区域）
            # 注：self.rng.choice(tuple_list) 会返回 ndarray，不能当 dict key。
            #     用 integers 拿索引再 list 取 key，确保拿到原生 tuple。
            keys = list(self.archive.keys())
            cell = keys[int(self.rng.integers(0, len(keys)))]
            _, base_x = self.archive[cell]
            x = np.asarray(base_x, dtype=float) + self.rng.normal(
                0, self.mutation_rate, self.dim)
            self._mutate_count += 1
        self._n_asked += 1
        return np.clip(x, 0.0, 1.0).tolist()

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        bh1_idx, bh2_idx = self.bh_indices
        # 行为描述符：vital_rest / pt_tradeoff 在 [0,1] 归一化空间的离散化
        c1 = min(self.n_bins - 1, int(round(float(x[bh1_idx]) * (self.n_bins - 1))))
        c2 = min(self.n_bins - 1, int(round(float(x[bh2_idx]) * (self.n_bins - 1))))
        key = (c1, c2)
        if key not in self.archive or score > self.archive[key][0]:
            self.archive[key] = (score, list(x))

    @property
    def best(self) -> Optional[float]:
        if not self.archive:
            return None
        return max(s for s, _ in self.archive.values())

    @property
    def coverage(self) -> float:
        return len(self.archive) / (self.n_bins ** 2)

    def archive_dump(self):
        """返回 5x5 grid (fill, score) 摘要，用于 report。
        grid 用 [[cell_score or None, ...]] 二维 list 序列化。"""
        grid = [[None]*self.n_bins for _ in range(self.n_bins)]
        for (c1, c2), (score, _) in self.archive.items():
            grid[c1][c2] = round(score, 1)
        return {"n_bins": self.n_bins,
                "filled_cells": len(self.archive),
                "coverage": round(self.coverage, 3),
                "best_score": self.best,
                "grid": grid,
                "explore_count": self._explore_count,
                "mutate_count": self._mutate_count,
                "n_asked": self._n_asked}
