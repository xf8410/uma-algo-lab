# -*- coding: utf-8 -*-
"""交叉熵方法 CEM（分布估计算法 EDA 代表）。

入选理由：不维护协方差矩阵、只用精英样本重拟合均值方差，47 维上
每代 12 次评估即可推进；与 CMA-ES 形成"重/轻"对照。方差下限防早熟。
"""
from typing import List

import numpy as np

from .base import Optimizer


class CemOptimizer(Optimizer):
    name = "cem"

    def __init__(self, dim: int, seed: int = 7, popsize: int = 12, elite_frac: float = 0.2, center=None):
        super().__init__(dim, seed, center)
        self.rng = np.random.default_rng(seed)
        self.popsize = popsize
        self.n_elite = max(2, int(popsize * elite_frac))
        self.mu = np.asarray(self.center, dtype=float)
        self.sigma = np.full(dim, 0.08)
        self.sigma_min = 0.012
        self.batch: List[tuple] = []
        self._first = True

    def ask(self) -> List[float]:
        if self._first:
            self._first = False
            # center 锚点不进 batch：分数只记 best，不参与精英更新（防混样污染第一代）
            return np.asarray(self.center, dtype=float).tolist()
        if len(self.batch) >= self.popsize:
            raise RuntimeError("CEM 本代候选未消费完（tell 次数不足），ask 预算错配")
        x = np.clip(self.rng.normal(self.mu, self.sigma), 0.0, 1.0)
        self.batch.append((x, None))
        return x.tolist()

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        for i, (bx, _) in enumerate(self.batch):
            if np.allclose(bx, x, atol=1e-6):
                self.batch[i] = (bx, score)
                break
        if len(self.batch) == self.popsize and all(s is not None for _, s in self.batch):
            self.batch.sort(key=lambda t: -t[1])
            elite = np.stack([bx for bx, _ in self.batch[: self.n_elite]])
            self.mu = elite.mean(axis=0)
            self.sigma = np.maximum(elite.std(axis=0) + 0.008, self.sigma_min)
            self.batch = []

    @property
    def best_x(self) -> np.ndarray:
        return self.mu
