# -*- coding: utf-8 -*-
"""模拟退火：单点邻域搜索 + 温度跳出局部最优。

入选理由：GA 已收敛的精细盆地内，SA 的逐维扰动+Metropolis 准则在极小
预算下比种群法省评估次数；几何降温保证收敛。整数字段扰动后 round。
"""
import math
from typing import List

import numpy as np

from .base import Optimizer


class SaOptimizer(Optimizer):
    name = "sa"

    def __init__(self, dim: int, seed: int = 7, t0: float = 0.05, t1: float = 0.004, steps: int = 400, center=None):
        super().__init__(dim, seed, center)
        self.rng = np.random.default_rng(seed)
        self.t0, self.t1, self.steps = t0, t1, max(steps, dim)
        self.cur = np.asarray(self.center, dtype=float)
        self.cur_score = -np.inf
        self.best_x = self.cur.copy()
        self.best_score = -np.inf
        self.k = 0

    def _temperature(self) -> float:
        r = self.k / max(1, self.steps)
        return self.t0 * (self.t1 / self.t0) ** r

    def _neighbor(self) -> np.ndarray:
        # 步长与温度联动：热时大步，冷时小步
        step = self._temperature()
        x = self.cur + self.rng.normal(0, step, self.dim)
        # 每步随机选 1/4 维度动，其余不动（高维下比全维扰动更稳）
        mask = self.rng.random(self.dim) < 0.25
        x = np.where(mask, x, self.cur)
        return np.clip(x, 0.0, 1.0)

    def ask(self) -> List[float]:
        if self.k == 0:
            return self.cur.tolist()
        return self._neighbor().tolist()

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        if self.k == 0:
            self.cur = np.asarray(x)
            self.cur_score = score
            self.best_x, self.best_score = self.cur.copy(), score
            self.k += 1
            return
        x = np.asarray(x)
        t = max(1e-6, self._temperature())
        if score >= self.cur_score or self.rng.random() < math.exp((score - self.cur_score) / (t * 3000 + 1)):
            self.cur, self.cur_score = x, score
        if score > self.best_score:
            self.best_x, self.best_score = x.copy(), score
        self.k += 1
