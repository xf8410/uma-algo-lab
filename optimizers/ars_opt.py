# -*- coding: utf-8 -*-
"""ARS 增强随机搜索：每代采 N 个方向，正负双侧评估，只用 top-k 方向更新。

入选理由（docs/blackbox-selection.md #2）：top-k 方向平均自带降噪
（比 OpenAI-ES 的全平均信噪比高），DeepMind 复现过它在连续控制上
打平 A3C 的战绩。45 维 × 30 eval 预算 = 5 代 × 每代 3 方向
（每方向 2 评估），每代都有真梯度信息。

超参：n_dir 每代方向数、topk 采纳方向数、step 步长、sigma 扰动半径。
"""
from typing import List

import numpy as np

from .base import Optimizer


class ArsOptimizer(Optimizer):
    name = "ars"

    def __init__(self, dim: int, seed: int = 7,
                 n_dir: int = 3, topk: int = 2, step: float = 0.05, sigma: float = 0.08,
                 center=None):
        super().__init__(dim, seed, center)
        self.rng = np.random.default_rng(seed)
        self.n_dir, self.topk, self.step, self.sigma = max(2, n_dir), max(1, topk), step, sigma
        self.best_x = np.asarray(self.center, dtype=float)
        self.best_score = -np.inf
        self.theta = self.best_x.copy()  # 策略参数（更新的对象）
        self.gen = 0                     # 当前代
        self.dirs: List[np.ndarray] = []  # 本代方向集
        self.pair = []                   # 本代已评 (idx, is_plus, score)
        self.pending_plus = True

    def ask(self) -> List[float]:
        # 每代开头：本代还没生成方向 → 生成
        if not self.dirs:
            self.dirs = [self.rng.normal(0, 1, self.dim) for _ in range(self.n_dir)]
            self.pair = []
            self.pending_plus = True
        idx = len(self.pair) // 2
        d = self.dirs[idx]
        if self.pending_plus:
            return np.clip(self.theta + self.sigma * d / np.linalg.norm(d), 0.0, 1.0).tolist()
        return np.clip(self.theta - self.sigma * d / np.linalg.norm(d), 0.0, 1.0).tolist()

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        x = np.asarray(x, dtype=float)
        if score > self.best_score:
            self.best_x, self.best_score = x.copy(), score
        idx = len(self.pair) // 2
        self.pair.append((idx, self.pending_plus, score))
        self.pending_plus = not self.pending_plus
        # 本代评完（2*n_dir 个）→ 更新
        if len(self.pair) == 2 * self.n_dir:
            rewards = []
            for i in range(self.n_dir):
                rp = next(s for (j, p, s) in self.pair if j == i and p)
                rm = next(s for (j, p, s) in self.pair if j == i and not p)
                rewards.append((rp - rm, self.dirs[i]))
            rewards.sort(key=lambda t: -abs(t[0]))
            top = rewards[: self.topk]
            denom = sum(abs(r) for r, _ in top) or 1.0
            update = sum(r * d / np.linalg.norm(d) for r, d in top) / denom
            self.theta = np.clip(self.theta + self.step * self.n_dir * update, 0.0, 1.0)
            self.gen += 1
            self.dirs = []
