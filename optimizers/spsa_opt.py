# -*- coding: utf-8 -*-
"""SPSA 同时扰动随机逼近：一次双侧扰动估全维梯度，每步只要 2 次评估。

入选理由（docs/blackbox-selection.md #1）：SA 退出主赛的根因是维度×预算墙——
1/4 维扰动在 45 维×30 eval 下每维只摸 7.5 次。SPSA 用 Bernoulli ±1 扰动
全维同时动，两个评估（正扰/负扰）就出一个 45 维梯度估计，预算效率翻 22 倍。
配 CRN（同种子正负扰对比）噪声对冲，这是它对噪声鲁棒的用法。

超参：a 步长（增益）、c 扰动半径、alpha/beta 衰减指数。标准取
alpha=0.602/beta=0.101（Spall 推荐的渐近最优），a 按 t0 归一。
"""
from typing import List

import numpy as np

from .base import Optimizer


class SpsaOptimizer(Optimizer):
    name = "spsa"

    def __init__(self, dim: int, seed: int = 7,
                 a: float = 0.08, c: float = 0.06, alpha: float = 0.602, beta: float = 0.101,
                 steps: int = 400, center=None):
        super().__init__(dim, seed, center)
        self.rng = np.random.default_rng(seed)
        self.a, self.c, self.alpha, self.beta = a, c, alpha, beta
        self.steps = max(steps, dim)
        self.best_x = np.asarray(self.center, dtype=float)
        self.best_score = -np.inf
        self.cur = self.best_x.copy()
        self.k = 0          # 步数（第 k 步产出第 2k/2k+1 次评估）
        self.plus = None    # 本步待评估的正扰点
        self.minus = None   # 本步待评估的负扰点

    def _ak(self) -> float:
        return self.a / (self.k + 1) ** self.alpha

    def _ck(self) -> float:
        return self.c / (self.k + 1) ** self.beta

    def ask(self) -> List[float]:
        if self.k == 0 and self.plus is None:
            # 首评估：当前中心（建立 best 基线）
            return self.cur.tolist()
        if self.plus is not None:
            return self.plus.tolist()
        # 生成一对 ± 扰动
        delta = self.rng.choice([-1.0, 1.0], size=self.dim)
        ck = self._ck()
        self._delta = delta
        self.plus = np.clip(self.cur + ck * delta, 0.0, 1.0)
        self.minus = np.clip(self.cur - ck * delta, 0.0, 1.0)
        return self.plus.tolist()

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        x = np.asarray(x, dtype=float)
        if score > self.best_score:
            self.best_x, self.best_score = x.copy(), score
        if self.k == 0 and self.plus is None:
            self.cur_score = score
            self.k = 1
            return
        if self.plus is not None and np.allclose(x, self.plus):
            self.plus_score = score
            return  # 等 minus
        if self.minus is not None and np.allclose(x, self.minus):
            # 梯度估计 + 步进
            # 用实际位移（clip 边界时正负侧不对称，取两侧真实差）
            actual = self.plus - self.minus
            g = np.where(actual != 0, (self.plus_score - score) / np.maximum(np.abs(actual), 1e-9) * np.sign(self._delta), 0.0)
            self.cur = np.clip(self.cur + self._ak() * g, 0.0, 1.0)
            self.cur_score = max(self.plus_score, score)
            self.plus = self.minus = None
            self.k += 1
            return
        raise RuntimeError("spsa tell 收到不认识的点")
