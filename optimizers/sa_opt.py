# -*- coding: utf-8 -*-
"""模拟退火 v2：自适应温度尺度 + 最优重启 + reheat。

v1 排表垫底的三个根因与修法（2026-10-17 sa_compare 5 seed 数据）：
  1. 接受准则的温度常数写死 t*3000+1，与分数尺度（~63000）不匹配，
     exp((ds)/(t*3000+1)) 在冷段永远 << 1，实际退化为纯贪心爬山，
     掉进盆地就出不来（seed 62672.1 崩盘的直接原因）。
     修：温度尺度 = 观测分数 std（首次评估后自适应），接受概率回到
     真正的 Metropolis 形态。
  2. 无重启：随机游走偏出冠军盆地后没有任何回拉机制。
     修：连续 no-improve 计数触发 best-restart（回到历史最优点重烧）。
  3. 单调降温没有 reheat：后半程全是小步贪心。
     修：重启时温度回弹到 t0*reheat_factor，保留盆地内二次探索。

入选理由不变：GA 已收敛的精细盆地内，SA 的逐维扰动 + Metropolis 准则
在极小预算下比种群法省评估次数；整数字段扰动后由 decode round。
"""
import math
from typing import List, Optional

import numpy as np

from .base import Optimizer


class SaOptimizer(Optimizer):
    name = "sa"
    # v2 超参：从 5 seed 排表反推（见 playbook/notes/20261017_sa_v2.md）
    def __init__(self, dim: int, seed: int = 7, t0: float = 0.05, t1: float = 0.004,
                 steps: int = 400, center=None,
                 restart_after: int = 8, reheat_factor: float = 0.5):
        super().__init__(dim, seed, center)
        self.rng = np.random.default_rng(seed)
        self.t0, self.t1, self.steps = t0, t1, max(steps, dim)
        self.restart_after = restart_after
        self.reheat_factor = reheat_factor
        self.cur = np.asarray(self.center, dtype=float)
        self.cur_score = -np.inf
        self.best_x = self.cur.copy()
        self.best_score = -np.inf
        self.k = 0            # 总步数（降温进度用）
        self.k_eff = 0        # 有效降温步（reheat 回退）
        self.no_improve = 0   # 连续无 best 改善计数
        self.score_scale = None  # 分数尺度（自适应温度）

    def _temperature(self) -> float:
        r = min(1.0, self.k_eff / max(1, self.steps))
        return self.t0 * (self.t1 / self.t0) ** r

    def _neighbor(self) -> np.ndarray:
        step = self._temperature()
        x = self.cur + self.rng.normal(0, step, self.dim)
        mask = self.rng.random(self.dim) < 0.25
        x = np.where(mask, x, self.cur)
        return np.clip(x, 0.0, 1.0)

    def _metropolis(self, delta: float, t: float) -> bool:
        if delta >= 0:
            return True
        # 温度尺度自适应：用观测分数 std 做分母（v1 的 3000 常数废除）
        scale = self.score_scale if self.score_scale else 100.0
        return self.rng.random() < math.exp(delta / max(1e-9, t * scale))

    def ask(self) -> List[float]:
        if self.k == 0:
            return self.cur.tolist()
        return self._neighbor().tolist()

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        x = np.asarray(x, dtype=float)
        # 分数尺度自适应：观测极差（首两个样本起）
        if self.score_scale is None and len(self.history) >= 2:
            scores = [h["score"] for h in self.history]
            rng_ = max(scores) - min(scores)
            self.score_scale = max(10.0, rng_)
        if self.k == 0:
            self.cur, self.cur_score = x, score
            self.best_x, self.best_score = x.copy(), score
            self.k += 1
            self.k_eff += 1
            return
        t = max(1e-6, self._temperature())
        delta = score - self.cur_score
        if self._metropolis(delta, t):
            self.cur, self.cur_score = x, score
        if score > self.best_score:
            self.best_x, self.best_score = x.copy(), score
            self.no_improve = 0
        else:
            self.no_improve += 1
        # 最优重启：连续无改善 → 回历史最优重烧（温度回弹 reheat_factor）
        if self.no_improve >= self.restart_after:
            self.cur = self.best_x.copy()
            self.cur_score = self.best_score
            self.no_improve = 0
            self.k_eff = max(0, self.k_eff - int(self.steps * self.reheat_factor))
        self.k += 1
        self.k_eff += 1
