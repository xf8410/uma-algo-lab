# -*- coding: utf-8 -*-
"""Powell 方向集法（自实现，纯 ask/tell）：逐方向一维线搜索 + 方向替换。

入选理由（docs/blackbox-selection.md #4）：经典 DFO 代表。scipy 版要
同步 callable 与 ask/tell 懒评估桥不兼容，自实现核心只有 80 行：
维护 d 个正交初始方向（坐标轴），每次沿当前方向做三点线搜索
（当前点 / +step / -step 择优步进），一个方向走完换下一个；
每轮结束用本轮总位移替换贡献最大的方向（Powell 共轭条件简化版）。

45 维 × 30 eval = 每方向平均 0.6 次评估——诚实说这预算对 Powell 很紧，
它给的是"经典方向集法 vs 进化系"的对照基准。
"""
from typing import List, Optional

import numpy as np

from .base import Optimizer


class PowellOptimizer(Optimizer):
    name = "powell"

    def __init__(self, dim: int, seed: int = 7, step: float = 0.1,
                 shrink: float = 0.5, center=None):
        super().__init__(dim, seed, center)
        self.rng = np.random.default_rng(seed)
        self.step, self.shrink = step, shrink
        self.best_x = np.asarray(self.center, dtype=float)
        self.best_score = -np.inf
        self.cur = self.best_x.copy()
        self.cur_score = -np.inf
        # 初始方向集：坐标轴
        self.dirs = [np.eye(dim)[i] for i in range(dim)]
        self.phase = "probe"   # probe（评估当前点）/ plus / minus / commit
        self.dir_idx = 0
        self.round_start = None
        self.round_start_score = -np.inf
        self.max_gain = 0.0
        self.max_gain_idx = 0
        self._candidate = None
        self._cand_score = -np.inf

    def ask(self) -> List[float]:
        if self.phase == "probe":
            self._candidate = self.cur
            return self.cur.tolist()
        d = self.dirs[self.dir_idx]
        if self.phase == "plus":
            self._candidate = np.clip(self.cur + self.step * d, 0.0, 1.0)
        else:  # minus
            self._candidate = np.clip(self.cur - self.step * d, 0.0, 1.0)
        return self._candidate.tolist()

    def _next_phase(self):
        if self.phase == "probe":
            self.phase = "plus"
        elif self.phase == "plus":
            self.phase = "minus"
        else:
            self.phase = "plus"
            self.dir_idx += 1
            if self.dir_idx >= len(self.dirs):
                # 一轮结束：方向替换 + 收缩步长 + 回到方向 0
                if self.round_start is not None:
                    disp = self.cur - self.round_start
                    if np.linalg.norm(disp) > 1e-12:
                        self.dirs[self.max_gain_idx] = disp / np.linalg.norm(disp)
                self.dir_idx = 0
                self.step *= self.shrink
                self.round_start = self.cur.copy()
                self.round_start_score = self.cur_score
                self.max_gain, self.max_gain_idx = 0.0, 0

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        if score > self.best_score:
            self.best_x, self.best_score = np.asarray(x, dtype=float).copy(), score
        if self.phase == "probe":
            self.cur_score = score
            self.round_start = self.cur.copy()
            self._next_phase()
            return
        gain = score - self.cur_score
        if gain > 0:
            self.cur, self.cur_score = self._candidate.copy(), score
            if gain > self.max_gain:
                self.max_gain, self.max_gain_idx = gain, self.dir_idx
        self._next_phase()
