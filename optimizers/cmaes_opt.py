# -*- coding: utf-8 -*-
"""CMA-ES：连续黑盒调参的黄金标准。

入选理由：GA（均匀交叉+高斯突变）在 45 维上收敛后，CMA-ES 的完整协方差
自适应能沿参数耦合方向做精细步长收缩，正是"GA 收敛后再挤分"的对症工具。
依赖 pip cmaes（轻量实现，Ikeru 原作）。

cmaes>=0.12 的 tell() 只接受"整代"列表 list[(x, value)]，单条喂会断言失败，
所以这里用攒代模式：ask 从当前代候选池弹，tell 攒满 popsize 才更新分布。
"""
from typing import List

import numpy as np
from cmaes import CMA

from .base import Optimizer


class CmaesOptimizer(Optimizer):
    name = "cmaes"

    def __init__(self, dim: int, seed: int = 7, sigma0: float = 0.06, center=None):
        super().__init__(dim, seed, center)
        self.opt = CMA(mean=np.asarray(self.center, dtype=float), sigma=sigma0,
                       population_size=max(8, 4 + int(3 * np.log(dim))),
                       seed=seed)
        self._pool: List[List[float]] = []   # 当前代尚未派发的候选
        self._solutions: List[tuple] = []    # 当前代已评估的 (x, -score)
        self._first = True                   # 首发固定评估 center（锚点）

    def _refill_pool(self) -> None:
        self._pool = [[min(1.0, max(0.0, float(v))) for v in self.opt.ask()]
                      for _ in range(self.opt.population_size)]

    def ask(self) -> List[float]:
        if self._first:
            self._first = False
            return list(self.center)
        if not self._pool:
            self._refill_pool()
        return self._pool.pop(0)

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        self._solutions.append((np.asarray(x, dtype=float), -score))  # cmaes 最小化
        if len(self._solutions) >= self.opt.population_size:
            self.opt.tell(self._solutions)
            self._solutions = []
