# -*- coding: utf-8 -*-
"""CMA-ES：连续黑盒调参的黄金标准。

入选理由：GA（均匀交叉+高斯突变）在 47 维上收敛后，CMA-ES 的完整协方差
自适应能沿参数耦合方向做精细步长收缩，正是"GA 收敛后再挤分"的对症工具。
依赖 pip cmaes（轻量实现，Ikeru 原作）。
"""
from typing import List

import numpy as np
from cmaes import CMA

from .base import Optimizer


class CmaesOptimizer(Optimizer):
    name = "cmaes"

    def __init__(self, dim: int, seed: int = 7, sigma0: float = 0.12):
        super().__init__(dim, seed)
        self.opt = CMA(mean=np.full(dim, 0.5), sigma=sigma0,
                       population_size=max(8, 4 + int(3 * np.log(dim))),
                       seed=seed)
        self.pending: dict = {}

    def ask(self) -> List[float]:
        x = self.opt.ask().tolist()
        return [min(1.0, max(0.0, v)) for v in x]

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        self.opt.tell(np.asarray(x), -score)  # cmaes 最小化
