# -*- coding: utf-8 -*-
"""贝叶斯优化（TPE）：高成本黑盒调参。

入选理由：评估一次 = 3×20 局模拟（分钟级），典型的昂贵黑盒。TPE 用
l(x)/g(x) 密度比建模"好点分布 vs 全体分布"，小预算下通常优于进化类。
依赖 optuna（sampler 静默模式）。
"""
import warnings
from typing import List

import numpy as np
import optuna

from .base import Optimizer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TpeOptimizer(Optimizer):
    name = "tpe"

    def __init__(self, dim: int, seed: int = 7, center=None):
        super().__init__(dim, seed, center)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
            study_name=f"umalab_tpe_{seed}",
        )
        # 首发固定评估 center（warm-start 锚点），后续 TPE 自主建模
        self.study.enqueue_trial(
            {f"x{i}": float(v) for i, v in enumerate(self.center)},
            skip_if_exists=True,
        )
        self.pending: dict = {}

    def ask(self) -> List[float]:
        t = self.study.ask()
        x = [t.suggest_float(f"x{i}", 0.0, 1.0) for i in range(self.dim)]
        self.pending[t.number] = x
        return x

    def tell(self, x: List[float], score: float) -> None:
        self.log(x, score)
        num = next(k for k, v in self.pending.items() if v == x)
        del self.pending[num]
        self.study.tell(num, score)
