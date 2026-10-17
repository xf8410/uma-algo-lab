# -*- coding: utf-8 -*-
from .base import Optimizer
from .cmaes_opt import CmaesOptimizer
from .cem_opt import CemOptimizer
from .ars_opt import ArsOptimizer
from .map_elites_opt import MapElitesOptimizer
from .powell_opt import PowellOptimizer
from .sa_opt import SaOptimizer
from .spsa_opt import SpsaOptimizer
from .tpe_opt import TpeOptimizer

REGISTRY = {
    "cmaes": CmaesOptimizer,
    "tpe": TpeOptimizer,
    "sa": SaOptimizer,
    "cem": CemOptimizer,
    "map_elites": MapElitesOptimizer,
    # 2026-10-17 第一梯队新增（docs/blackbox-selection.md）：
    # spsa 一次扰动估全维梯度（2 评估/步）；ars top-k 方向平均；
    # powell 自实现方向集（经典 DFO 对照）
    "spsa": SpsaOptimizer,
    "ars": ArsOptimizer,
    "powell": PowellOptimizer,
}
