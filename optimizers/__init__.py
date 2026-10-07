# -*- coding: utf-8 -*-
from .base import Optimizer
from .cmaes_opt import CmaesOptimizer
from .cem_opt import CemOptimizer
from .map_elites_opt import MapElitesOptimizer
from .sa_opt import SaOptimizer
from .tpe_opt import TpeOptimizer

REGISTRY = {
    "cmaes": CmaesOptimizer,
    "tpe": TpeOptimizer,
    "sa": SaOptimizer,
    "cem": CemOptimizer,
    "map_elites": MapElitesOptimizer,
}
