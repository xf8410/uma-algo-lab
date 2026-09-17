# -*- coding: utf-8 -*-
"""优化器统一接口：全部工作在 [0,1]^d 归一化空间。

ask() 产出候选向量，tell() 回填 fitness；评估次数（预算）由外层 compare.py 控制，
保证不同优化器在同预算下赛马。整数字段在 decode 时 round，优化器无感知。
"""
import time
from abc import ABC, abstractmethod
from typing import List, Optional


class Optimizer(ABC):
    name = "base"

    def __init__(self, dim: int, seed: int = 7, center: Optional[List[float]] = None):
        self.dim = dim
        self.rng_seed = seed
        # warm-start 起点：GA 冠军向量（None = 空间中心 0.5，向后兼容）
        self.center = [min(1.0, max(0.0, float(v))) for v in center] if center else [0.5] * dim
        self.history: List[dict] = []
        self.t0 = time.time()

    @abstractmethod
    def ask(self) -> List[float]:
        ...

    @abstractmethod
    def tell(self, x: List[float], score: float) -> None:
        ...

    def log(self, x: List[float], score: float) -> None:
        self.history.append({"t": round(time.time() - self.t0, 1), "score": score})

    @property
    def best(self) -> Optional[float]:
        return max((h["score"] for h in self.history), default=None)
