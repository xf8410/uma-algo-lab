# -*- coding: utf-8 -*-
"""赛马娘育成策略算法实验室。

模拟器（xulai1001/umaai-rs）是固定黑盒，本仓库只做三件事：
1. 把参数向量喂给它（genome.py 与 Rust parse_override_toml 严格对偶）
2. 收集分数（evaluator.py：CRN 同种子、缓存、holdout）
3. 用不同优化器在同一协议下赛马（optimizers/）
"""
__version__ = "0.1.0"
