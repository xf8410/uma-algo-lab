# 黑箱算法族大清单拆解：四轮清单 + 13 类黑箱族的合并筛选（v1）

> 日期：2026-10-17
> 输入：原始清单 + 三轮补充 + 爬塔类比轮 + 黑箱框架轮（DFO/NES/BO 变体/多保真/代理模型/系统辨识/MBRL/偏好学习/生成多样性/神经进化/模糊/约束多目标/主动学习）
> 问题定性（用户框架，已对齐）：**赛马娘对优化器 = 高成本带噪声黑箱；能读内存 = 可转白箱**。
> 育成 = 黑箱优化（参数面）+ 序列决策（决策面）+ 多目标资源分配（子问题）。
> 原则：**模拟器/实验室里已有的不重加**；每个新算法必须说清接哪个面、预期多大收益、成本多少。

---

## 〇、先对齐分数口径（回应"GA 68000 怎么 lab 倒退到 63323"）

**不是退步，是三个口径**（校准报告 20260918_eval_core_calibration_r8top1.md）：

| 口径 | 数值 | 构成 |
|---|---|---|
| GA 锚 | 71415.6 | 7 卡组 × 60 局精评均值（含 speed_wisdom 70758 高分组） |
| 主力卡组单局 | 68000~70758 | speed_wisdom 70758 / spd2_gut0 68080 / speed 66884 |
| lab smoke | 63323 | **3 个初筛卡组**（speed 66884 + stamina 67129 + power_wisdom 56002）× 100 局均值，power_wisdom 是低分组拖了均值 |

手算验证：(66884+67129+56002)/3 = 63338 ≈ lab 实测 63323（差 15 = 局数回归噪声）。
**结论：lab 没有退步，是 smoke 初筛档刻意用 3 个便宜卡组控制单次评估成本。跨口径不可比。**
要跟 GA 锚同台，用 full 档（7 build × 100 局）——lab-smoke.yml 有这个档位。

---

## 一、已有的（不重加，防重复造轮子）

| 面 | 已有 | 出处/状态 |
|---|---|---|
| 参数面 | CMA-ES / TPE / SA(v2 修复中) / CEM / MAP-Elites / CVaR 适应度 | optimizers/ REGISTRY 5 个 + cvar_compare |
| 参数面 | GA（已收敛到 71415.6 天花板） | Rust 仓 ga-optimize，r8 收账 |
| 参数面 | **smoke/full 双档 = 天然 multi-fidelity** | evaluator.py SMOKE_SEEDS/FULL_SEEDS——但还缺"早期淘汰"调度（见二.3） |
| 决策面 | MCTS 框架 + flat_search 骨架（**叶子评估器空着**） | 上游 umaai-rs search/ |
| 决策面 | 手写规则（35+70 参数，GA 拧过） | policy.rs + genome |
| 决策面 | NN 评估器（EXP-001~009b，66734.8 冠军后归档） | uma-ramen-nn-lab |
| 数据 | 5.6 万局训练数据（754 维编码就绪） | teacher-collect |

## 二、新候选（按"预期收益 × 实现成本"排序）

### 第一梯队：参数面增量，一周内能进 REGISTRY 赛马

| # | 算法 | 接哪个面 | 大白话 | 为什么可能有用 | 成本 |
|---|---|---|---|---|---|
| 1 | **SPSA** | 参数面 | 每步只扰一次方向估梯度，2 次评估走一步，**不管多少维** | 45 维下预算效率之王：30 eval = 15 步真梯度信息；配 CRN（已有）对冲噪声 | 极低（30 行） |
| 2 | **ARS 增强随机搜索** | 参数面 | 随机方向试探，只记最好的几个方向的加权平均 | top-k 方向平均自带降噪，比 OpenAI-ES 省一半评估；DeepMind 复现过强基线 | 低（50 行） |
| 3 | **xNES** | 参数面 | CMA-ES 的信息几何表亲，直接搜分布参数 | 与 CMA-ES 形成"同族轻量对照"——如果 xNES 显著弱于 CMA-ES，说明协方差自适应是关键，这本身是结论 | 低（80 行） |
| 4 | **Powell / COBYLA** | 参数面 | scipy 自带的坐标方向法/线性插值信任域 | 零实现成本；45 维连续空间 Powell 可试；结果差也有"经典 DFO 打不过进化系"的档案价值 | 零（scipy） |
| 5 | **BORE** | 参数面 | 把 BO 的采集函数变成"好/坏二分类"（LightGBM） | TPE 的亲戚但用 GBDT，对 45 维混合空间比 optuna 默认更对症 | 中（150 行+lightgbm） |

**第一梯队赛马协议**：沿用 sa_compare 5 seed × 30 iter CRN 配对，一次 run 出全表。

**赛马结果（run 4，2026-10-18 收账，详见 playbook/notes/20261018_tier1_settle.md）**：
- ARS：mean 第 2（63438.3）+ 全场最稳（std 44.1）→ 晋级候补，10 seed × 60 iter 待裁决
- SPSA：显著垫底（-682.8, p=0.0049）→ 记档退出；mock 王（精确回中）真 bench 翻车，
  梯度估计路线与离散+噪声景观冲突，档案结论
- Powell：显著垫底（-601.0, p=0.0001）+ 确定性路径（std=0）→ 记档退出；
  "经典 DFO 打不过进化系"预判兑现
- 参数面六算法全 n.s.（除两负）→ 30 eval 预算下参数面边际收益挤干，
  重心转第二梯队调度（Hyperband）与第三梯队 GBDT 叶子评估器

### 第二梯队：调度与验收升级（不换算法，换实验管理学）

| # | 方法 | 接哪个面 | 大白话 | 增量 |
|---|---|---|---|---|
| 6 | **Successive Halving / Hyperband 化** | lab 调度 | smoke 档先海选（1 seed×30 局），活下来的才上 full 档精评 | 已有双档但没淘汰逻辑——加"半数淘汰"调度，同样算力多筛 3-5 倍候选 |
| 7 | **UCB/Thompson 预算分配** | lab 调度 | 哪个 (算法×马娘×卡组) 组合还没试够就多给谁跑 | 多优化器并行时自动平衡探索/收账，替代手动 dispatch |
| 8 | **Bradley-Terry 成对验收** | holdout | 不比绝对分（噪声大），比"同种子下谁高"的胜率 | holdout 验收从"均值差"升级"配对胜率"，统计功效更高 |
| 9 | **qEHVI / ParEGO 多目标 BO** | 参数面 | 同时优化 mean 和 CVaR，画 Pareto 前沿让用户选风险口味 | CVaR 已有单目标版，MOBO 是它 的自然升级；成本中高（botorch），先排队 |

### 第三梯队：决策面（真正的大头，但工作量大，另立战场）

| # | 方法 | 接哪个面 | 增量 |
|---|---|---|---|
| 10 | **GBDT 叶子评估器** | flat_search/MCTS 的空叶子 | 5.6 万局数据 + LightGBM = "看状态报分"，比 NN 推理快不用 GPU；这是把空转的搜索框架点亮的最便宜火柴（algo-selection 第二档老名单，仍空着） |
| 11 | **Expert Iteration 正式化** | MCTS↔学生闭环 | nn-lab EXP-009 自蒸馏的正式形态：MCTS 打谱→GBDT/NN 学生→更快的 MCTS→再打谱 |
| 12 | **NMCS/NRPA（嵌套搜索）** | 决策面备选 | 爬塔类比轮的核心候选：单玩家序列搜索，60 回合育成=爬塔；但先等 10 号点亮叶子再评估价值 |

### 不适用（有明确理由，写死挡掉，不再回头）

- **世界模型/Dreamer/MBRL 全家（PILCO/PETS/PlaNet）**：我们**有真模拟器**（一局几秒可大量重放），学世界模型是负收益
- **GAN/扩散/VAE 生成族**：黑盒评估下没有"生成新状态"的需求
- **GNN**：状态非天然图结构
- **博弈论/对抗搜索全家（CFR/PSRO/League）**：单 agent，无对手
- **控制论全家（PID/LQR/MPC 显式版）**：无显式动力学方程（MPC 的模拟器快进版=flat_search，已有）
- **因果发现（PC/NOTEARS/DML）**：数据是自家模拟器产的，因果结构已知
- **SAT/CP/ILP 全家全局版**：育成无可写约束方程；子问题（技能点分配）留第三档档案
- **模糊逻辑/ANFIS**：手写规则的连续化伪装，参数面已被 GA/CMA-ES 覆盖
- **NEAT/HyperNEAT/CPPN**：搜拓扑，我们的网络是固定 754 维编码
- **SINDy/Koopman/DMD 系统辨识**：模拟器白盒可读，不需要辨识
- **RLHF/DPO 偏好对齐**：奖励就是分数，定义不模糊（偏好学习只借 Bradley-Terry 做验收，见 #8）

## 三、跟用户推荐组合的对表

| 用户推荐 | 本仓落点 |
|---|---|
| 只看结果：CMA-ES/OpenAI-ES/ARS + 代理模型 + qEHVI + Hyperband | CMA-ES✓；ARS=#2；代理模型=#10 GBDT 叶子；qEHVI=#9；Hyperband=#6 |
| 能读内存：MDP/MCTS/DP 白箱 | 上游模拟器 + jidianbei 基底（10-16 建仓）就是这条线 |
| 找多流派：MAP-Elites/CMA-MAE | MAP-Elites✓；CMA-MAE 排队（等 MAP-Elites 有结论） |
| 真实育成成本高：多保真+迁移+元学习 | 双档已有雏形=#6；迁移/元学习暂缓（先把手头算法比完） |
| 比赛对抗：MCCFR/PSRO | 不适用（无对手） |
| 奖励难定义：RLHF/偏好BO | 奖励=分数不难；只借 Bradley-Terry=#8 |

## 四、执行序（建议）

1. **本周**：#1 SPSA + #2 ARS + #4 Powell 进 REGISTRY，sa_compare 协议一次排表（SA v2 同场收账，run 3 在跑）
2. **下周**：#6 Hyperband 化 + #8 Bradley-Terry 验收（改 lab 调度层，不动优化器）
3. **决策面另立项**：#10 GBDT 叶子评估器——接 flat_search 空叶，5.6 万局数据现成；这条线产出后 #11/#12 才有意义
4. xNES/BORE/#9 MOBO 视第一梯队结果补位
```prior
existing_registry = [cmaes, tpe, sa, cem, map_elites, cvar_fitness]
score_baseline_smoke_3build = 63323
score_baseline_full_7build = 71415.6_ga_anchor
tier1_new = [spsa, ars, xnes, powell, bore]
tier2_scheduling = [hyperband_sh, ucb_budget, bradley_terry_holdout, qehvi]
tier3_decision = [gbdt_leaf_evaluator, exit_loop, nmcs_nrpa]
rejected = [world_models, gan_diffusion, gnns, game_theory, control_theory, causal_discovery, sat_cp_ilp, fuzzy, neat_family, sysid, rlhf]
```
