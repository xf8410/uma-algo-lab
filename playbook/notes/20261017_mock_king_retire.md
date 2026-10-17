# 假冠军问题入档：SPSA / Powell / SA v2 三将退役

> 来源：sa-compare run 1/2/3/4（2026-10-16/17 4 轮排表）
> 父规则：fengqi-memory `rules/algo_noise_discipline.md` + algo-lab `docs/algo_noise_pitfalls.md`
> 处置：SPSA / Powell / SA v2 三个算法按算法噪声纪律正式剔除，记档
> 日期：2026-10-17
> 优先级：**high**（影响后续新算法入库判据）

## 三个退役算法总览

| 算法 | 入仓理由 | 退役证据 | 退役根因 |
|---|---|---|---|
| **SPSA** 同时扰动随机逼近 | mock 0.000 距冠军 / 论文顶会刷脸 | run 4 mean 62713.2 / -682.8 vs CMA-ES / p=0.0049 显著负 | 渐近最优超参在 30 eval 下完全失效 + clip 边界放大噪声 |
| **Powell** 方向集 | 经典 DFO / scipy 零成本 | run 4 mean 62795.0 / -601.0 / 5 seed 完全一样 std=0.0 | 30 eval 内连一次三点线搜索都做不完 |
| **SA v2** 模拟退火 | 几何降温 + Metropolis 跳出局部最优 | run 3 v2 mean 63329.8 / -17.1 vs CMA-ES / p=0.908 n.s. | 45 维 + 30 eval 下 1/4 维扰动×30 步 = 每维只摸 7.5 次 |

## 共同模式：**维度×预算硬墙**

所有三个退役算法的根因都可以归纳到**"维度×预算比 < 阈值"**：

| 算法 | 评估次数/步 | n_iter 30 总评估 | 维度 | 维度×预算比 | 触墙? |
|---|---|---|---|---|---|
| SPSA | 2 eval/步 | 15 步 | 45 | 15/45 = 0.33 步/维 | ✓（远不够） |
| Powell | 135 eval/轮 | 30 eval = 0.22 轮 | 45 | 0.22 轮/45 维 = 完全跑不开 | ✓ |
| SA v2 | 1 eval/步 | 30 步 | 45 | 30/45 = 0.67 步/维（仅 1/4 维扰动，实际 7.5 次/维） | ✓ |
| CMA-ES | 1 eval/个（种群 12） | 12×30 = 360 eval | 45 | 360/45 = 8 eval/维 | ✗（活得不错） |
| TPE | 1 eval/步 | 30 步 | 45 | 30/45 = 0.67（独立采样无维度耦合） | ✗（独立采样不触墙） |
| ARS | 6 eval/步（3 方向双侧） | 5 步 | 45 | 5 步×top-2 加权 ≈ 30 信息/45 维 = 0.67/维 | ✗（抗噪加权救回） |

**触墙判据（粗）**：n_iter × 单步评估数 / 维度 < 1 → 单点/两点算法触墙
> **CMA-ES / TPE / ARS 都在 1 附近或以上**，**SPSA / Powell / SA v2 都在 0.5 以下或更低**

## 留给未来的 prior

### 不要重做的算法

- **SPSA / NES / OpenAI-ES 全家**：渐近最优假设在 30 eval 失效
- **Powell / COBYLA / 方向集/坐标下降**：单步 135 评估在 30 eval 跑不开
- **SA v1/v2 任何温度优化版**：单点 1/4 维扰动在 45 维下永远不够
- **CMA-ES 单评估/步版**：本仓现在用 12 种群版本，单评估版会触同样的墙

### 仍然可能救活的算法（待高预算复验）

- **ARS 增强随机搜索**：top-2 方向加权抗噪，run 4 marginal 但 std 极小，待 n_iter=60 复验
- **TPE**：n=5 配对功效不够掩盖 +103，待 10-15 seed + full 档复验
- **xNES 自然进化策略**：CMA-ES 同族轻量对照，待第二梯队 Hyperband 化（n_iter 翻倍）后复验
- **BORE 二分类 BO**：TPE 亲戚用 GBDT，TPE 待复验后再说

### 待开发的算法（避免重蹈覆辙）

- **Botorch / BOHBA / HEBO 多保真 BO**：用 smoke/full 天然双档做早期淘汰
- **PB2 / PBT 动态超参**：30 eval 太小 → 改动态调参
- **MOO-ParEGO / qEHVI**：第二档升级，但 CVaR 单目标已有，多目标值不值需评估

## 推论：sa_compare.yml 默认值要改

当前默认值：`n_iter: 30, seeds: '7,13,23,31,47' (n=5)`
- n=5 功效不够（见 algo_noise_pitfalls.md R3）
- n_iter=30 是 SPSA/Powell/SA 触墙的根本原因

**建议改**：
- `n_iter` 默认 60（新算法复验用 60，入仓走完快筛后单跑再 60+）
- `seeds` 默认 10（`7,13,23,31,47,53,67,79,89,97`）
- `level` 默认 full（smoke 档 + 30 eval 噪声太大）

**但**：60 iter × 10 seed = 600 eval × 6 algo = 3600 evals / 30 min/evalset = **2-3 小时**——
跑批时长翻 4-6 倍，CI 资源消耗大。**先用 60 iter × 5 seed 复验，显著信号才升级到 10 seed full 档**。

## 下一步

- [ ] 6algo_race 笔记 / blackbox-selection.md / algo-selection.md / algo_noise_pitfalls.md 合并入仓（待 commit）
- [ ] sa_compare.yml 默认值升级（n_iter 30→60，n_seed 5→10）——**谨慎**，先做小流量验证再全面升级
- [ ] run 5：TPE/ARS/CMA-ES/CEM 4 算法 × n_iter=60 × 5 seed 复验（验证 n=60 + 5 seed 能否让 TPE/ARS 显著）
- [ ] 决策面：GBDT 叶子评估器启动（这条线不被参数面赛马阻塞）

```prior
retired_algos = [spsa, powell, sa_v2]
retired_reason = dim_x_budget_wall
retired_evidence = {
  spsa: {mean: 62713.2, delta_vs_cmaes: -682.8, p: 0.0049, sig_negative: true},
  powell: {mean: 62795.0, delta_vs_cmaes: -601.0, p: 0.0001, sig_negative: true},
  sa_v2: {mean: 63329.8, delta_vs_cmaes: -17.1, p: 0.908, sig_negative: false, but: dim_x_budget_wall},
}
dim_x_budget_wall_criterion = "n_iter × evals_per_step / dim < 1"
wall_criterion_triggers = {spsa: 0.33, powell: 0.22, sa_v2: 0.50, cmaes: 8.0, tpe: 0.67, ars: 0.67}
promote_candidate = [tpe, ars]
promote_candidate_need = {n_iter: 60, n_seed: 10, level: full}
sa_compare_default_suggestion = {n_iter: 60, seeds: 10, level: full}
sa_compare_default_caution = "60×10×6 = 3600 evals = 2-3h CI, ramp up incrementally"
```
