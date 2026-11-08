# 第一梯队收账：SPSA/ARS/Powell 真 bench 首秀（run 4）

> 来源：sa-compare run 4（2026-10-17 dispatch，6 算法 × 5 seed × 30 iter CRN 配对）
> 日期：2026-10-18
> 优先级：high（ARS 晋级候补；SPSA/Powell 记档退出）
> 注：本笔记与 dbb8fca 的 20261017_6algo_race.md 独立成文（那个是用户侧视角），
> 合流后如内容重叠以两份都保留为原则（各自的 CRN 数据切片可能不同）。

## 排表（mean best，Δ vs cmaes 基准）

| 排名 | 算法 | mean | std | range | Δ | 显著性 |
|---|---|---|---|---|---|---|
| 1 | tpe | 63499.0 | 174.4 | [63306, 63644] | +103.0 | n.s. |
| 2 | **ars（新）** | **63438.3** | **44.1** | [63388, 63503] | +42.2 | n.s. |
| 3 | cem | 63435.8 | 83.1 | [63320, 63522] | +39.8 | n.s. |
| 4 | cmaes | 63396.0 | 85.1 | [63273, 63475] | 基准 | — |
| 5 | powell（新） | 62795.0 | 0.0 | [62795, 62795] | -601.0 | **✓ sig 负** (p=0.0001) |
| 6 | spsa（新） | 62713.2 | 303.0 | [62578, 63255] | -682.8 | **✓ sig 负** (p=0.0049) |

## 三条判读

### 1. ARS：mean 第 2 + 全场最稳，晋级候补
- std 44.1 是六个算法里最低（tpe 174 / cem 83 / cmaes 85）——top-k 方向平均的
  降噪在真 bench 上兑现了（mock 里它 0.133-0.233 不起眼，真场景稳才是真本事）
- range 下界 63388 全场最高：最差 seed 也不掉下 63388（cmaes 最差 63273）
- Δ+42.2 不显著（p=0.39），量级 < 150 不够 promote；但"下界最高+std 最低"
  值得 **promote_mild 观察位**：后续跑 10 seed × 60 iter 加预算裁决

### 2. SPSA：mock 王 → 真垫底，"mock 过拟合"教训入库
- mock（光滑二次碗+正弦）：3 seed 全部精确回中冠军（距 0.000）
- 真 bench（离散+噪声+clip）：mean 62713 **显著负于基准**（-682.8, p=0.0049）
- 归因：真景观里 clip 边界 + 整数 round 让 ±1 扰动的有效位移远小于名义值；
  a=0.08 步长在 63000 分数尺度上一步跳出盆地；噪声 ±50 级让 2 评估梯度
  估计的 SNR 崩掉
- **档案结论：梯度估计路线（一次扰动估全维）在这个问题上不可行**——
  不是 SPSA 实现错，是问题性质（离散+噪声）与连续光滑假设冲突

### 3. Powell：std=0.0 揭示确定性 + 噪声下贪心下滑
- 5 seed 全部 62795.0 同分：Powell 方向集固定（坐标轴），probe/plus/minus
  序列不依赖 seed——rng 没实际参与路径
- mean 62795 < 起点冠军 63323：三点线搜索在噪声景观下方向判断错误，
  一路贪心走低 528 分
- **档案结论：经典 DFO 方向集法在该问题打不过进化系**（blackbox-selection
  #4 的预判兑现，有数据支撑了）

## 对 blackbox-selection.md 执行序的更新

- #1 SPSA：记档退出（负结果保留）
- #2 ARS：**晋级候补**，下一步 10 seed × 60 iter 裁决 promote
- #4 Powell：记档退出（预判兑现）
- #3 xNES：维持排队——SPSA 负结果不连带 xNES（它是分布参数搜索不是梯度估计），
  但优先级降（第一梯队三将两负一平，参数面边际收益证据继续走低）
- **重心转向第二/三梯队的信号更强了**：参数面六算法 5 seed 全 n.s.（除负结果），
  说明 30 eval 预算下参数面提升空间基本挤干——Hyperband 调度（#6）和
  GBDT 叶子评估器（#10）是下一条提分线

```prior
run4_rank = [tpe, ars, cem, cmaes, powell, spsa]
ars_mean = 63438.3
ars_std = 44.1
ars_range_floor = 63388
spsa_delta = -682.8
powell_delta = -601.0
verdict_spsa = retired
verdict_powell = retired
verdict_ars = promote_mild_candidate
next_step = ars_10seed_60iter + tier2_hyperband + tier3_gbdt_leaf
```
