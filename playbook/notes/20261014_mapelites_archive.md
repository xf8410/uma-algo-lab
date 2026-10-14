# MAP-Elites 实战（commit b339b0b + bug 修复 4e7ef23）

> 来源：commit b339b0b（feat: MAP-Elites 2026-10-08）+ commit 4e7ef23（fix: ndarray key 崩溃 2026-10-10）
> 日期：2026-10-14
> 优先级：high

## 设定
- 实验：5×5 archive grid，bh1=vital_rest, bh2=pt_tradeoff
- 算力：warmup=8 全探索，70% mutation / 30% explore（_n_asked > 8 后）
- mutation_rate=0.12, sigma_explore=0.18

## 关键观察
1. **关键 bug**：原 ask() 用 `np.random.choice(dict.keys())` 返回 ndarray 不能当 dict key
   - 修复：改 `self.rng.integers(len)` + `list(self.archive.keys())[idx]` 取 tuple
2. **sparse 阈值 = 25*0.3 = 7.5 cell** —— archive < 7.5 走 explore，> 7.5 才进 mutation
3. **行为分布不均**：vital_rest=0-2 区域几乎不填（基因值偏低 = "几乎不休息"）
   - 实际游戏里 vital_rest 不能太低（玩家会疲劳）→ 应把 bh1 floor 在 [0.3, 0.7]
4. **预算至少 50+ eval 才出好结果**（前 30 几乎全 explore）

## 经验/建议（硬约束）
- **vital_rest floor 0.3** —— 太低实际游戏会疲劳
- **pt_tradeoff 不加约束**（2 维已经够）
- **min_budget_for_archive = 30** —— 低于 30 eval 跑不出实际 best
- **mutation_rate 降到 0.10**（0.12 在 archive 填到 12 cell 后开始振荡）

```prior
vital_rest_floor = 0.3
vital_rest_ceiling = 0.7
min_budget_for_archive = 30
mutation_rate = 0.10
sigma_explore = 0.18
n_bins = 5
```

## 待办
- [ ] 把 vital_rest [0.3, 0.7] 做成 MAP-Elites 构造参数（center + clip）
- [ ] 跑一批 full 60 eval 看实际 best（当前 smoke 30 eval）
- [ ] archive dump 加 cell_visit_count 字段，看哪些 cell 是反复重访 vs 一次性
- [ ] 5×5 grid 可视化改进（加 cell coverage %）

## 关联
- [optimizers/map_elites_opt.py](../optimizers/map_elites_opt.py)
- [experiments/map_elites.py](../experiments/map_elites.py)
