# Playbook 机制（algo-lab 第四刀）

> **核心理念**：每跑完一批实验，把"什么有效、什么没用、什么意外"提炼成 markdown 笔记存档；
> 下一代优化器启动时直接当先验读，把"踩过的坑"和"摸到的甜区"代代传下去。
>
> **精神同源**：MirroS 团队 AgentGarten 项目（docs/agentgarten_references.md）的"Playbook"机制——
> 智能体每轮结束后自我复盘写笔记，下一代拿到 task + 历史 playbook 启动。"4 轮学会搭掩体"vs 传统 RL 2500 万局，
> 关键不是算力大，是代际经验传递。
>
> **对 algo-lab 的具体化**：把"代际经验"沉淀为 `playbook/notes/*.md`，
> 每代优化器（CMA-ES / TPE / SA / CEM / MAP-Elites）启动时 `lab.playbook_io.read_notes()`
> 读全部笔记 → `parse_to_prior()` 提取 mutation_rate / center / sigma / bounds 调整建议 → 注入到优化器构造参数。

## 目录

```
playbook/
├── README.md          ← 本文件（机制总览）
├── format.md          ← 一条笔记的标准格式
├── notes/             ← 实战笔记
│   ├── 20261014_cmaes_n60.md
│   ├── 20261014_cvar_alpha02.md
│   ├── 20261014_mapelites_archive.md
│   └── ...            ← 每跑一批实验，手动或自动追加一条
```

## 使用方式

### 读（warm-start）
```python
from lab.playbook_io import read_notes, parse_to_prior

notes = read_notes(playbook_dir="playbook/notes")
prior = parse_to_prior(notes)
# prior = {"mutation_rate": 0.10, "sigma_explore": 0.15, "vital_rest_floor": 0.3, ...}

opt = CmaesOptimizer(dim=45, seed=seed, center=center,
                      mutation_rate=prior.get("mutation_rate", 0.12),
                      sigma=prior.get("sigma_explore", 0.18))
```

### 写（manual）
跑完实验后用 `lab.playbook_io.append_note(...)` 追加新笔记到 `playbook/notes/`。

### 写（auto，后续第五刀）
- `experiments/compare.py` / `cvar_compare.py` / `map_elites.py` 跑完 → 自动提取关键观察
- 写新 `playbook/notes/<date>_<exp>_<tag>.md`

## 笔记格式

见 [`format.md`](format.md)。

## 优先级机制

每条笔记有 `priority` 字段（high / medium / low）：
- high：必须采纳的硬约束（如"vital_rest 不能低于 0.3"）
- medium：建议参数（如"mutation_rate 用 0.10"）
- low：观察（如"CVaR 跟 mean 排名不一致的 genome 占 30%"）

`parse_to_prior()` 默认只采纳 high + medium 优先级。

## 落地节奏

- **第四刀（本次）**：手写 3 条初始笔记（CVaR / MAP-Elites / AgentGarten）+ 演示 warm-start
- **第五刀（后续）**：实验 runner 跑完自动追加笔记（不靠人写）
- **第六刀（后续）**：fail_mode 诊断 → 笔记自动生成
- **第七刀（后续）**：warm-start 实测对比（"读 prior 启动"vs"裸启动"哪个收敛快）

## 关联文档
- [docs/agentgarten_references.md](../docs/agentgarten_references.md) — 借鉴源头
- [docs/algo-selection.md](../docs/algo-selection.md) — 算法筛选四档
- [experiments/playbook_demo.py](../experiments/playbook_demo.py) — 演示

## 维护人
- 风起（按主人算法噪声纪律 + 仓 README 规约）
- 2026-10-14 落地
