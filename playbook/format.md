# Playbook 笔记格式

> 一条笔记 = 一个 .md 文件 = 一次实验/观察的完整复盘。
> 命名：`YYYYMMDD_<scope>_<tag>.md`（例：`20261014_cmaes_n60.md`）。

## 模板

```markdown
# <一句话标题>

> 来源：<commit hash 或 PR 链接 或 手动观察>
> 日期：YYYY-MM-DD
> 优先级：high | medium | low

## 设定
- 实验：<优化器 / 实验名 / 关键参数>
- 算力：<eval 数 / 时间 / 局数>
- 数据：<哪个仓的哪个 commit / 哪个数据快照>

## 关键观察
1. <一句话观察 1>
2. <一句话观察 2>
3. <一句话观察 3>

## 经验/建议
- <可执行的建议 1，例：mutation_rate 调到 0.10>
- <可执行的建议 2>
- <硬约束，例：vital_rest floor 在 0.3>

## 待办
- [ ] <下一步要验证/实施的项>
- [ ] ...

## 关联
- <链接 1>
- <链接 2>
```

## 字段含义

| 字段 | 作用 | 解析影响 |
|---|---|---|
| `优先级` | high/medium/low | `parse_to_prior()` 只采纳 high + medium |
| `经验/建议` | 一行一个 `-` | 自动 parse 成 `prior` dict（key=value） |
| `硬约束` | 经验里包含"必须/不能/≥/≤"的句子 | `parse_to_prior()` 标为 bound |

## parse_to_prior 规则（v1）

- 读所有 `playbook/notes/*.md`
- 优先级 < medium 跳过
- 经验条目按行 parse：
  - `mutation_rate 调到 0.10` → `prior["mutation_rate"] = 0.10`
  - `sigma 默认 0.18` → `prior["sigma"] = 0.18`
  - `vital_rest 不能低于 0.3` → `prior["vital_rest_floor"] = 0.3`
  - `archive grid 至少 30 eval 才填满` → `prior["min_budget_for_archive"] = 30`
- 重复 key 后写的覆盖先写的（按文件名时间序）
- 解析失败的不报错，记录到 stderr

## 何时不写
- 一次跑批没新发现 → 不写（避免噪声）
- 发现但不确定 → 写 `priority: low`，注明"待验证"
- 跟现有笔记矛盾 → 写新笔记时引用旧笔记，正面回应

## 维护约束
- 旧笔记不删只追加（跟 fengqi-memory 仓 README 规约对齐："只许追加/合并，禁止裸删除；要删先 archive/"）
- archive/ 子目录放过期笔记（被新笔记覆盖的旧结论）
