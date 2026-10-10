# AgentGarten 借鉴分析（algo-lab 视角）

> 调研时间：2026-10-10  
> 信息源：[MirroS Blog](https://mirros.ai/blog/worlds-for-evolving-agents) · [技术报告 arXiv:2610.12374](https://arxiv.org/pdf/2610.12374) · [GitHub](https://github.com/MirroS-Lab/AgentGarten)  
> 适用读者：回到 algo-lab 仓想了解"为什么加 Playbook 机制 / fail_mode 诊断"的人

---

## TL;DR

AgentGarten（MirroS 团队 2026-10-08 开源）里**最值得 algo-lab 抄的，是它的"实验手册（Playbook）"机制**。每代智能体把自己学到的写成 markdown 笔记传给下一代。这对算力小、靠经验迭代的 algo-lab 特别合适，落地成本几乎为零。

神经渲染器、Adversarial Forcing 训练法、H100 GPU 流水线这些，跟赛马娘没关系，不抄。

---

## AgentGarten 是啥（一段话版）

MirroS 团队 2026-10-08 开源了 AgentGarten——给 AI 智能体当练习场用。它把"代码定义的世界"和"实时神经渲染的画面"接起来，让智能体能像玩 Minecraft 一样一边看画面一边操作。

最有冲击力的对比：2019 年 OpenAI 拿强化学习玩捉迷藏，AI 学会搭掩体要 **2500 万局**，借坡道翻墙要 **1 亿局**。MirroS 用预训练智能体 + 写"实验手册"，**4 轮**学会搭掩体，**10 轮**学会翻墙。

---

## 四大核心创新（按对 algo-lab 的相关性排序）

### 🟢 1. 实验手册（Playbook）——直接可抄

机制很朴素：
- 每轮结束，智能体**自己写笔记**：试了啥、看到啥、哪条判断存疑、下次该验啥
- 笔记按"一条经验一个文件"存档
- 下一轮智能体一启动，就拿到前面所有轮次的手册当"先验知识"

**对 algo-lab 的价值**：
- algo-lab 现在做 GA / CMA-ES / TPE 优化 genome，结果散在 `.npz`、`.json`、报告里
- 下一代基本从零开始（除非手工 seed injection）
- Playbook 机制 = 把"哪类 genome 提升大、哪类参数会崩"结构化沉淀下来

**落地形式**：
- 短期：人手写 `algo-lab/docs/playbook.md`（核心结论 + 失败模式 + 跨优化器经验）
- 中期：experiment runner 跑完自动 append 一条
- 长期：warm start 注入下一代优化器

跟现有 `--seed-genome` 互补不冲突——seed 复用 genome，Playbook 复用"这 genome 在什么条件下有效"。

### 🟢 2. 失败模式诊断（fail_mode breakdown）——直接可抄

AgentGarten 的 playbook 里专门有一段叫"防坑指南"——记录智能体容易在哪翻车、下次怎么避免。

algo-lab 当前只报"平均分 / 失败率"，**没报"哪一局死在哪一步"**——这是信息损失。

**改进**：`lab/evaluator.py` 加 `evaluate_with_diagnostics()`，返回 fail_mode 分布：
- `vital_zero_step_45_60`：45-60 回合体力归零占比
- `skill_missed_3_5`：3-5 个关键技能没触发占比
- `race_position_drop`：比赛中掉到第 6 之后占比
- `pt_underflow`：PT 值低于阈值占比

拿到 fail_mode 后：
- 下一代 mutation **针对 fail_mode 反向扰动**
- Playbook 自动追加"这批跑下来 XX fail_mode 比例最高"作为一条经验
- 用户看报告能直接看"马容易在哪一关翻车"

### 🟡 3. 任务世界代码化——思路可学，工程不直接抄

AgentGarten 最大工程创新："代码世界 + 共享渲染器"——加新任务 = 写新代码，不用动渲染器。

**algo-lab 现状**：上游 umaai-rs 黑盒零修改（红线），改不了。

**借鉴角度**：把**实验编排**（不是模拟器）代码化——`tasks/cvar_vs_mean.yaml`、`tasks/map_elites_archive.yaml`……runner 读 task 文件就知道要跑啥、怎么跑、结果怎么解读。

**但**：algo-lab 脚本现在 5 个不到，谈"代码化"太早。先记下这个方向，等实验超过 10 种再统一。

### 🔴 4. Adversarial Forcing / 神经渲染器 / 流式推理——完全不抄

这些是针对视频世界模型的训练/推理技巧：

| 技术 | 用途 | algo-lab 用得上吗 |
|------|------|------------------|
| Adversarial Forcing（exact replay + 真实视频对抗信号） | 视频扩散模型蒸馏 | ❌ |
| 神经渲染器（30fps H100 480p） | 几何条件画成画面 | ❌ 端侧渲染 |
| 流式 KV cache + Triton 融合算子 | 视频模型推理加速 | ❌ |
| Cosmos3-Nano + Wan 2.2 VAE | 视频生成基础模型 | ❌ |

**一句话：AgentGarten 论文 80% 是给做视频 AI 的人看的。赛马娘项目是给做数值黑盒优化的人看的。能借鉴的就 20%——Playbook + fail_mode。**

---

## 跟 algo-lab 现状的对照

| 维度 | AgentGarten | algo-lab | 差距 / 行动 |
|------|------------|----------|------------|
| **环境** | 代码世界 + 共享渲染器 | 上游黑盒（零修改） | 模拟器改不了，实验编排可声明式化 |
| **决策** | 预训练大模型写 Python 动作 | 70 维 genome + MCTS/GA | 路线不同，借鉴面窄 |
| **优化** | 4-10 轮 playbook 传递 | 50+ 代 GA + 多优化器赛马 | algo-lab 算力允许更系统化 |
| **经验沉淀** | 智能体自写 playbook | 散在报告/cache npz | **🟢 缺统一 playbook 机制** |
| **失败诊断** | 智能体自报"哪关翻车" | 只报 fail_rate 不知怎么死的 | **🟢 缺 fail_mode breakdown** |
| **评估器复用** | 共享渲染器 | cache by (hash, level) | 🟡 已不错，加 fail_mode 即可 |
| **跨代迁移** | 强（playbook 强制传代） | 弱（要人手 seed injection） | **🟢 核心改进点** |

---

## 落地下一步

### 这周（成本低价值高）

1. **Playbook 草稿**：`docs/playbook.md` 人手写一版  
   - 段 1：核心结论（vital_rest / pt_tradeoff 最佳区间、mutation_rate 经验值、sigma_explore 经验值）
   - 段 2：失败模式清单（哪类 fail_mode 常见、对应怎么调）
   - 段 3：跨优化器经验（哪个优化器在什么预算下性价比最高）

2. **fail_mode breakdown 字段**：`lab/evaluator.py` 加 `evaluate_with_diagnostics()`  
   - 复用现有 `_run_bench` + cache
   - 从 bench 的 stdout/stderr 解析关键事件
   - 半天改 + 半天测

### 下周（依赖前两条做完后）

3. **Playbook 自动追加**：runner 跑完自动 append 一条（跨 run 显著变化时触发）
4. **Warm start 注入**：下一代优化器读取 `playbook.md` 当初始 center / sigma / 搜索偏好

### 不做

- ❌ 神经渲染器 / Adversarial Forcing / 流式推理——跟数值黑盒无关

---

## 跟其他调研的关系

之前《AI 涌现与智能爆炸论文调研》结论里有一条"用 PBT（Population Based Training）优化超参"——PBT 核心是"多代超参 + 经验传递"。

AgentGarten 的 Playbook 机制在精神上跟 PBT **完全一致**，只是 PBT 在神经网络训练里、Playbook 在智能体推理里。algo-lab 的 Playbook 落地后，**自然会把 algo-lab 往 PBT 路线推**——方向对。

---

## 参考资料

- [MirroS Blog](https://mirros.ai/blog/worlds-for-evolving-agents) — 写作最清晰，70% 内容来自这里
- [技术报告 arXiv:2610.12374](https://arxiv.org/pdf/2610.12374) — 细节更多
- [GitHub: MirroS-Lab/AgentGarten](https://github.com/MirroS-Lab/AgentGarten) — 目前只发了渲染器训练代码
- [OpenAI 捉迷藏原论文](https://arxiv.org/abs/1909.07528) — 2500 万局 / 1 亿局对比基准
