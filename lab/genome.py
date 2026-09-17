# -*- coding: utf-8 -*-
"""基因组三形态互转：向量 x <-> genome dict <-> TOML 文本。

TOML 输出与 Rust 端 parse_override_toml 严格对偶：
- 只写被覆盖的键（None = 保留 preset，不写行）
- 数组三元组 null 槽位写 null
- 顶层裸键（无段头），Rust 端跳过段头行
向量统一工作在 [0,1]^d 归一化空间，decode 时线性映射到各字段值域。
"""
from typing import Dict, List, Optional, Tuple

from .space import ALL_FIELDS, Field


def free_layout() -> List[Tuple[Field, Optional[int]]]:
    """(字段, 数组槽位序号 or None) 的扁平布局，与 space.free_dim() 一致"""
    layout = []
    for f in ALL_FIELDS:
        if f.is_fixed():
            continue
        if f.typ.startswith("arr"):
            for i, slot in enumerate(f.lo):
                if slot is not None:
                    layout.append((f, i))
        else:
            layout.append((f, None))
    return layout


def decode(x: List[float]) -> Dict:
    """[0,1]^d 向量 -> genome dict（未覆盖字段不出现在 dict）"""
    assert len(x) == len(free_layout()), f"维度不匹配: {len(x)} vs {len(free_layout())}"
    genome: Dict = {}
    for val, (f, slot) in zip(x, free_layout()):
        val = min(1.0, max(0.0, val))
        if f.typ.startswith("arr"):
            lo, hi = f.lo[slot]
            is_int = f.typ in ("arri32", "arrusize")
            v = lo + (hi - lo) * val
            v = int(round(v)) if is_int else float(v)
            arr = list(genome.get(f.name, [None, None, None]))
            arr[slot] = v
            genome[f.name] = arr
        elif f.typ == "bool":
            genome[f.name] = bool(val >= 0.5)
        elif f.typ in ("i32", "u32"):
            genome[f.name] = int(round(f.lo + (f.hi - f.lo) * val))
        else:  # f32
            genome[f.name] = float(f.lo + (f.hi - f.lo) * val)
    return genome


def center_vector() -> List[float]:
    """center 基因组对应的初始向量（全部取值域中点附近的确定映射）"""
    from .space import free_dim
    return [0.5] * free_dim()


def genome_to_toml(genome: Dict) -> str:
    """genome dict -> TOML 文本（Rust parse_override_toml 可直接解析）"""
    lines = [
        "# algo-lab 覆盖层（自动生成；未列出的键 = 保留 RecommendedRamenTrainer preset）",
        "",
    ]
    for name, val in genome.items():
        if val is None:
            continue
        if isinstance(val, list):
            parts = []
            for p in val:
                if p is None:
                    parts.append("null")
                elif isinstance(p, bool):
                    parts.append("true" if p else "false")
                elif isinstance(p, float) and p == int(p) and abs(p) < 1e9:
                    parts.append(str(p))
                else:
                    parts.append(str(p))
            lines.append(f"{name} = [{', '.join(parts)}]")
        elif isinstance(val, bool):
            lines.append(f"{name} = {'true' if val else 'false'}")
        elif isinstance(val, float):
            lines.append(f"{name} = {val!r}")
        else:
            lines.append(f"{name} = {val}")
    lines.append("")
    return "\n".join(lines)


def genome_hash(genome: Dict) -> str:
    """与 Rust 端同语义的稳定哈希（dict 序列化后 sha1 前 16 位）"""
    import hashlib
    import json
    blob = json.dumps(genome, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
