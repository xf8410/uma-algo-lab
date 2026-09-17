# -*- coding: utf-8 -*-
"""基因组三形态互转：向量 x <-> genome dict <-> TOML 文本。

TOML 输出与 Rust 端 parse_override_toml 严格对偶：
- 只写被覆盖的键（None = 保留 preset，不写行）
- 数组三元组 null 槽位写 null
- 顶层裸键（无段头），Rust 端跳过段头行
向量统一工作在 [0,1]^d 归一化空间，decode 时线性映射到各字段值域。
"""
import re
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


def apply_fixed(genome: Dict) -> Dict:
    """把 GA 定型的固定 bool 字段合入 genome（覆盖层完整性对齐冠军 TOML）。

    固定 bool 不占向量维度，但 preset 默认值与 GA 定型值有 4 处不一致
    （如 friend_outing_replaces_rest preset=false / 定型=true），缺失会让
    warm-start 起点偏离 71415.6 基准。空间定义值即定型值。
    """
    for f in ALL_FIELDS:
        if f.typ == "bool" and f.lo is not None and f.name not in genome:
            genome[f.name] = bool(f.lo)
    return genome


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
            v = int(round(v)) if is_int else round(float(v), 6)
            arr = list(genome.get(f.name, [None, None, None]))
            arr[slot] = v
            genome[f.name] = arr
        elif f.typ == "bool":
            genome[f.name] = bool(val >= 0.5)
        elif f.typ in ("i32", "u32"):
            genome[f.name] = int(round(f.lo + (f.hi - f.lo) * val))
        else:  # f32
            genome[f.name] = round(float(f.lo + (f.hi - f.lo) * val), 6)
    return apply_fixed(genome)


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
                elif isinstance(p, float):
                    parts.append(str(round(p, 6)))
                else:  # int 槽位
                    parts.append(str(p))
            lines.append(f"{name} = [{', '.join(parts)}]")
        elif isinstance(val, bool):
            lines.append(f"{name} = {'true' if val else 'false'}")
        elif isinstance(val, float):
            # f32 有效位 ~7，round 6 去浮点尾巴（85.11449999999999 -> 85.11445）
            lines.append(f"{name} = {round(val, 6)!r}")
        else:
            lines.append(f"{name} = {val}")
    lines.append("")
    return "\n".join(lines)


def toml_to_value(toml_str: str) -> Dict:
    """TOML 覆盖层文本 -> genome dict（方言感知：接受数组 null 槽位）。

    与 Rust parse_override_toml 同语义：只认 `key = value` 行，段头跳过，
    null = 槽位不覆盖。缺失键不出现在 dict。
    """
    genome: Dict = {}
    for raw in toml_str.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("["):
            continue
        if "=" not in line:
            continue
        name, _, rhs = line.partition("=")
        name, rhs = name.strip(), rhs.strip()

        def _atom(tok: str):
            tok = tok.strip()
            if tok == "null":
                return None
            if tok == "true":
                return True
            if tok == "false":
                return False
            if re.fullmatch(r"-?\d+", tok):
                return int(tok)
            return float(tok)

        if rhs.startswith("[") and rhs.endswith("]"):
            inner = rhs[1:-1].strip()
            genome[name] = [] if not inner else [_atom(t) for t in inner.split(",")]
        else:
            genome[name] = _atom(rhs)
    return genome


def toml_to_vector(toml_str: str) -> List[float]:
    """TOML 覆盖层 -> [0,1]^d 向量（优化器 warm-start 起点）。

    TOML 未写的键取 0.5（值域中点近似）；null 槽位不占向量维度
    （free_layout 只收非 None 槽位，与 decode 严格对偶）。
    """
    genome = toml_to_value(toml_str)
    x: List[float] = []
    for f, slot in free_layout():
        if f.typ.startswith("arr"):
            arr = genome.get(f.name)
            v = arr[slot] if isinstance(arr, list) and slot < len(arr) else None
            if v is None:
                x.append(0.5)
            else:
                lo, hi = f.lo[slot]
                x.append(min(1.0, max(0.0, (float(v) - lo) / (hi - lo))))
        elif f.typ == "bool":
            x.append(1.0 if genome.get(f.name) else 0.0)
        else:
            if f.name not in genome:
                x.append(0.5)
            else:
                v = float(genome[f.name])
                x.append(min(1.0, max(0.0, (v - f.lo) / (f.hi - f.lo))))
    return x


def genome_hash(genome: Dict) -> str:
    """与 Rust 端同语义的稳定哈希（dict 序列化后 sha1 前 16 位）"""
    import hashlib
    import json
    blob = json.dumps(genome, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
