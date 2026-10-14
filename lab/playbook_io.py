"""Playbook I/O：读 / 解析 / 写入。

约定（见 playbook/format.md）：
- 每条笔记 = 一个 .md 文件，命名 YYYYMMDD_<scope>_<tag>.md
- 优先级由 frontmatter 或正文首行 "优先级：high/medium/low" 给出
- 解析：从 ```prior ... ``` 代码块读 key = value（一行一对）
- 后写入的覆盖先写入的（按文件名 YYYYMMDD 顺序）

约定 v1：只解析"```prior ... ```"代码块，不做散文 NER（避免误读）。
散文观察仍保留给人看，但 prior 注入只取代码块。
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import Any

PRIOR_BLOCK = re.compile(r"```prior\s*\n(.*?)```", re.DOTALL)
PRIOR_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
PRIORITY_RE = re.compile(r"优先级[：:]\s*(high|medium|low)", re.IGNORECASE)
DATE_PREFIX = re.compile(r"^(\d{8})_")


def _priority_of(text: str) -> str:
    m = PRIORITY_RE.search(text)
    return (m.group(1).lower() if m else "medium")


def list_notes(playbook_dir: str | Path) -> list[Path]:
    """列出所有 notes/*.md，按日期前缀排序（旧→新）。"""
    p = Path(playbook_dir)
    if not p.exists():
        return []
    notes = sorted(p.glob("*.md"))
    return notes


def read_notes(
    playbook_dir: str | Path,
    min_priority: str = "medium",
) -> list[dict[str, Any]]:
    """读所有 notes（按 min_priority 过滤），返回 [{path, priority, date, text, prior}, ...]。

    prior 是从 ```prior``` 块解析的 dict；priority 是 high/medium/low。
    """
    prio_order = {"low": 0, "medium": 1, "high": 2}
    cutoff = prio_order.get(min_priority, 1)
    out = []
    for note_path in list_notes(playbook_dir):
        text = note_path.read_text(encoding="utf-8")
        prio = _priority_of(text)
        if prio_order.get(prio, 1) < cutoff:
            continue
        m = DATE_PREFIX.match(note_path.name)
        date = m.group(1) if m else ""
        prior = parse_prior_block(text)
        out.append({"path": note_path, "priority": prio, "date": date, "text": text, "prior": prior})
    return out


def parse_prior_block(text: str) -> dict[str, str]:
    """从 markdown 正文里抓 ```prior ... ``` 块，按行解析 key = value。"""
    m = PRIOR_BLOCK.search(text)
    if not m:
        return {}
    block = m.group(1)
    out: dict[str, str] = {}
    for line in block.splitlines():
        m2 = PRIOR_LINE.match(line)
        if not m2:
            continue
        key, value = m2.group(1), m2.group(2)
        # 去掉尾部注释
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        out[key] = value
    return out


def parse_to_prior(
    notes: list[dict[str, Any]],
    coerce: bool = True,
) -> dict[str, Any]:
    """把多笔记的 prior 合并成单 dict：后写笔记的 key 覆盖先写的（同 key 提醒）。

    coerce=True：尝试把字符串转 int/float/bool/list。
    """
    out: dict[str, Any] = {}
    seen: dict[str, str] = {}  # key -> 来源 note filename
    for n in notes:
        for k, v in n["prior"].items():
            if k in seen and seen[k] != n["path"].name:
                # 同 key 覆盖，提示
                print(
                    f"[playbook] prior 冲突: {k!r} 由 {seen[k]} 覆盖为 {n['path'].name}",
                    file=sys.stderr,
                )
            out[k] = _coerce(v) if coerce else v
            seen[k] = n["path"].name
    return out


def _coerce(value: str) -> Any:
    """把字符串尝试转成 int / float / bool / list / 原样字符串。"""
    s = value.strip()
    # bool
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    # int
    try:
        return int(s)
    except ValueError:
        pass
    # float
    try:
        return float(s)
    except ValueError:
        pass
    # list: [a, b, c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x.strip()) for x in inner.split(",")]
    return s


def append_note(
    playbook_dir: str | Path,
    filename: str,
    content: str,
) -> Path:
    """追加一条新笔记到 playbook/notes/。filename 必须以 .md 结尾。"""
    p = Path(playbook_dir) / filename
    if p.exists():
        raise FileExistsError(f"{p} 已存在，不覆盖。Playbook 只许追加不裸删（见 format.md）")
    p.write_text(content, encoding="utf-8")
    return p


# --- 演示：直接 python -m lab.playbook_io 时打 prior summary ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="playbook 工具：读 + 解析 prior")
    ap.add_argument("--dir", default="playbook/notes", help="playbook notes 目录")
    ap.add_argument("--priority", default="medium", choices=["low", "medium", "high"])
    args = ap.parse_args()
    notes = read_notes(args.dir, args.priority)
    print(f"# 命中笔记 {len(notes)} 条（min_priority={args.priority}）")
    for n in notes:
        print(f"  - {n['path'].name} [priority={n['priority']}] prior={n['prior']}")
    merged = parse_to_prior(notes)
    print(f"\n# 合并 prior（后写覆盖先写）：")
    for k, v in merged.items():
        print(f"  {k} = {v!r}")
