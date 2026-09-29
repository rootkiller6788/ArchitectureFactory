"""配置加载 / 校验 / 版本摘要。

**版本摘要只由「被搜索对象的定义」决定**：任务 + 设计空间 + 种子。
**不由搜索配置决定** —— 换一个搜索器、改一个 ``trials``，应该产生新的试验，
但不该产生新的版本号。否则「同一版本下的架构数据库」这句话就失去意义了
（每个搜索器一个版本，那还比较什么）。这条由 ``tests/test_config.py`` 钉住。
"""

from __future__ import annotations

import json
import os
from typing import Any

from .ir import expand_space
from .rng import sha256_text
from .schema import METRIC_DIRECTIONS, ArchSpec

__all__ = [
    "load_config",
    "validate_config",
    "expand_specs",
    "version_digest",
    "protocol_tag",
    "objectives_of",
    "resolve_path",
]

REQUIRED = ("task", "space", "proxy", "objectives", "search")


def load_config(path: str) -> dict[str, Any]:
    """读 JSON 配置。**不做路径魔法** —— 显式传入，便于测试与复现。"""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """缺键 / 未知目标 / 方向表不全 —— 一律在开跑前报错，别跑到一半才发现。"""
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"配置缺字段：{missing}")
    for o in cfg["objectives"]:
        if o not in METRIC_DIRECTIONS:
            raise ValueError(f"未知目标 {o!r}（可选：{sorted(METRIC_DIRECTIONS)}）")
    dirs = cfg.get("metric_directions", {})
    for o in cfg["objectives"]:
        if o not in dirs:
            raise ValueError(f"目标 {o!r} 缺 metric_directions 条目")
        if dirs[o] not in ("max", "min"):
            raise ValueError(f"目标 {o!r} 的方向必须是 max/min，得到 {dirs[o]!r}")
    if not cfg["space"].get("mixers") or not cfg["space"].get("ffns"):
        raise ValueError("space 至少要给出 mixers 与 ffns")


def objectives_of(cfg: dict[str, Any]) -> tuple[str, ...]:
    return tuple(cfg["objectives"])


def expand_specs(cfg: dict[str, Any]) -> list[ArchSpec]:
    """展开设计空间。顺序由配置的嵌套顺序决定 —— 可复现。"""
    return expand_space(cfg["space"], int(cfg["task"]["vocab_size"]))


def _observed_definition(cfg: dict[str, Any]) -> dict[str, Any]:
    """「被搜索对象」的定义：任务 + 空间 + 种子。**不含 proxy / search / trials。**"""
    task = {k: v for k, v in cfg["task"].items() if not k.startswith("_")}
    space = {k: v for k, v in cfg["space"].items() if not k.startswith("_")}
    return {"task": task, "space": space, "seed": int(cfg.get("seed", 0))}


def version_digest(cfg: dict[str, Any]) -> str:
    """内容寻址版本号：对「被搜索对象」的定义取 sha256（前 12 位）。"""
    return sha256_text(json.dumps(_observed_definition(cfg), sort_keys=True))[:12]


def protocol_tag(cfg: dict[str, Any]) -> str:
    """训练协议标签。**它进试验记录，不进版本号** —— 见模块 docstring。"""
    from .proxy import protocol_digest

    return protocol_digest(cfg)


def resolve_path(cfg: dict[str, Any], key: str, default: str) -> str:
    """从 ``cfg["paths"]`` 取路径，缺省用 ``default``。相对路径按当前工作目录解析。"""
    paths = cfg.get("paths", {})
    p = paths.get(key, default)
    return os.path.abspath(p)
