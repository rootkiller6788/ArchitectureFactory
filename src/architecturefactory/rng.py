"""确定性随机：一律 ``sha256`` 派生，绝不用内置 ``hash()``。

两条纪律（与同工作区其它项目一致）：

1. **禁止内置 ``hash()``**：``PYTHONHASHSEED`` 会让它跨进程不稳定，
   直接破坏「同一配置重跑结果一致」这条前提。一律 ``sha256`` 派生。
2. **不用全局 ``random`` 状态**：全局状态会被任何一次调用推进，导致结果依赖调用顺序。
   按 ``(seed, 用途, 对象 id)`` 派生**独立实例**。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def _join(parts: tuple[Any, ...]) -> str:
    return "|".join(str(p) for p in parts)


def derive_seed(*parts: Any) -> int:
    """从若干标识派生一个稳定种子（取 sha256 前 8 字节）。"""
    digest = hashlib.sha256(_join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def make_rng(*parts: Any) -> random.Random:
    """派生一个独立的 ``random.Random`` 实例。"""
    return random.Random(derive_seed(*parts))


def stable_hash64(*parts: Any) -> int:
    """取全部字节的 64 位稳定哈希（用于内容寻址 / 去重键）。"""
    digest = hashlib.sha256(_join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
