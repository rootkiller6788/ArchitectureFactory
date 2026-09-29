"""ArchitectureFactory —— 把架构变成可解析、可组合、可搜索的 IR。

核心命题（README 第二节）：**架构不该是代码分支**，而应是可序列化、可组合、
可搜索的图。同一套组件（Mixer × FFN × Norm × Residual）装配出 Transformer /
Mamba / MoE / Hybrid，于是「架构设计空间」变成一个可枚举、可搜索的空间。

本包只导出稳定对象；内部模块可按需 ``from architecturefactory import autograd`` 取用。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
