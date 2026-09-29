"""Architecture IR：``Spec → IR`` 解析、结构合法性校验、设计空间展开。

**这是本项目「不是代码分支」这条命题的落点。** 一个架构被解析成
``Embedding → Block × N → Head`` 的图，每个 block 只是「组件名字」的集合；
组件实现按名字查表装配（:mod:`architecturefactory.components`）。
于是「枚举设计空间」「变异」「交叉」全部作用在**数据**上，而不是在代码上。
"""

from __future__ import annotations

from typing import Any, Iterable

from .schema import (
    ATTENTIVE_MIXERS,
    ArchIR,
    ArchSpec,
    BlockIR,
    FFNS,
    MIXERS,
    NORMS,
    RESIDUALS,
)

#: FFN 宽度系数（组件与参数公式共用同一份常量，避免两处各写一个）
MLP_EXPANSION = 4
SWIGLU_EXPANSION = 2


def ffn_width(ffn: str, d_model: int) -> int:
    """FFN 中间宽度。``swiglu`` 有三块矩阵，所以系数更小以保持参数量可比。"""
    if ffn == "swiglu":
        return SWIGLU_EXPANSION * d_model
    return MLP_EXPANSION * d_model


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def parse(spec: ArchSpec, vocab_size: int) -> ArchIR:
    """``ArchSpec → ArchIR``。hybrid 的 mixer 逐层交替（偶数层 attention、奇数层 ssm）。"""
    s = spec.canonical()
    blocks: list[BlockIR] = []
    for i in range(s.n_layers):
        m = s.mixer_at(i)
        blocks.append(
            BlockIR(
                mixer=m,
                ffn=s.ffn,
                norm=s.norm,
                residual=s.residual,
                d_model=s.d_model,
                n_heads=s.n_heads if m == "attention" else 1,
                n_experts=s.n_experts if s.ffn == "moe" else 0,
            )
        )
    return ArchIR(vocab_size=vocab_size, d_model=s.d_model, blocks=blocks)


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------
def validate(ir: ArchIR, max_params: int | None = None, max_seq_len: int = 0) -> list[str]:
    """返回**错误列表**（空 = 合法）。构建前先校验，别让坏结构走到实例化。"""
    errs: list[str] = []
    if ir.d_model <= 0:
        errs.append(f"d_model 必须为正：{ir.d_model}")
    if ir.n_layers <= 0:
        errs.append("至少要有 1 个 block")
    for i, b in enumerate(ir.blocks):
        if b.mixer not in MIXERS:
            errs.append(f"block{i}: 未知 mixer {b.mixer!r}")
        if b.ffn not in FFNS:
            errs.append(f"block{i}: 未知 ffn {b.ffn!r}")
        if b.norm not in NORMS:
            errs.append(f"block{i}: 未知 norm {b.norm!r}")
        if b.residual not in RESIDUALS:
            errs.append(f"block{i}: 未知 residual {b.residual!r}")
        if b.mixer == "attention":
            if b.n_heads <= 0:
                errs.append(f"block{i}: attention 的 n_heads 必须为正")
            elif b.d_model % b.n_heads != 0:
                errs.append(
                    f"block{i}: d_model({b.d_model}) 不能被 n_heads({b.n_heads}) 整除"
                )
        if b.ffn == "moe" and b.n_experts < 2:
            errs.append(f"block{i}: moe 至少要有 2 个专家，得到 {b.n_experts}")
    if max_params is not None:
        est = estimate_params(ir, max_seq_len)
        if est > max_params:
            errs.append(f"参数量 {est} 超过预算 {max_params}")
    return errs


# ----------------------------------------------------------------------
# 参数估计（校验用；权威计数来自实例化后的模型，见 profile.py）
# ----------------------------------------------------------------------
def _mixer_params(mixer: str, d: int) -> int:
    if mixer == "attention":
        return 4 * d * d + 4 * d  # Wq,Wk,Wv,Wo + 各自 bias
    if mixer == "ssm":
        return 3 * d * d + 3 * d  # 门 W_a、值 W_v、输出 W_o + 各自 bias
    raise ValueError(f"未知 mixer：{mixer}")


def _ffn_params(ffn: str, d: int, n_experts: int) -> int:
    if ffn == "mlp":
        f = MLP_EXPANSION * d
        return 2 * d * f + f + d
    if ffn == "swiglu":
        f = SWIGLU_EXPANSION * d
        return 3 * d * f
    if ffn == "moe":
        f = MLP_EXPANSION * d
        per_expert = 2 * d * f + f + d
        router = d * n_experts + n_experts
        return n_experts * per_expert + router
    raise ValueError(f"未知 ffn：{ffn}")


def _norm_params(norm: str, d: int) -> int:
    return 2 * d if norm == "layernorm" else d


def estimate_params(ir: ArchIR, max_seq_len: int = 0) -> int:
    """参数量公式（与组件实现必须一致 —— 由测试钉住）。

    ``max_seq_len > 0`` 时计入位置嵌入表。**位置嵌入对所有架构一视同仁**，
    它是「输入表示」的一部分而不是「架构」的一部分：见 :func:`parse` 的说明。
    """
    d = ir.d_model
    total = ir.vocab_size * d  # embedding
    if max_seq_len > 0:
        total += max_seq_len * d  # 位置嵌入表
    for b in ir.blocks:
        total += _mixer_params(b.mixer, d)
        total += _ffn_params(b.ffn, d, b.n_experts)
        total += 2 * _norm_params(b.norm, d)  # 每个 block 两个 norm（mixer 前 + ffn 前）
    total += d * ir.vocab_size  # 输出头（与 embedding 同形）
    return total


# ----------------------------------------------------------------------
# 设计空间展开
# ----------------------------------------------------------------------
def expand_space(space: dict[str, Any], vocab_size: int = 0) -> list[ArchSpec]:
    """笛卡尔积展开设计空间；按 ``spec_id`` 去重（无关字段已归一）。

    展开顺序 = 配置里的嵌套顺序，保证**可复现**：同一份配置永远给出同一个列表顺序。
    """
    specs: list[ArchSpec] = []
    seen: set[str] = set()
    for mixer in space["mixers"]:
        for ffn in space["ffns"]:
            for norm in space["norms"]:
                for n_layers in space["n_layers"]:
                    for d_model in space["d_models"]:
                        heads: Iterable[int] = (
                            space.get("n_heads", [1]) if mixer in ATTENTIVE_MIXERS else [1]
                        )
                        experts: Iterable[int] = (
                            space.get("n_experts", [0]) if ffn == "moe" else [0]
                        )
                        for h in heads:
                            for e in experts:
                                if mixer in ATTENTIVE_MIXERS and d_model % h != 0:
                                    continue
                                spec = ArchSpec(
                                    mixer=mixer,
                                    ffn=ffn,
                                    norm=norm,
                                    n_layers=n_layers,
                                    d_model=d_model,
                                    n_heads=h,
                                    n_experts=e,
                                ).canonical()
                                if spec.spec_id in seen:
                                    continue
                                seen.add(spec.spec_id)
                                specs.append(spec)
    return specs


def canonical_demo_specs() -> dict[str, ArchSpec]:
    """README 承诺的四种结构：Transformer / Mamba / MoE / Hybrid。

    它们是「同一套组件不同装配」的最小证据 —— 四个 spec 共用全部组件实现，
    只有 ``mixer`` / ``ffn`` 两个字段不同。
    """
    return {
        "transformer": ArchSpec("attention", "swiglu", "rmsnorm", 2, 24, n_heads=4),
        "mamba": ArchSpec("ssm", "swiglu", "rmsnorm", 2, 24),
        "moe": ArchSpec("attention", "moe", "rmsnorm", 2, 24, n_heads=4, n_experts=4),
        "hybrid": ArchSpec("hybrid", "moe", "rmsnorm", 2, 24, n_heads=4, n_experts=4),
    }
