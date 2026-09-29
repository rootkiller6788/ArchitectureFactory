"""组件库：Mixer / FFN / Norm / Residual —— **按名字注册，按名字装配**。

**这是「架构是数据，不是代码分支」这句话的实现处。**

README 第二节点名了要避免的写法：

    if architecture == "transformer": ...
    elif architecture == "mamba": ...

所以本模块里**没有一处** ``if spec.mixer == ...`` / ``if spec.ffn == ...`` 的前向分支。
每个组件都只声明两件事：

1. 我需要哪些参数（名字 + 形状 + 初始化类型）—— :class:`ParamDef`；
2. 拿到 ``x`` 和我的参数之后怎么算 —— ``forward``。

然后注册进 :data:`MIXER_IMPLS` / :data:`FFN_IMPLS` / :data:`NORM_IMPLS`。
装配器（:mod:`architecturefactory.builder`）只做 ``REGISTRY[名字]`` 查表。

**这条性质是可测的**：``tests/test_components.py`` 里把注册表里某一项临时替换成
一个假实现，再 ``build()``，断言假实现被调用到了 —— 若装配器写死成 ``if/elif``，
这个测试必然失败。

参数量公式在 :mod:`architecturefactory.ir` 里另写了一份（用于构建前校验）。
两份必须一致，由 ``tests/test_profile.py`` 钉住 —— 公式与实现漂移是这类项目
最常见的静默 bug。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .autograd import (
    Tensor,
    add,
    add_bias,
    concat_cols,
    constant,
    embed,
    gather_rows,
    gelu,
    layernorm,
    matmul,
    matvec,
    mul,
    rmsnorm,
    row_scale,
    scale,
    sigmoid,
    silu,
    softmax_rows,
    split_cols,
    stack_rows,
    take_col,
    take_row,
    transpose,
)
from .ir import ffn_width

# ----------------------------------------------------------------------
# 参数声明
# ----------------------------------------------------------------------
#: 初始化类型。``weight`` 用 ±init_scale 均匀分布；``bias`` 置零；
#: ``norm_w`` 置一、``norm_b`` 置零 —— 归一化层的权重必须初始化为 1，
#: 否则「归一化」在初始化时等于在缩放，训练一开始就偏了。
KIND_WEIGHT = "weight"
KIND_BIAS = "bias"
KIND_NORM_W = "norm_w"
KIND_NORM_B = "norm_b"


@dataclass(frozen=True)
class ParamDef:
    """组件对参数的一条声明。``name`` 是组件内的相对名（``wq`` / ``b1`` / ...）。"""

    name: str
    shape: tuple[int, ...]
    kind: str = KIND_WEIGHT


# ----------------------------------------------------------------------
# 组件协议
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MixerImpl:
    """序列混合器。输入输出都是 ``[L, d]``。

    ``param_defs(d_model, n_heads)`` —— 只有注意力型需要 ``n_heads``；
    ``forward(x, params, n_heads)`` —— ``params`` 是**已切好**的本组件参数字典。
    """

    name: str
    param_defs: Callable[[int, int], list[ParamDef]]
    forward: Callable[[Tensor, dict[str, Tensor], int], Tensor]


@dataclass(frozen=True)
class FfnImpl:
    """逐位置前馈。输入输出都是 ``[L, d]``。"""

    name: str
    param_defs: Callable[[int, int], list[ParamDef]]
    forward: Callable[[Tensor, dict[str, Tensor], int], Tensor]


@dataclass(frozen=True)
class NormImpl:
    name: str
    param_defs: Callable[[int], list[ParamDef]]
    forward: Callable[[Tensor, dict[str, Tensor]], Tensor]


# ----------------------------------------------------------------------
# Mixer 实现
# ----------------------------------------------------------------------
def _attention_defs(d: int, n_heads: int) -> list[ParamDef]:
    # 4 个 [d,d] 投影 + 4 个 [d] 偏置 —— 与 ir._mixer_params 对齐
    return [
        ParamDef("wq", (d, d)),
        ParamDef("bq", (d,), KIND_BIAS),
        ParamDef("wk", (d, d)),
        ParamDef("bk", (d,), KIND_BIAS),
        ParamDef("wv", (d, d)),
        ParamDef("bv", (d,), KIND_BIAS),
        ParamDef("wo", (d, d)),
        ParamDef("bo", (d,), KIND_BIAS),
    ]


def _attention_forward(x: Tensor, p: dict[str, Tensor], n_heads: int) -> Tensor:
    """多头因果自注意力。``[L,d] -> [L,d]``。

    多头切分用 :func:`split_cols` / :func:`concat_cols`，因果掩码交给
    :func:`softmax_rows` 的 ``mask`` 参数（``mask[i][j] = j > i``）。
    """
    L, d = len(x.data), len(x.data[0])
    dh = d // n_heads
    q = add_bias(matmul(x, p["wq"]), p["bq"])
    k = add_bias(matmul(x, p["wk"]), p["bk"])
    v = add_bias(matmul(x, p["wv"]), p["bv"])
    qs = split_cols(q, [dh] * n_heads)
    ks = split_cols(k, [dh] * n_heads)
    vs = split_cols(v, [dh] * n_heads)
    mask = [[j > i for j in range(L)] for i in range(L)]
    inv_scale = 1.0 / math.sqrt(dh)
    heads: list[Tensor] = []
    for h in range(n_heads):
        scores = scale(matmul(qs[h], transpose(ks[h])), inv_scale)  # [L,L]
        attn = softmax_rows(scores, mask)
        heads.append(matmul(attn, vs[h]))  # [L,dh]
    y = matmul(concat_cols(heads), p["wo"])
    return add_bias(y, p["bo"])


def _ssm_defs(d: int, n_heads: int) -> list[ParamDef]:
    # 门 / 值 / 输出 三组 [d,d] + 偏置 —— 与 ir._mixer_params 对齐
    return [
        ParamDef("wa", (d, d)),
        ParamDef("ba", (d,), KIND_BIAS),
        ParamDef("wv", (d, d)),
        ParamDef("bv", (d,), KIND_BIAS),
        ParamDef("wo", (d, d)),
        ParamDef("bo", (d,), KIND_BIAS),
    ]


def _ssm_forward(x: Tensor, p: dict[str, Tensor], n_heads: int) -> Tensor:
    """输入相关衰减的门控线性递推（对角 SSM 的最小可训练形态）。

    ``a_t = sigmoid(x_t W_a)``，``v_t = x_t W_v``，``h_t = a_t ⊙ h_{t-1} + (1-a_t) ⊙ v_t``，
    ``y_t = h_t W_o``。因果、``O(L·d²)``、**没有任何两两交互** —— 与注意力在结构上
    与成本上都不同，这正是要观测的差异。``n_heads`` 对本组件无意义。
    """
    L, d = len(x.data), len(x.data[0])
    a = sigmoid(add_bias(matmul(x, p["wa"]), p["ba"]))  # [L,d] 衰减 ∈ (0,1)
    v = add_bias(matmul(x, p["wv"]), p["bv"])  # [L,d]
    ones = constant([1.0] * d)
    h = constant([0.0] * d)
    rows: list[Tensor] = []
    for t in range(L):
        a_t = take_row(a, t)
        v_t = take_row(v, t)
        keep = mul(a_t, h)
        write = mul(add(scale(a_t, -1.0), ones), v_t)  # (1 - a_t) ⊙ v_t
        h = add(keep, write)
        rows.append(add(matvec(h, p["wo"]), p["bo"]))  # [d]
    return stack_rows(rows)  # [L,d]


# ----------------------------------------------------------------------
# FFN 实现
# ----------------------------------------------------------------------
def _mlp_defs(d: int, n_experts: int) -> list[ParamDef]:
    f = ffn_width("mlp", d)
    return [
        ParamDef("w1", (d, f)),
        ParamDef("b1", (f,), KIND_BIAS),
        ParamDef("w2", (f, d)),
        ParamDef("b2", (d,), KIND_BIAS),
    ]


def _mlp_forward(x: Tensor, p: dict[str, Tensor], n_experts: int) -> Tensor:
    h = gelu(add_bias(matmul(x, p["w1"]), p["b1"]))
    return add_bias(matmul(h, p["w2"]), p["b2"])


def _swiglu_defs(d: int, n_experts: int) -> list[ParamDef]:
    f = ffn_width("swiglu", d)
    # 三块矩阵、无偏置（与 ir._ffn_params 的 3·d·f 对齐）
    return [ParamDef("w1", (d, f)), ParamDef("w3", (d, f)), ParamDef("w2", (f, d))]


def _swiglu_forward(x: Tensor, p: dict[str, Tensor], n_experts: int) -> Tensor:
    gate = silu(matmul(x, p["w1"]))
    up = matmul(x, p["w3"])
    return matmul(mul(gate, up), p["w2"])


def _moe_defs(d: int, n_experts: int) -> list[ParamDef]:
    f = ffn_width("moe", d)
    defs = [
        ParamDef("router", (d, n_experts)),
        ParamDef("router_b", (n_experts,), KIND_BIAS),
    ]
    for e in range(n_experts):
        defs += [
            ParamDef(f"w1_{e}", (d, f)),
            ParamDef(f"b1_{e}", (f,), KIND_BIAS),
            ParamDef(f"w2_{e}", (f, d)),
            ParamDef(f"b2_{e}", (d,), KIND_BIAS),
        ]
    return defs


def _moe_forward(x: Tensor, p: dict[str, Tensor], n_experts: int) -> Tensor:
    """软路由 MoE：``softmax(x W_r)`` 加权全部专家输出。

    刻意**不做 top-k 离散选择** —— 离散选择让梯度只走被选中的专家，代理训练在
    150 步内会退化。软混合是这里可复现且可训练的选择；真实系统当然用 top-k。
    """
    gates = softmax_rows(add_bias(matmul(x, p["router"]), p["router_b"]))  # [L,E]
    out: Tensor | None = None
    for e in range(n_experts):
        h = gelu(add_bias(matmul(x, p[f"w1_{e}"]), p[f"b1_{e}"]))
        y = add_bias(matmul(h, p[f"w2_{e}"]), p[f"b2_{e}"])  # [L,d]
        contrib = row_scale(y, take_col(gates, e))  # 每行乘自己的门控
        out = contrib if out is None else add(out, contrib)
    assert out is not None
    return out


# ----------------------------------------------------------------------
# Norm 实现
# ----------------------------------------------------------------------
def _layernorm_defs(d: int) -> list[ParamDef]:
    return [ParamDef("w", (d,), KIND_NORM_W), ParamDef("b", (d,), KIND_NORM_B)]


def _rmsnorm_defs(d: int) -> list[ParamDef]:
    return [ParamDef("w", (d,), KIND_NORM_W)]


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------
MIXER_IMPLS: dict[str, MixerImpl] = {
    "attention": MixerImpl("attention", _attention_defs, _attention_forward),
    "ssm": MixerImpl("ssm", _ssm_defs, _ssm_forward),
}

FFN_IMPLS: dict[str, FfnImpl] = {
    "mlp": FfnImpl("mlp", _mlp_defs, _mlp_forward),
    "swiglu": FfnImpl("swiglu", _swiglu_defs, _swiglu_forward),
    "moe": FfnImpl("moe", _moe_defs, _moe_forward),
}

NORM_IMPLS: dict[str, NormImpl] = {
    "layernorm": NormImpl(
        "layernorm", _layernorm_defs, lambda x, p: layernorm(x, p["w"], p["b"])
    ),
    "rmsnorm": NormImpl("rmsnorm", _rmsnorm_defs, lambda x, p: rmsnorm(x, p["w"])),
}

#: ``hybrid`` 不是独立组件 —— 它由 ``ArchSpec.mixer_at`` 逐层展开成
#: ``attention`` / ``ssm`` 两个已注册实现。**注册表里不该出现 hybrid**：
#: 否则「架构」就又变回一个特例分支了。这条由测试钉住。
HYBRID_MIXER = "hybrid"


def mixer_impl(name: str) -> MixerImpl:
    try:
        return MIXER_IMPLS[name]
    except KeyError:  # pragma: no cover - 配置写错时的兜底提示
        raise KeyError(f"未注册的 mixer：{name!r}（已注册：{sorted(MIXER_IMPLS)}）") from None


def ffn_impl(name: str) -> FfnImpl:
    try:
        return FFN_IMPLS[name]
    except KeyError:  # pragma: no cover
        raise KeyError(f"未注册的 ffn：{name!r}（已注册：{sorted(FFN_IMPLS)}）") from None


def norm_impl(name: str) -> NormImpl:
    try:
        return NORM_IMPLS[name]
    except KeyError:  # pragma: no cover
        raise KeyError(f"未注册的 norm：{name!r}（已注册：{sorted(NORM_IMPLS)}）") from None


__all__ = [
    "KIND_WEIGHT",
    "KIND_BIAS",
    "KIND_NORM_W",
    "KIND_NORM_B",
    "ParamDef",
    "MixerImpl",
    "FfnImpl",
    "NormImpl",
    "MIXER_IMPLS",
    "FFN_IMPLS",
    "NORM_IMPLS",
    "HYBRID_MIXER",
    "mixer_impl",
    "ffn_impl",
    "norm_impl",
]
