"""结构画像：参数量 / FLOPs / 激活元素数。**全部可手算、可逐位复现。**

这是本项目两个目标里的**成本轴**。质量靠代理训练测（:mod:`architecturefactory.proxy`），
成本靠这里算 —— 因为「参数量」和「FLOPs」不该被测量噪声污染，它们是被定义出来的。

关键不变量（由 ``tests/test_profile.py`` 钉住）：

    profile(model).params == ir.estimate_params(ir)

即「实例化后数出来的参数量」必须等于「校验阶段用的公式」。两份实现漂移是这类项目
最典型的静默 bug：校验说没超预算，装配出来却超了。

FLOPs 口径：**乘加算 2 次浮点运算**（一次乘、一次加）。只数矩阵乘法与归一化，
嵌入查表不算（它没有浮点运算）。公式写在函数里、逐项注释，方便手算核对。
"""

from __future__ import annotations

from .builder import Model
from .ir import estimate_params, ffn_width
from .schema import ArchIR, Profile

__all__ = ["profile_model", "profile_ir", "estimate_flops", "estimate_activations"]


def estimate_flops(ir: ArchIR, seq_len: int) -> int:
    """一次前向的浮点运算量（乘加算 2）。"""
    L, d = seq_len, ir.d_model
    macs = 0
    for b in ir.blocks:
        if b.mixer == "attention":
            # 4 个 [d,d] 投影：4·L·d·d ；scores + P@V：2·L²·d
            macs += 4 * L * d * d + 2 * L * L * d
        elif b.mixer == "ssm":
            # 3 个投影 + 递推的逐元素乘加（约 3 项/元素）
            macs += 3 * L * d * d + 3 * L * d
        else:  # pragma: no cover - IR 已经过校验
            raise ValueError(f"未知 mixer：{b.mixer}")
        f = ffn_width(b.ffn, d)
        if b.ffn == "mlp":
            macs += 2 * L * d * f
        elif b.ffn == "swiglu":
            macs += 3 * L * d * f
        elif b.ffn == "moe":
            # 每个专家都是一次 MLP；路由是 L·d·E
            macs += b.n_experts * (2 * L * d * f) + L * d * b.n_experts
        else:  # pragma: no cover
            raise ValueError(f"未知 ffn：{b.ffn}")
        macs += 2 * (2 * L * d)  # 两个归一化
    macs += L * d * ir.vocab_size  # 输出头
    return 2 * macs


def estimate_activations(ir: ArchIR, seq_len: int) -> int:
    """前向过程中产生过的激活元素总数（不是峰值）——「这个结构有多宽」。"""
    L = seq_len
    total = 0
    for b in ir.blocks:
        f = ffn_width(b.ffn, ir.d_model)
        if b.ffn == "moe":
            f *= b.n_experts
        total += L * (ir.d_model + f)  # 残差流 + 最宽的中间层
        if b.mixer == "attention":
            total += L * L  # 注意力矩阵
    total += L * ir.vocab_size  # 输出 logits
    return total


def profile_ir(ir: ArchIR, seq_len: int) -> Profile:
    return Profile(
        params=estimate_params(ir, max_seq_len=seq_len),
        flops=estimate_flops(ir, seq_len),
        activation_elems=estimate_activations(ir, seq_len),
        n_blocks=ir.n_layers,
        d_model=ir.d_model,
    )


def profile_model(model: Model, seq_len: int) -> Profile:
    """实测画像。``params`` 来自**真的数了一遍参数**，其余来自公式。"""
    return Profile(
        params=model.num_params(),
        flops=estimate_flops(model.ir, seq_len),
        activation_elems=estimate_activations(model.ir, seq_len),
        n_blocks=model.ir.n_layers,
        d_model=model.ir.d_model,
    )
