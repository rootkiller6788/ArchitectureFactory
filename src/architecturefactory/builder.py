"""装配器：``ArchIR → Model``。**只查表，不判断架构。**

:func:`build` 的全部逻辑就是：遍历 IR 的 block，对每个 block 的
``mixer`` / ``ffn`` / ``norm`` 三个名字去 :mod:`architecturefactory.components`
的注册表里取实现，按 ``ParamDef`` 声明造参数，然后串起来。

这里**没有任何** ``if spec.mixer == "attention"`` —— 加一个新 mixer 只需要
在 components 里注册一个实现 + 在 schema 的枚举里加一个名字，
本文件一个字都不用改。这是「架构是数据」的可验证含义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .autograd import Tensor, add, embed, gather_rows, matmul
from .components import (
    KIND_BIAS,
    KIND_NORM_B,
    KIND_NORM_W,
    KIND_WEIGHT,
    ParamDef,
    ffn_impl,
    mixer_impl,
    norm_impl,
)
from .ir import validate
from .rng import make_rng
from .schema import ArchIR, ArchSpec

__all__ = ["Model", "build", "build_from_spec"]


# ----------------------------------------------------------------------
# 初始化
# ----------------------------------------------------------------------
def _init_values(shape: tuple[int, ...], kind: str, scale: float, rng: Any) -> list[Any]:
    """按 ``kind`` 造初始值。**归一化权重置一、偏置置零** —— 见 components 的说明。"""
    if kind == KIND_BIAS or kind == KIND_NORM_B:
        if len(shape) == 1:
            return [0.0] * shape[0]
        return [[0.0] * shape[1] for _ in range(shape[0])]
    if kind == KIND_NORM_W:
        return [1.0] * shape[0]
    if len(shape) == 1:
        return [rng.uniform(-scale, scale) for _ in range(shape[0])]
    rows, cols = shape
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _shape_size(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
@dataclass
class BlockParams:
    """一个 block 的参数字典（已按组件切好，装配时直接递进去）。"""

    norm1: dict[str, Tensor]
    mixer: dict[str, Tensor]
    norm2: dict[str, Tensor]
    ffn: dict[str, Tensor]


@dataclass
class Model:
    """可训练模型。``forward`` 返回**全部位置**的 logits ``[L, V]``。

    只监督最后一个位置时由调用方 :func:`architecturefactory.autograd.gather_rows`
    取行 —— 模型不替调用方猜「你要哪几个位置」。
    """

    ir: ArchIR
    params: dict[str, Tensor] = field(default_factory=dict)
    blocks: list[BlockParams] = field(default_factory=list)
    init_scale: float = 0.08
    seed: int = 0
    max_seq_len: int = 0
    spec: ArchSpec | None = None

    # -- 前向 ----------------------------------------------------------
    def _embed(self, ids: Sequence[int]) -> Tensor:
        """词嵌入（+ 位置嵌入，若启用）。**所有架构走同一条表示路径。**"""
        x = embed(self.params["emb"], list(ids))  # [L,d]
        if self.max_seq_len > 0:
            x = add(x, gather_rows(self.params["pos_emb"], range(len(ids))))
        return x

    def forward(self, ids: Sequence[int]) -> Tensor:
        x = self._embed(ids)
        for b, bp in zip(self.ir.blocks, self.blocks):
            h = norm_impl(b.norm).forward(x, bp.norm1)
            h = mixer_impl(b.mixer).forward(h, bp.mixer, b.n_heads)
            x = add(x, h)  # pre-norm 残差
            h2 = norm_impl(b.norm).forward(x, bp.norm2)
            h2 = ffn_impl(b.ffn).forward(h2, bp.ffn, b.n_experts)
            x = add(x, h2)
        return matmul(x, self.params["head"])  # [L,V]

    def hidden(self, ids: Sequence[int]) -> Tensor:
        """到最后一个 block 输出为止（不含 head）。探针/画图用。"""
        x = self._embed(ids)
        for b, bp in zip(self.ir.blocks, self.blocks):
            h = norm_impl(b.norm).forward(x, bp.norm1)
            h = mixer_impl(b.mixer).forward(h, bp.mixer, b.n_heads)
            x = add(x, h)
            h2 = norm_impl(b.norm).forward(x, bp.norm2)
            h2 = ffn_impl(b.ffn).forward(h2, bp.ffn, b.n_experts)
            x = add(x, h2)
        return x

    # -- 参数 ----------------------------------------------------------
    def parameter_list(self) -> list[Tensor]:
        """**按名字排序**返回，保证任何调用方拿到稳定顺序（优化器状态依赖它）。"""
        return [self.params[k] for k in sorted(self.params)]

    def num_params(self) -> int:
        return sum(_shape_size(t.shape) for t in self.params.values())

    def zero_grad(self) -> None:
        for t in self.params.values():
            t.zero_grad()

    def param_report(self) -> dict[str, int]:
        return {k: _shape_size(v.shape) for k, v in sorted(self.params.items())}

    def spec_id(self) -> str:
        """装配出来的模型的标识：优先用 spec_id，只有 IR 时退化为 IR 指纹。"""
        if self.spec is not None:
            return self.spec.spec_id
        return "ir-" + self.ir.fingerprint()[:12]


# ----------------------------------------------------------------------
# 装配
# ----------------------------------------------------------------------
def build(
    ir: ArchIR, init_scale: float = 0.08, seed: int = 0, max_seq_len: int = 0
) -> Model:
    """``ArchIR → Model``。**只查表**：见模块 docstring。

    ``max_seq_len > 0`` 时挂一张位置嵌入表。位置嵌入是**输入表示**的一部分，
    所有架构共享同一张表 —— 比较的是 mixer/ffn 的装配，不是「谁有位置信息」。
    """
    errs = validate(ir, max_seq_len=max_seq_len)
    if errs:
        raise ValueError("IR 非法，拒绝装配：\n  - " + "\n  - ".join(errs))

    model = Model(ir=ir, init_scale=init_scale, seed=seed, max_seq_len=max_seq_len)
    fp = ir.fingerprint()
    params: dict[str, Tensor] = {}

    def make(prefix: str, defs: list[ParamDef]) -> dict[str, Tensor]:
        """按声明造参数。每个参数**独立派生 RNG** —— 初始化与遍历顺序无关。"""
        out: dict[str, Tensor] = {}
        for d in defs:
            rng = make_rng(seed, "init", fp, prefix + d.name)
            vals = _init_values(d.shape, d.kind, init_scale, rng)
            t = Tensor(vals)
            params[prefix + d.name] = t
            out[d.name] = t
        return out

    d = ir.d_model
    params["emb"] = Tensor(_init_values((ir.vocab_size, d), KIND_WEIGHT, init_scale,
                                        make_rng(seed, "init", fp, "emb")))
    if max_seq_len > 0:
        params["pos_emb"] = Tensor(
            _init_values((max_seq_len, d), KIND_WEIGHT, init_scale,
                         make_rng(seed, "init", fp, "pos_emb"))
        )
    for i, b in enumerate(ir.blocks):
        pre = f"block{i}."
        bp = BlockParams(
            norm1=make(pre + "norm1.", norm_impl(b.norm).param_defs(d)),
            mixer=make(pre + "mixer.", mixer_impl(b.mixer).param_defs(d, b.n_heads)),
            norm2=make(pre + "norm2.", norm_impl(b.norm).param_defs(d)),
            ffn=make(pre + "ffn.", ffn_impl(b.ffn).param_defs(d, b.n_experts)),
        )
        model.blocks.append(bp)
    params["head"] = Tensor(_init_values((d, ir.vocab_size), KIND_WEIGHT, init_scale,
                                         make_rng(seed, "init", fp, "head")))
    model.params = params
    return model


def build_from_spec(
    spec: ArchSpec,
    vocab_size: int,
    init_scale: float = 0.08,
    seed: int = 0,
    max_seq_len: int = 0,
) -> Model:
    from .ir import parse

    model = build(
        parse(spec, vocab_size), init_scale=init_scale, seed=seed, max_seq_len=max_seq_len
    )
    model.spec = spec.canonical()
    return model
