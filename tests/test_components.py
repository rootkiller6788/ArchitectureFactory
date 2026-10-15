"""组件库：**按名字注册、按名字装配** —— 这条性质必须是可测的，不能只是声称。

本文件里最重要的一条是 :func:`test_assembler_uses_registry_not_branches`：
把注册表里某一项替换成一个假实现，再 ``build()``，断言假实现被调用到了。
若装配器写死成 ``if mixer == "attention": ...``，这个测试必然失败。
**这是「架构是数据，不是代码分支」唯一的硬证据。**
"""

from __future__ import annotations

import pytest

from architecturefactory import components as C
from architecturefactory.autograd import Tensor
from architecturefactory.builder import build_from_spec
from architecturefactory.ir import canonical_demo_specs, parse
from architecturefactory.schema import FFNS, MIXERS, NORMS, ArchSpec


# ----------------------------------------------------------------------
# 注册表完整性
# ----------------------------------------------------------------------
def test_every_enum_value_has_an_impl():
    """``schema`` 里的枚举值必须都有实现 —— 否则设计空间里会有「装不出来的点」。"""
    assert set(MIXERS) - {C.HYBRID_MIXER} <= set(C.MIXER_IMPLS)
    assert set(FFNS) <= set(C.FFN_IMPLS)
    assert set(NORMS) <= set(C.NORM_IMPLS)


def test_hybrid_is_not_a_component():
    """``hybrid`` 必须**不在**注册表里 —— 它是逐层展开出来的，不是一种组件。

    如果 hybrid 进了注册表，就意味着某处要 ``if mixer == 'hybrid'`` 特判，
    那「架构是数据」这句话就漏了。
    """
    assert C.HYBRID_MIXER not in C.MIXER_IMPLS


def test_unknown_name_raises_with_hint():
    with pytest.raises(KeyError, match="未注册的 mixer"):
        C.mixer_impl("nope")
    with pytest.raises(KeyError, match="未注册的 ffn"):
        C.ffn_impl("nope")
    with pytest.raises(KeyError, match="未注册的 norm"):
        C.norm_impl("nope")


# ----------------------------------------------------------------------
# 装配走查表（核心）
# ----------------------------------------------------------------------
def test_assembler_uses_registry_not_branches(cfg):
    """把注册表里的一项换成假实现，装配必须用假的 —— 证明是查表，不是分支。"""
    calls: list[str] = []

    def fake_forward(x: Tensor, p: dict[str, Tensor], n_heads: int) -> Tensor:
        calls.append("fake")
        return x  # 恒等，够用来观察「有没有被调用」

    real = C.MIXER_IMPLS["attention"]
    C.MIXER_IMPLS["attention"] = C.MixerImpl("attention", real.param_defs, fake_forward)
    try:
        spec = ArchSpec("attention", "swiglu", "rmsnorm", 1, 12, n_heads=2)
        model = build_from_spec(spec, 12, max_seq_len=5)
        model.forward([0, 1, 2, 3, 4])
    finally:
        C.MIXER_IMPLS["attention"] = real

    assert calls, "装配器没有走注册表 —— 它把 mixer 写死了"
    assert len(calls) == 1  # 1 层 = 1 次 mixer 调用


def test_assembler_uses_registry_for_ffn(cfg):
    calls: list[str] = []

    def fake_ffn(x: Tensor, p: dict[str, Tensor], n_experts: int) -> Tensor:
        calls.append("fake")
        return x

    real = C.FFN_IMPLS["mlp"]
    C.FFN_IMPLS["mlp"] = C.FfnImpl("mlp", real.param_defs, fake_ffn)
    try:
        spec = ArchSpec("ssm", "mlp", "rmsnorm", 2, 12)
        model = build_from_spec(spec, 12, max_seq_len=5)
        model.forward([0, 1, 2, 3, 4])
    finally:
        C.FFN_IMPLS["mlp"] = real
    assert len(calls) == 2  # 2 层 = 2 次 ffn 调用


# ----------------------------------------------------------------------
# 各组件形状
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mixer", ["attention", "ssm"])
@pytest.mark.parametrize("ffn", ["mlp", "swiglu", "moe"])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
def test_component_shapes_preserved(mixer, ffn, norm):
    """每个组件的输出形状必须与输入一致：``[L,d] -> [L,d]``。"""
    d, L = 12, 5
    spec = ArchSpec(mixer, ffn, norm, 1, d, n_heads=2, n_experts=2)
    model = build_from_spec(spec, 12, max_seq_len=L)
    ids = [0, 1, 2, 3, 4]
    hidden = model.hidden(ids)
    assert hidden.shape == (L, d)
    logits = model.forward(ids)
    assert logits.shape == (L, 12)


def test_hybrid_alternates_mixer():
    """hybrid 逐层交替 attention/ssm —— 由 ``mixer_at`` 决定，不是由组件决定。"""
    spec = ArchSpec("hybrid", "mlp", "rmsnorm", 3, 12, n_heads=2)
    assert [spec.mixer_at(i) for i in range(3)] == ["attention", "ssm", "attention"]
    ir = parse(spec, 12)
    assert [b.mixer for b in ir.blocks] == ["attention", "ssm", "attention"]


def test_hybrid_uses_only_registered_mixers():
    """hybrid 展开后每一层用的都必须是注册表里真实存在的组件。"""
    ir = parse(ArchSpec("hybrid", "moe", "layernorm", 2, 16, n_heads=4, n_experts=2), 12)
    for b in ir.blocks:
        assert b.mixer in C.MIXER_IMPLS


# ----------------------------------------------------------------------
# 四种经典结构共用组件
# ----------------------------------------------------------------------
def test_four_canonical_architectures_share_components():
    """Transformer / Mamba / MoE / Hybrid —— 四个 spec 只差两个字段，共用全部实现。"""
    demo = canonical_demo_specs()
    assert set(demo) == {"transformer", "mamba", "moe", "hybrid"}
    used_mixers = set()
    used_ffns = set()
    for name, spec in demo.items():
        ir = parse(spec, 12)
        for b in ir.blocks:
            used_mixers.add(b.mixer)
            used_ffns.add(b.ffn)
    # 四种结构只用到 2 个 mixer + 3 个 ffn，且全部来自同一张注册表
    assert used_mixers == {"attention", "ssm"}
    assert used_ffns == {"swiglu", "moe"}
    assert used_mixers <= set(C.MIXER_IMPLS)
    assert used_ffns <= set(C.FFN_IMPLS)


def test_attention_requires_divisible_heads():
    """d_model 不能被 n_heads 整除必须在校验阶段被拒 —— 不能等到装配才炸。"""
    from architecturefactory.ir import validate

    ir = parse(ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=5), 12)
    errs = validate(ir)
    assert any("整除" in e for e in errs)


def test_moe_needs_two_experts():
    from architecturefactory.ir import validate

    ir = parse(ArchSpec("attention", "moe", "rmsnorm", 1, 12, n_heads=2, n_experts=1), 12)
    assert any("至少要有 2 个专家" in e for e in validate(ir))


def test_ssm_ignores_heads_in_spec_id():
    """ssm 的 n_heads 无意义 —— 不该进 spec_id，否则同一架构会有两个 id。"""
    a = ArchSpec("ssm", "mlp", "rmsnorm", 1, 12, n_heads=1)
    b = ArchSpec("ssm", "mlp", "rmsnorm", 1, 12, n_heads=4)
    assert a.spec_id == b.spec_id


def test_mlp_ignores_experts_in_spec_id():
    a = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2, n_experts=0)
    b = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2, n_experts=4)
    assert a.spec_id == b.spec_id
