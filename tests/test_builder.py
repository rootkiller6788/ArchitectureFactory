"""装配器与画像。**最重要的一条不变量：实测参数量 == 公式估计量。**

这两份实现（:func:`builder.Model.num_params` 与 :func:`ir.estimate_params`）
是分开写的，因为它们服务于不同阶段（装配后 vs 构建前校验）。
两份漂移的后果是静默的：校验说没超预算，装配出来却超了。
所以必须对**空间里的每一个点**都钉一遍，而不是抽查几个。
"""

from __future__ import annotations

import pytest

from architecturefactory.builder import build, build_from_spec
from architecturefactory.config import expand_specs
from architecturefactory.ir import canonical_demo_specs, estimate_params, parse
from architecturefactory.profile import (
    estimate_activations,
    estimate_flops,
    profile_ir,
    profile_model,
)
from architecturefactory.schema import ArchSpec


@pytest.mark.parametrize("name", ["transformer", "mamba", "moe", "hybrid"])
def test_four_canonical_architectures_build_and_run(name):
    spec = canonical_demo_specs()[name]
    model = build_from_spec(spec, 12, max_seq_len=9)
    logits = model.forward([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert logits.shape == (9, 12)
    assert model.spec is not None and model.spec.spec_id == spec.spec_id


def test_params_formula_matches_instantiation_for_whole_space(cfg):
    """**全空间**逐点核对参数量：公式 == 实测。"""
    seq_len = 5
    bad = []
    for spec in expand_specs(cfg):
        ir = parse(spec, 12)
        model = build(ir, max_seq_len=seq_len)
        got = model.num_params()
        want = estimate_params(ir, max_seq_len=seq_len)
        if got != want:
            bad.append((spec.spec_id, got, want))
    assert not bad, f"公式与实测不符（spec_id, 实测, 公式）：{bad[:5]}"


def test_profile_matches_estimate(cfg):
    spec = ArchSpec("attention", "swiglu", "rmsnorm", 2, 16, n_heads=4)
    ir = parse(spec, 12)
    model = build(ir, max_seq_len=5)
    a = profile_model(model, 5)
    b = profile_ir(ir, 5)
    assert a.params == b.params
    assert a.flops == b.flops
    assert a.activation_elems == b.activation_elems


def test_position_embedding_counted_when_enabled():
    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    ir = parse(spec, 12)
    without = estimate_params(ir, max_seq_len=0)
    with_pos = estimate_params(ir, max_seq_len=5)
    assert with_pos - without == 5 * 12


def test_no_pos_emb_when_disabled():
    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    model = build_from_spec(spec, 12, max_seq_len=0)
    assert "pos_emb" not in model.params
    with_pos = build_from_spec(spec, 12, max_seq_len=5)
    assert "pos_emb" in with_pos.params


# ----------------------------------------------------------------------
# 确定性
# ----------------------------------------------------------------------
def test_initialization_is_deterministic():
    """同一 (seed, spec) 必须给出逐位相同的初始参数。"""
    spec = ArchSpec("hybrid", "moe", "layernorm", 2, 16, n_heads=4, n_experts=2)
    a = build_from_spec(spec, 12, seed=7, max_seq_len=5)
    b = build_from_spec(spec, 12, seed=7, max_seq_len=5)
    for k in a.params:
        assert a.params[k].data == b.params[k].data, f"参数 {k} 不一致"


def test_initialization_differs_across_seeds():
    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    a = build_from_spec(spec, 12, seed=1, max_seq_len=5)
    b = build_from_spec(spec, 12, seed=2, max_seq_len=5)
    assert a.params["emb"].data != b.params["emb"].data


def test_init_stream_keyed_by_ir_and_name():
    """初始化种子 = ``(seed, IR 指纹, 参数名)``。

    **含 IR 指纹是有意的**：不同架构的初始化互相独立。否则两个架构会共享同一条
    随机流，它们就不是「两个独立样本」，比较出来的差异里混着初始化的相关性。

    含参数名则保证了「与遍历顺序无关」：加一个组件不会把后面所有参数打乱 ——
    否则每加一个组件，历史试验全部不可比。
    """
    spec1 = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    spec2 = ArchSpec("attention", "mlp", "rmsnorm", 2, 12, n_heads=2)
    m1 = build_from_spec(spec1, 12, seed=3, max_seq_len=5)
    m2 = build_from_spec(spec2, 12, seed=3, max_seq_len=5)
    # 同名参数在不同 IR 下互相独立
    assert m1.params["head"].data != m2.params["head"].data
    # 同一模型内不同参数不同
    assert m1.params["emb"].data != m1.params["head"].data
    # 同一 (seed, IR) 重建逐位一致
    m3 = build_from_spec(spec1, 12, seed=3, max_seq_len=5)
    assert m3.params["head"].data == m1.params["head"].data


def test_norm_weights_start_at_one_and_biases_at_zero():
    """归一化权重置一、偏置置零 —— 否则「归一化」在初始化时等于在缩放。"""
    spec = ArchSpec("attention", "swiglu", "layernorm", 1, 12, n_heads=2)
    m = build_from_spec(spec, 12, max_seq_len=5)
    assert m.params["block0.norm1.w"].data == [1.0] * 12
    assert m.params["block0.norm1.b"].data == [0.0] * 12
    assert m.params["block0.mixer.bq"].data == [0.0] * 12


def test_illegal_ir_refuses_to_build():
    """非法 IR 必须在装配前被拒 —— 报错信息要带原因。"""
    from architecturefactory.schema import ArchIR, BlockIR

    ir = ArchIR(
        vocab_size=12,
        d_model=12,
        blocks=[BlockIR("nope", "mlp", "rmsnorm", "pre", 12, 1, 0)],
    )
    with pytest.raises(ValueError, match="IR 非法"):
        build(ir)


# ----------------------------------------------------------------------
# 成本轴单调性
# ----------------------------------------------------------------------
def test_flops_grow_with_layers_and_width():
    base = estimate_flops(parse(ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2), 12), 5)
    deeper = estimate_flops(parse(ArchSpec("attention", "mlp", "rmsnorm", 2, 12, n_heads=2), 12), 5)
    wider = estimate_flops(parse(ArchSpec("attention", "mlp", "rmsnorm", 1, 16, n_heads=2), 12), 5)
    assert deeper > base
    assert wider > base


def test_attention_flops_quadratic_in_seq_len():
    """注意力对序列长度是二次的，ssm 是线性的 —— 这是两个 mixer 的成本分水岭。"""
    att = parse(ArchSpec("attention", "mlp", "rmsnorm", 1, 16, n_heads=2), 12)
    ssm = parse(ArchSpec("ssm", "mlp", "rmsnorm", 1, 16), 12)
    a2, a10 = estimate_flops(att, 2), estimate_flops(att, 10)
    s2, s10 = estimate_flops(ssm, 2), estimate_flops(ssm, 10)
    assert a10 / a2 > s10 / s2  # 注意力的增长更快


def test_activations_include_attention_matrix():
    att = parse(ArchSpec("attention", "mlp", "rmsnorm", 1, 16, n_heads=2), 12)
    ssm = parse(ArchSpec("ssm", "mlp", "rmsnorm", 1, 16), 12)
    assert estimate_activations(att, 9) > estimate_activations(ssm, 9)


def test_moe_costs_more_than_mlp_at_same_width():
    mlp = estimate_flops(parse(ArchSpec("attention", "mlp", "rmsnorm", 1, 16, n_heads=2), 12), 5)
    moe = estimate_flops(
        parse(ArchSpec("attention", "moe", "rmsnorm", 1, 16, n_heads=2, n_experts=2), 12), 5
    )
    assert moe > mlp
