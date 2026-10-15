"""ArchSpec / ArchIR：可逆 id、空间展开、结构校验、参数公式。"""

from __future__ import annotations

from architecturefactory import ir
from architecturefactory.schema import ArchSpec


def test_spec_id_roundtrip():
    specs = [
        ArchSpec("attention", "swiglu", "rmsnorm", 2, 24, n_heads=4),
        ArchSpec("ssm", "mlp", "layernorm", 3, 16),
        ArchSpec("hybrid", "moe", "rmsnorm", 2, 32, n_heads=4, n_experts=4),
        ArchSpec("attention", "moe", "layernorm", 1, 24, n_heads=2, n_experts=2),
    ]
    for s in specs:
        sid = s.spec_id
        assert ArchSpec.from_spec_id(sid).spec_id == sid


def test_canonical_collapses_irrelevant_fields():
    # ssm 用不到 n_heads、mlp 用不到 n_experts ⇒ 归一后同 id
    a = ArchSpec("ssm", "mlp", "rmsnorm", 2, 24, n_heads=8, n_experts=9)
    b = ArchSpec("ssm", "mlp", "rmsnorm", 2, 24)
    assert a.spec_id == b.spec_id
    assert a.canonical().n_heads == 1
    assert a.canonical().n_experts == 0


def test_hybrid_mixer_alternates():
    s = ArchSpec("hybrid", "mlp", "rmsnorm", 4, 24, n_heads=4)
    assert [s.mixer_at(i) for i in range(4)] == ["attention", "ssm", "attention", "ssm"]
    assert ArchSpec("attention", "mlp", "rmsnorm", 4, 24, n_heads=4).mixer_at(3) == "attention"


def test_parse_builds_ir():
    s = ArchSpec("hybrid", "moe", "rmsnorm", 3, 24, n_heads=4, n_experts=2)
    g = ir.parse(s, vocab_size=12)
    assert g.n_layers == 3
    assert [b.mixer for b in g.blocks] == ["attention", "ssm", "attention"]
    # attention 块带 heads，ssm 块归一为 1
    assert g.blocks[0].n_heads == 4
    assert g.blocks[1].n_heads == 1
    # 每个块都是 moe
    assert all(b.n_experts == 2 for b in g.blocks)


def test_ir_roundtrip():
    s = ArchSpec("attention", "swiglu", "rmsnorm", 2, 24, n_heads=4)
    g = ir.parse(s, vocab_size=12)
    g2 = type(g).from_dict(g.to_dict())
    assert g2.to_dict() == g.to_dict()
    assert g2.fingerprint() == g.fingerprint()


def test_validate_catches_bad_shapes():
    s = ArchSpec("attention", "mlp", "rmsnorm", 1, 24, n_heads=5)  # 24 % 5 != 0
    errs = ir.validate(ir.parse(s, 12))
    assert any("整除" in e for e in errs)

    bad = ir.parse(ArchSpec("ssm", "moe", "rmsnorm", 1, 24, n_experts=1), 12)
    assert any("专家" in e for e in ir.validate(bad))


def test_expand_space_is_deduped_and_deterministic():
    space = {
        "mixers": ["attention", "ssm"],
        "ffns": ["mlp", "moe"],
        "norms": ["rmsnorm"],
        "n_layers": [1, 2],
        "d_models": [16, 24],
        "n_heads": [2, 4],
        "n_experts": [2],
    }
    a = ir.expand_space(space, 12)
    b = ir.expand_space(space, 12)
    ids_a = [s.spec_id for s in a]
    assert ids_a == [s.spec_id for s in b]  # 顺序可复现
    assert len(ids_a) == len(set(ids_a))  # 无重复
    # attention 有 h2/h4（且 16、24 都能被 2、4 整除）；ssm 无 heads
    assert any("h4" in i for i in ids_a)
    assert all("h" not in i for i in ids_a if i.startswith("ssm"))


def test_param_estimate_is_monotone():
    small = ir.estimate_params(ir.parse(ArchSpec("attention", "mlp", "rmsnorm", 1, 16, n_heads=4), 12))
    big = ir.estimate_params(ir.parse(ArchSpec("attention", "mlp", "rmsnorm", 3, 32, n_heads=4), 12))
    assert big > small > 0


def test_canonical_demo_specs_are_four_distinct_structures():
    demo = ir.canonical_demo_specs()
    assert set(demo) == {"transformer", "mamba", "moe", "hybrid"}
    ids = {k: v.spec_id for k, v in demo.items()}
    assert len(set(ids.values())) == 4
