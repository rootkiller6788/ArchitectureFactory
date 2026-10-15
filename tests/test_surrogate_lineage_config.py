"""代理模型 / 血统图 / 配置摘要。"""

from __future__ import annotations

import copy
import math

import pytest

from architecturefactory.config import (
    expand_specs,
    protocol_tag,
    validate_config,
    version_digest,
)
from architecturefactory.lineage import build_lineage
from architecturefactory.schema import ArchSpec, Evaluation, Trial
from architecturefactory.surrogate import FEATURE_VERSION, encode, fit, lcb


# ----------------------------------------------------------------------
# 代理模型
# ----------------------------------------------------------------------
def test_feature_vector_is_fixed_length():
    """特征必须**定长** —— 架构是变长的（hybrid 有 n_heads、ssm 没有）。"""
    specs = [
        ArchSpec("attention", "swiglu", "rmsnorm", 1, 12, n_heads=2),
        ArchSpec("ssm", "mlp", "layernorm", 2, 16),
        ArchSpec("hybrid", "moe", "rmsnorm", 2, 16, n_heads=4, n_experts=2),
    ]
    lengths = {len(encode(s)) for s in specs}
    assert len(lengths) == 1
    assert FEATURE_VERSION == 1


def test_feature_vector_distinguishes_architectures():
    a = encode(ArchSpec("attention", "swiglu", "rmsnorm", 1, 12, n_heads=2))
    b = encode(ArchSpec("ssm", "swiglu", "rmsnorm", 1, 12))
    assert a != b


def _hist():
    out = []
    for i, (mixer, acc, params) in enumerate([
        ("attention", 0.9, 3000),
        ("ssm", 0.2, 3000),
        ("attention", 0.8, 2000),
        ("hybrid", 0.7, 2500),
        ("ssm", 0.25, 2800),
        ("attention", 0.6, 1800),
    ]):
        spec = ArchSpec(mixer, "swiglu", "rmsnorm", 1, 12, n_heads=2) if mixer != "ssm" \
            else ArchSpec("ssm", "swiglu", "rmsnorm", 1, 12)
        out.append(Trial(f"t{i}", spec, Evaluation(acc, 1 - acc, params, 1000, 0.0), "test"))
    return out


def test_fit_learns_the_mixer_effect():
    """代理必须抓住「attention 明显好于 ssm」这个主效应 —— 否则它指不出方向。"""
    model = fit(_hist(), ("accuracy", "params"))
    good = ArchSpec("attention", "swiglu", "rmsnorm", 1, 12, n_heads=2)
    bad = ArchSpec("ssm", "swiglu", "rmsnorm", 1, 12)
    mu_good, _, _ = model.predict(good)
    mu_bad, _, _ = model.predict(bad)
    assert mu_good < mu_bad  # 标量化是最小化，越小越好


def test_lcb_prefers_unexplored_region():
    """NW 回归不能外推，所以 LCB 里的距离惩罚必须让「没探过的地方」重新有吸引力。"""
    model = fit(_hist(), ("accuracy", "params"))
    explored = ArchSpec("attention", "swiglu", "rmsnorm", 1, 12, n_heads=2)
    far = ArchSpec("hybrid", "moe", "layernorm", 2, 16, n_heads=4, n_experts=2)
    _, _, d_explored = model.predict(explored)
    _, _, d_far = model.predict(far)
    assert d_far > d_explored


def test_lcb_is_deterministic():
    h = _hist()
    a = fit(h, ("accuracy", "params"))
    b = fit(h, ("accuracy", "params"))
    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    assert lcb(a, spec, 2.0) == lcb(b, spec, 2.0)


def test_fit_single_point_does_not_crash():
    model = fit(_hist()[:1], ("accuracy", "params"))
    mu, sigma, _ = model.predict(ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2))
    assert math.isfinite(mu) and math.isfinite(sigma)


# ----------------------------------------------------------------------
# 血统图
# ----------------------------------------------------------------------
def _trials_with_parents():
    a = Trial("a", ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2),
              Evaluation(0.9, 0.1, 3000, 1000, 0.0), "random")
    b = Trial("b", ArchSpec("ssm", "mlp", "rmsnorm", 1, 12),
              Evaluation(0.2, 0.8, 3000, 1000, 0.0), "evolution", parents=["a"])
    c = Trial("c", ArchSpec("hybrid", "mlp", "rmsnorm", 2, 12, n_heads=2),
              Evaluation(0.5, 0.5, 3000, 1000, 0.0), "bayesian", parents=["a", "b"])
    return [a, b, c]


def test_lineage_two_nodes_per_trial():
    lin = build_lineage(_trials_with_parents())
    assert len(lin.nodes) == 6
    assert sum(1 for n in lin.nodes if n.kind == "propose") == 3
    assert sum(1 for n in lin.nodes if n.kind == "evaluate") == 3


def test_lineage_edges():
    lin = build_lineage(_trials_with_parents())
    assert len([e for e in lin.edges if e.kind == "evaluates"]) == 3
    assert len([e for e in lin.edges if e.kind == "derives"]) == 3  # b←a, c←a, c←b


def test_lineage_random_search_is_shallow():
    """随机搜索没有 derives 边 —— 深度应当就是 2（propose → evaluate）。"""
    ts = [
        Trial(f"t{i}", ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2),
              Evaluation(0.5, 0.5, 3000, 1000, 0.0), "random")
        for i in range(4)
    ]
    lin = build_lineage(ts)
    assert lin.derive_edges() == []
    assert lin.depth() == 2


def test_lineage_evolution_is_deeper():
    lin = build_lineage(_trials_with_parents())
    assert lin.depth() > 2


def test_lineage_roundtrip_and_render():
    lin = build_lineage(_trials_with_parents())
    from architecturefactory.lineage import Lineage

    back = Lineage.from_dict(lin.to_dict())
    assert len(back.nodes) == len(lin.nodes)
    assert len(back.edges) == len(lin.edges)
    dot = lin.to_dot()
    assert dot.startswith("digraph") and "derives" not in dot  # 样式在 style 里
    assert "--derives-->" in lin.to_ascii()
    assert "lineage:" in lin.to_ascii()


def test_lineage_children_lookup():
    """``p:a`` 的后继有三类：自己的 evaluate 节点 + 以 a 为父本的两个 propose 节点。"""
    lin = build_lineage(_trials_with_parents())
    kids = {n.node_id for n in lin.children("p:a")}
    assert kids == {"e:a", "p:b", "p:c"}
    # 只看探索结构（derives 边）时，后代才是「血缘上的孩子」
    derives = {e.b for e in lin.derive_edges() if e.a == "p:a"}
    assert derives == {"p:b", "p:c"}


# ----------------------------------------------------------------------
# 配置摘要
# ----------------------------------------------------------------------
def test_version_digest_ignores_search_config(cfg):
    """**版本只由被搜索对象的定义决定。** 换搜索器/改预算不该换版本号。"""
    base = version_digest(cfg)
    c = copy.deepcopy(cfg)
    c["search"]["trials"] = 999
    c["search"]["searchers"] = ["random"]
    c["proxy"]["steps"] = 7
    assert version_digest(c) == base


def test_version_digest_changes_with_task_or_space(cfg):
    base = version_digest(cfg)
    c1 = copy.deepcopy(cfg)
    c1["task"]["n_items"] = 3
    assert version_digest(c1) != base
    c2 = copy.deepcopy(cfg)
    c2["space"]["n_layers"] = [1]
    assert version_digest(c2) != base
    c3 = copy.deepcopy(cfg)
    c3["seed"] = cfg["seed"] + 1
    assert version_digest(c3) != base


def test_protocol_tag_changes_with_proxy(cfg):
    """协议指纹反过来：**训练协议一变就必须变**，否则历史试验不可比。"""
    base = protocol_tag(cfg)
    c = copy.deepcopy(cfg)
    c["proxy"]["lr"] = 0.99
    assert protocol_tag(c) != base


def test_validate_config_rejects_unknown_objective(cfg):
    c = copy.deepcopy(cfg)
    c["objectives"] = ["accuracy", "nonsense"]
    with pytest.raises(ValueError, match="未知目标"):
        validate_config(c)


def test_validate_config_rejects_missing_direction(cfg):
    c = copy.deepcopy(cfg)
    c["metric_directions"].pop("params")
    with pytest.raises(ValueError, match="缺 metric_directions"):
        validate_config(c)


def test_validate_config_rejects_bad_direction(cfg):
    c = copy.deepcopy(cfg)
    c["metric_directions"]["params"] = "smaller"
    with pytest.raises(ValueError, match="必须是 max/min"):
        validate_config(c)


def test_validate_config_rejects_missing_section(cfg):
    c = copy.deepcopy(cfg)
    c.pop("proxy")
    with pytest.raises(ValueError, match="配置缺字段"):
        validate_config(c)


def test_space_expansion_is_deterministic_and_deduped(cfg):
    a = expand_specs(cfg)
    b = expand_specs(cfg)
    assert [s.spec_id for s in a] == [s.spec_id for s in b]
    ids = [s.spec_id for s in a]
    assert len(set(ids)) == len(ids)
    assert len(a) == 120
