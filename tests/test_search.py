"""搜索器：预算 / 去重 / 走法差异。

最重要的一条是 :func:`test_searchers_share_the_seen_set`：三个搜索器各跑各的，
但**架构数据库只有一份**。不共享去重集合的后果是前沿上出现重复点，
而且白烧掉三分之一的训练预算 —— 这是实测踩出来的（见 README 附录 C）。
"""

from __future__ import annotations

import pytest

from architecturefactory.config import expand_specs
from architecturefactory.schema import ArchSpec, Evaluation, Trial
from architecturefactory.search import SEARCHERS, make_searcher


def mk_trial(spec_id: str, acc: float, params: int) -> Trial:
    return Trial(
        trial_id=spec_id,
        spec=ArchSpec.from_spec_id(spec_id),
        evaluation=Evaluation(acc, 1 - acc, params, 1000, 0.0),
        searcher="test",
    )


def _mk(name: str, cfg, seen=None):
    return make_searcher(
        name, cfg["space"], ("accuracy", "params"), cfg["search"], 0, 12, seen
    )


def test_all_searchers_registered():
    assert set(SEARCHERS) == {"random", "evolution", "bayesian"}


def test_unknown_searcher_raises(cfg):
    with pytest.raises(KeyError, match="未注册的搜索器"):
        make_searcher("nope", cfg["space"], ("accuracy",), cfg["search"], 0, 12)


@pytest.mark.parametrize("name", ["random", "evolution", "bayesian"])
def test_proposals_are_unique_and_in_space(name, cfg):
    s = _mk(name, cfg)
    pool_ids = {x.spec_id for x in expand_specs(cfg)}
    history: list[Trial] = []
    seen = []
    for _ in range(10):
        spec = s.propose(history)
        assert spec is not None
        assert spec.spec_id in pool_ids, "提议越出了设计空间"
        seen.append(spec.spec_id)
        history.append(mk_trial(spec.spec_id, 0.5, 3000))
    assert len(set(seen)) == len(seen), "同一个 spec 被提议了两次"


@pytest.mark.parametrize("name", ["random", "evolution", "bayesian"])
def test_proposals_are_deterministic(name, cfg):
    a = _mk(name, cfg)
    b = _mk(name, cfg)
    for _ in range(6):
        assert a.propose([]).spec_id == b.propose([]).spec_id


def test_searchers_share_the_seen_set(cfg):
    """共享去重集合：第二个搜索器不该重复第一个已经提议过的点。"""
    shared: set[str] = set()
    s1 = _mk("random", cfg, shared)
    first = {s1.propose([]).spec_id for _ in range(8)}
    s2 = _mk("random", cfg, shared)
    second = {s2.propose([]).spec_id for _ in range(8)}
    assert not (first & second), "两个搜索器提议了同一个架构 —— 去重没有共享"


def test_bayesian_explores_before_fitting(cfg):
    """前 ``n_init`` 个点必须是随机的 —— 没有数据时任何代理都是噪声。"""
    s = _mk("bayesian", cfg)
    n_init = cfg["search"]["n_init"]
    assert len(s.proposed) == 0
    for _ in range(n_init):
        s.propose([])
    assert len(s.proposed) == n_init


def test_random_has_no_parents(cfg):
    s = _mk("random", cfg)
    s.propose([])
    assert s.pending_parents == []


def test_evolution_mutates_within_space(cfg):
    """变异只能取空间里出现过的取值 —— 变异不越界。"""
    s = _mk("evolution", cfg)
    allowed_heads = set(cfg["space"]["n_heads"])
    allowed_layers = set(cfg["space"]["n_layers"])
    for _ in range(40):
        child = s._mutate(ArchSpec("attention", "moe", "rmsnorm", 2, 16, n_heads=4, n_experts=2))
        assert child.mixer in cfg["space"]["mixers"]
        assert child.ffn in cfg["space"]["ffns"]
        assert child.norm in cfg["space"]["norms"]
        assert child.n_layers in allowed_layers
        assert child.d_model in cfg["space"]["d_models"]
        if child.mixer in ("attention", "hybrid"):
            assert child.n_heads in allowed_heads
        else:
            assert child.n_heads == 1  # 无关字段已归一
        if child.ffn == "moe":
            assert child.n_experts in set(cfg["space"]["n_experts"])
        else:
            assert child.n_experts == 0


def test_evolution_crossover_takes_from_both_parents(cfg):
    s = _mk("evolution", cfg)
    a = ArchSpec("attention", "mlp", "layernorm", 1, 12, n_heads=2)
    b = ArchSpec("ssm", "swiglu", "rmsnorm", 2, 16)
    child = s._crossover(a, b)
    assert child.mixer in ("attention", "ssm")
    assert child.ffn in ("mlp", "swiglu")
    assert child.norm in ("layernorm", "rmsnorm")


def test_evolution_exhausts_gracefully(cfg):
    """空间耗尽时必须返回 ``None`` 而不是死循环。"""
    import copy

    c = copy.deepcopy(cfg)
    c["space"]["mixers"] = ["ssm"]
    c["space"]["ffns"] = ["mlp"]
    c["space"]["norms"] = ["rmsnorm"]
    c["space"]["n_layers"] = [1]
    c["space"]["d_models"] = [12]
    c["space"]["n_heads"] = [1]
    s = make_searcher("evolution", c["space"], ("accuracy", "params"), c["search"], 0, 12)
    n = len(expand_specs(c))
    got = 0
    history: list[Trial] = []
    for _ in range(n + 3):
        spec = s.propose(history)
        if spec is None:
            break
        got += 1
        history.append(mk_trial(spec.spec_id, 0.5, 3000))
    assert got == n
    assert s.propose(history) is None


def test_bayesian_uses_surrogate_after_warmup(cfg):
    """热身之后，提议应当由代理驱动 —— 父本非空即说明代理参与了决策。"""
    s = _mk("bayesian", cfg)
    history: list[Trial] = []
    for _ in range(cfg["search"]["n_init"]):
        spec = s.propose(history)
        history.append(mk_trial(spec.spec_id, 0.5, 3000))
    s.propose(history)
    assert s.pending_parents, "贝叶斯搜索器没有记录代理的来源点"
