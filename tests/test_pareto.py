"""帕累托：支配 / 前沿 / 超体积 / 拥挤距离 / 标量化。

超体积用**手算得出的已知值**验证，不用「跑一遍看看」。超体积是本项目对外
唯一的单一数字，它算错没人会立刻发现 —— 所以必须有独立可验的算例。
"""

from __future__ import annotations

import math

import pytest

from architecturefactory.pareto import (
    crowding_distance,
    dominates,
    frontier_ids,
    hypervolume,
    ideal_point,
    non_dominated_sort,
    random_weights,
    reference_point,
    scalarize,
)
from architecturefactory.schema import ArchSpec, Evaluation, Trial

OBJ = ("accuracy", "params")


def mk(spec_id: str, acc: float, params: int, flops: int = 1000) -> Trial:
    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    return Trial(
        trial_id=spec_id,
        spec=spec,
        evaluation=Evaluation(
            accuracy=acc, loss=1.0 - acc, params=params, flops=flops, train_seconds=0.0
        ),
        searcher="test",
    )


# ----------------------------------------------------------------------
# 支配
# ----------------------------------------------------------------------
def test_dominates_basic():
    # accuracy 越大越好、params 越小越好
    a = (0.9, 100)
    b = (0.8, 200)
    assert dominates(a, b, OBJ)
    assert not dominates(b, a, OBJ)


def test_dominates_requires_strict_improvement():
    """处处相等不算支配 —— 否则两个相同的点会互相支配，前沿直接空掉。"""
    assert not dominates((0.5, 100), (0.5, 100), OBJ)


def test_dominates_tradeoff_is_incomparable():
    assert not dominates((0.9, 200), (0.8, 100), OBJ)
    assert not dominates((0.8, 100), (0.9, 200), OBJ)


def test_dominates_tolerance_swallows_noise():
    """``tol`` 内的差异视为「不差」—— 浮点噪声不该改变前沿成员。"""
    assert not dominates((0.5, 100), (0.5 - 1e-12, 100), OBJ, tol=1e-6)


# ----------------------------------------------------------------------
# 前沿
# ----------------------------------------------------------------------
def test_frontier_is_the_tradeoff_curve():
    ts = [mk("a", 0.9, 300), mk("b", 0.8, 200), mk("c", 0.7, 100), mk("d", 0.5, 400)]
    # d 被 a 支配（更差且更贵）
    assert frontier_ids(ts, OBJ) == ["a", "b", "c"]


def test_frontier_deterministic_order():
    ts = [mk("c", 0.7, 100), mk("a", 0.9, 300), mk("b", 0.8, 200)]
    assert frontier_ids(ts, OBJ) == ["a", "b", "c"]


def test_frontier_empty_input():
    assert frontier_ids([], OBJ) == []


def test_non_dominated_sort_ranks():
    ts = [mk("a", 0.9, 300), mk("b", 0.8, 200), mk("c", 0.7, 100), mk("d", 0.5, 400)]
    fronts = non_dominated_sort(ts, OBJ)
    assert set(fronts[0]) == {"a", "b", "c"}
    assert fronts[1] == ["d"]


def test_crowding_distance_boundaries_are_infinite():
    ts = [mk("a", 0.9, 300), mk("b", 0.8, 200), mk("c", 0.7, 100)]
    d = crowding_distance(ts, OBJ)
    assert d["a"] == math.inf and d["c"] == math.inf
    assert 0.0 <= d["b"] < math.inf


# ----------------------------------------------------------------------
# 超体积（手算校验）
# ----------------------------------------------------------------------
def test_hypervolume_single_point_hand_computed():
    """一个点 (acc=0.5, params=100)，参考点 (0.4, 150)。

    最小化坐标：点 (-0.5, 100)，参考 (-0.4, 150)。
    被支配区域 = (0.5-0.4) × (150-100) = 0.1 × 50 = 5。
    """
    hv = hypervolume([(0.5, 100)], (0.4, 150), OBJ)
    assert hv == pytest.approx(5.0)


def test_hypervolume_two_points_hand_computed():
    """两点 (0.5,100) 与 (0.6,120)，参考 (0.4,150)。

    最小化：A=(-0.5,100)，B=(-0.6,120)，ref=(-0.4,150)。
    A 的支配盒：x∈[-0.5,-0.4] 宽 0.1，y∈[100,150] 高 50 → 5
    B 的支配盒：x∈[-0.6,-0.4] 宽 0.2，y∈[120,150] 高 30 → 6
    重叠：x∈[-0.5,-0.4] × y∈[120,150] → 0.1 × 30 = 3
    并集 = 5 + 6 - 3 = **8**
    """
    hv = hypervolume([(0.5, 100), (0.6, 120)], (0.4, 150), OBJ)
    assert hv == pytest.approx(8.0)


def test_hypervolume_dominated_point_adds_nothing():
    """加一个被支配的点，超体积必须不变。"""
    base = hypervolume([(0.5, 100), (0.6, 120)], (0.4, 150), OBJ)
    with_dominated = hypervolume([(0.5, 100), (0.6, 120), (0.45, 130)], (0.4, 150), OBJ)
    assert with_dominated == pytest.approx(base)


def test_hypervolume_zero_when_ref_dominates():
    assert hypervolume([(0.5, 100)], (0.9, 50), OBJ) == pytest.approx(0.0)


def test_hypervolume_monotone_in_frontier():
    """多一个非支配点，超体积只增不减。"""
    a = hypervolume([(0.5, 100)], (0.4, 150), OBJ)
    b = hypervolume([(0.5, 100), (0.7, 140)], (0.4, 150), OBJ)
    assert b > a


def test_hypervolume_3d_is_positive_and_monotone():
    """三维（accuracy, params, flops）也要能算 —— 目标数不是写死的 2。"""
    objs = ("accuracy", "params", "flops")
    ref = (0.4, 150, 2000)
    a = hypervolume([(0.5, 100, 1000)], ref, objs)
    b = hypervolume([(0.5, 100, 1000), (0.7, 140, 1500)], ref, objs)
    assert a > 0 and b > a


# ----------------------------------------------------------------------
# 参考点 / 理想点
# ----------------------------------------------------------------------
def test_reference_point_is_worse_than_all():
    ts = [mk("a", 0.9, 300), mk("b", 0.7, 100)]
    r = reference_point(ts, OBJ)
    assert r[0] < 0.7          # accuracy 方向：比最差的还差
    assert r[1] > 300          # params 方向：比最贵的还贵


def test_reference_point_handles_zero_range():
    """极差为 0 时参考点不能与前沿重合（否则超体积恒为 0）。"""
    ts = [mk("a", 0.5, 100), mk("b", 0.5, 100)]
    r = reference_point(ts, OBJ)
    assert r[0] < 0.5 and r[1] > 100


def test_ideal_point():
    ts = [mk("a", 0.9, 300), mk("b", 0.7, 100)]
    assert ideal_point(ts, OBJ) == (0.9, 100)


# ----------------------------------------------------------------------
# 标量化
# ----------------------------------------------------------------------
def test_scalarize_prefers_balanced_point():
    """Tchebycheff：在理想点固定的情况下，离理想点越近分数越低。"""
    ideal = (1.0, 0)
    close = mk("close", 0.9, 100)
    far = mk("far", 0.5, 500)
    w = (0.5, 0.5)
    assert scalarize(close, OBJ, w, ideal) < scalarize(far, OBJ, w, ideal)


def test_scalarize_zero_at_ideal():
    ideal = (0.9, 100)
    t = mk("t", 0.9, 100)
    assert scalarize(t, OBJ, (0.5, 0.5), ideal) == pytest.approx(0.0)


def test_random_weights_on_simplex():
    from architecturefactory.rng import make_rng

    rng = make_rng(1, "w")
    for _ in range(20):
        w = random_weights(rng, 3)
        assert len(w) == 3
        assert all(v > 0 for v in w)
        assert sum(w) == pytest.approx(1.0)


def test_random_weights_deterministic():
    from architecturefactory.rng import make_rng

    a = random_weights(make_rng(9, "w"), 3)
    b = random_weights(make_rng(9, "w"), 3)
    assert a == b
