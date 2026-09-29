"""帕累托：支配 / 非支配排序 / 拥挤距离 / 超体积 / 标量化。

**这是本项目的产物形态。** 搜索的输出不是「最好的那个架构」，而是**一条前沿**：
在 (质量, 成本) 平面上互不支配的一组架构。理由是显而易见的 —— 「最好」需要一个
偏好权重，而偏好是使用者的输入、不是搜索的结论。搜索该给的是**所有不被淘汰的选项**。

坐标系约定：所有目标在内部一律转成**最小化**（``max`` 方向的取负），
这样支配判断、超体积、标量化都只有一套代码。方向表来自 :data:`schema.METRIC_DIRECTIONS`。

参考点：超体积需要一个「比所有点都差」的点。取
``ref_i = min_i - pad``（max 目标）或 ``max_i + pad``（min 目标），``pad`` 默认 5% 极差。
**参考点必须记录进产物** —— 否则两次搜索的超体积数字不可比。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .schema import METRIC_DIRECTIONS, Trial

__all__ = [
    "objective_vector",
    "dominates",
    "frontier_ids",
    "non_dominated_sort",
    "crowding_distance",
    "reference_point",
    "hypervolume",
    "scalarize",
    "random_weights",
    "ideal_point",
]


# ----------------------------------------------------------------------
# 目标向量
# ----------------------------------------------------------------------
def objective_vector(trial: Trial, objectives: Sequence[str]) -> tuple[float, ...]:
    """按 ``objectives`` 取出原始目标值（**方向未归一**）。"""
    return tuple(trial.metric(o) for o in objectives)


def _to_minimize(
    vec: Sequence[float], objectives: Sequence[str]
) -> tuple[float, ...]:
    """转成最小化坐标：``max`` 方向取负。"""
    return tuple(
        -v if METRIC_DIRECTIONS[o] == "max" else v for v, o in zip(vec, objectives)
    )


# ----------------------------------------------------------------------
# 支配 / 前沿
# ----------------------------------------------------------------------
def dominates(
    a: Sequence[float],
    b: Sequence[float],
    objectives: Sequence[str],
    tol: float = 0.0,
) -> bool:
    """``a`` 是否支配 ``b``（在**最小化**坐标下）：处处不差，且至少一处严格更好。

    ``tol`` 给数值噪声留余量：小于 ``tol`` 的差视为「不差」。浮点相等判断必须
    留余量，否则同一组数在两次运行里可能给出不同的前沿（见 skill 坑 21）。
    """
    am = _to_minimize(a, objectives)
    bm = _to_minimize(b, objectives)
    strictly_better = False
    for x, y in zip(am, bm):
        if x > y + tol:
            return False
        if x < y - tol:
            strictly_better = True
    return strictly_better


def frontier_ids(
    trials: Iterable[Trial], objectives: Sequence[str], tol: float = 0.0
) -> list[str]:
    """非支配集合。**输出按 (第一个目标从优到劣, trial_id) 排序** —— 稳定可复现。"""
    ts = list(trials)
    if not ts:
        return []
    vecs = {t.trial_id: objective_vector(t, objectives) for t in ts}
    out: list[str] = []
    for t in ts:
        v = vecs[t.trial_id]
        dominated = False
        for u in ts:
            if u.trial_id == t.trial_id:
                continue
            if dominates(vecs[u.trial_id], v, objectives, tol):
                dominated = True
                break
        if not dominated:
            out.append(t.trial_id)
    first = objectives[0]
    reverse = METRIC_DIRECTIONS[first] == "max"
    out.sort(key=lambda tid: (-vecs[tid][0] if reverse else vecs[tid][0], tid))
    return out


def non_dominated_sort(
    trials: Sequence[Trial], objectives: Sequence[str], tol: float = 0.0
) -> list[list[str]]:
    """NSGA-II 快速非支配排序，返回 ``[[rank0 的 id], [rank1 的 id], ...]``。"""
    n = len(trials)
    vecs = [objective_vector(t, objectives) for t in trials]
    ids = [t.trial_id for t in trials]
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    count = [0] * n
    fronts: list[list[int]] = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(vecs[i], vecs[j], objectives, tol):
                dominated_by[i].append(j)
            elif dominates(vecs[j], vecs[i], objectives, tol):
                count[i] += 1
        if count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt: list[int] = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [[ids[i] for i in front] for front in fronts if front]


def crowding_distance(
    trials: Sequence[Trial], objectives: Sequence[str]
) -> dict[str, float]:
    """NSGA-II 拥挤距离。边界点记为 ``inf`` —— 它们必须被保留。"""
    n = len(trials)
    dist = {t.trial_id: 0.0 for t in trials}
    if n == 0:
        return dist
    if n <= 2:
        return {t.trial_id: math.inf for t in trials}
    for o in objectives:
        ordered = sorted(trials, key=lambda t: t.metric(o))
        lo = ordered[0].metric(o)
        hi = ordered[-1].metric(o)
        dist[ordered[0].trial_id] = math.inf
        dist[ordered[-1].trial_id] = math.inf
        span = hi - lo
        if span <= 0.0:
            continue
        for k in range(1, n - 1):
            prev_v = ordered[k - 1].metric(o)
            next_v = ordered[k + 1].metric(o)
            dist[ordered[k].trial_id] += (next_v - prev_v) / span
    return dist


# ----------------------------------------------------------------------
# 参考点 / 超体积
# ----------------------------------------------------------------------
def reference_point(
    trials: Sequence[Trial], objectives: Sequence[str], pad: float = 0.05
) -> tuple[float, ...]:
    """「比所有点都差」的参考点（**原始坐标**，方向未归一）。

    ``pad`` 是极差的比例；极差为 0 时退化为一个绝对小量，避免参考点与前沿重合
    导致超体积恒为 0。
    """
    if not trials:
        return tuple(0.0 for _ in objectives)
    out: list[float] = []
    for o in objectives:
        vals = [t.metric(o) for t in trials]
        lo, hi = min(vals), max(vals)
        p = (hi - lo) * pad if hi > lo else max(abs(hi) * pad, 1e-9)
        out.append(lo - p if METRIC_DIRECTIONS[o] == "max" else hi + p)
    return tuple(out)


def hypervolume(
    points: Sequence[Sequence[float]],
    ref: Sequence[float],
    objectives: Sequence[str],
) -> float:
    """超体积（**最小化坐标**，递归切片，支持任意维）。

    ``points`` 用原始坐标传入，``ref`` 用原始坐标传入；内部统一转到最小化。
    递归：按最后一个坐标升序扫，每段厚度乘上「剩余维度的子超体积」。
    """
    if not points:
        return 0.0
    pts = [_to_minimize(p, objectives) for p in points]
    r = _to_minimize(ref, objectives)
    return _hv_recursive(pts, r)


def _hv_recursive(pts: list[tuple[float, ...]], ref: tuple[float, ...]) -> float:
    kept = [p for p in pts if all(p[i] < ref[i] for i in range(len(ref)))]
    if not kept:
        return 0.0
    dim = len(ref)
    if dim == 1:
        return ref[0] - min(p[0] for p in kept)
    kept.sort(key=lambda p: p[-1])
    volume = 0.0
    prev = ref[-1]
    for i in range(len(kept) - 1, -1, -1):
        z = kept[i][-1]
        if z < prev:
            sub = [p[:-1] for p in kept[: i + 1]]
            volume += (prev - z) * _hv_recursive(sub, ref[:-1])
            prev = z
    return volume


# ----------------------------------------------------------------------
# 标量化（给进化/贝叶斯搜索当「单一目标」用）
# ----------------------------------------------------------------------
def random_weights(rng: Any, n: int) -> tuple[float, ...]:
    """在单纯形上均匀采一组权重（Dirichlet(1,...,1) 的指数法）。"""
    e = [-math.log(max(rng.random(), 1e-12)) for _ in range(n)]
    s = sum(e)
    return tuple(v / s for v in e)


def scalarize(
    trial: Trial,
    objectives: Sequence[str],
    weights: Sequence[float],
    ideal: Sequence[float],
) -> float:
    """加权 Tchebycheff 标量化（**越小越好**）。

    ``max_i w_i·|f_i - z*_i|``。选 Tchebycheff 而不是加权和，是因为加权和只能
    取到凸前沿上的点 —— 而架构搜索的前沿经常是非凸的（一个便宜但略差的架构
    可能恰好落在凹处）。这是 ParEGO 类方法的标准做法。
    """
    v = objective_vector(trial, objectives)
    worst = 0.0
    for val, w, z, o in zip(v, weights, ideal, objectives):
        d = abs(val - z)
        worst = max(worst, w * d)
    return worst


def ideal_point(trials: Sequence[Trial], objectives: Sequence[str]) -> tuple[float, ...]:
    """理想点（每个目标各自最好的值，原始坐标）。"""
    if not trials:
        return tuple(0.0 for _ in objectives)
    out: list[float] = []
    for o in objectives:
        vals = [t.metric(o) for t in trials]
        out.append(max(vals) if METRIC_DIRECTIONS[o] == "max" else min(vals))
    return tuple(out)
