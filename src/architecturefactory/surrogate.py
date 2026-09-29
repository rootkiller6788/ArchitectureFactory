"""代理模型：Nadaraya–Watson 核回归（**GP 的零依赖替身**）。

贝叶斯优化要一个「便宜的预测器 + 不确定度」。标准答案是高斯过程 —— 但 GP 需要
矩阵求逆，在没有 numpy 的前提下要么手写 Cholesky（几十行，且数值不稳），要么放弃。

**Nadaraya–Watson 核回归**是这里的正确取舍：

    mu(x)   = Σ_i k(x, x_i)·y_i / Σ_i k(x, x_i)
    var(x)  = Σ_i k(x, x_i)·(y_i - mu(x))² / Σ_i k(x, x_i)

它是「局部加权平均」，天然给出一个不确定度代理（加权方差），不需要求逆、
不需要拟合超参、逐位可复现。代价是：它不能外推（远离数据点时方差不会真的变大），
所以 :func:`lcb` 里额外加了一项「距离惩罚」把探索压力补回来 —— 这一点在
README 附录 C 里写明了，因为它是一个**已知的近似**，不是「GP」。

特征编码必须**定长**：架构是变长的（hybrid 的 n_heads 有意义、ssm 的没有），
所以先把 :class:`ArchSpec` 展成一个固定维度的向量（类别用 one-hot，数值用归一化）。
编码一变，历史数据就不可比 —— 编码由 ``FEATURE_VERSION`` 标记。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .pareto import ideal_point, random_weights, scalarize
from .schema import (
    ATTENTIVE_MIXERS,
    FFNS,
    MIXERS,
    NORMS,
    ArchSpec,
    Trial,
)

__all__ = ["SurrogateModel", "fit", "lcb", "encode", "FEATURE_VERSION"]

#: 特征编码版本。改了编码 = 历史拟合不可比，必须换版本。
FEATURE_VERSION = 1

#: 类别字段的取值表（顺序固定 —— 特征向量必须稳定）
_CAT = (
    ("mixer", MIXERS),
    ("ffn", FFNS),
    ("norm", NORMS),
)


def encode(spec: ArchSpec) -> tuple[float, ...]:
    """``ArchSpec → 定长特征向量``。类别 one-hot + 数值归一化。"""
    s = spec.canonical()
    feats: list[float] = []
    for field, values in _CAT:
        v = getattr(s, field)
        feats.extend(1.0 if v == opt else 0.0 for opt in values)
    # 数值字段：各自除一个固定的量级，保证不同字段贡献可比
    feats.append(s.n_layers / 3.0)
    feats.append(s.d_model / 32.0)
    feats.append(s.n_heads / 4.0 if s.mixer in ATTENTIVE_MIXERS else 0.0)
    feats.append(s.n_experts / 4.0 if s.ffn == "moe" else 0.0)
    return tuple(feats)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


@dataclass
class SurrogateModel:
    """拟合好的代理。``xs``/``ys`` 是训练点（特征 + 标量化目标）。"""

    xs: list[tuple[float, ...]]
    ys: list[float]
    bandwidth: float
    source_trials: list[Trial]

    def predict(self, spec: ArchSpec) -> tuple[float, float, float]:
        """返回 ``(mu, sigma, 到最近训练点的距离)``。"""
        x = encode(spec)
        h = self.bandwidth
        num = 0.0
        den = 0.0
        nearest = math.inf
        for xi, yi in zip(self.xs, self.ys):
            d2 = sum((p - q) * (p - q) for p, q in zip(x, xi))
            if d2 < nearest:
                nearest = d2
            w = math.exp(-d2 / (2.0 * h * h))
            num += w * yi
            den += w
        if den <= 0.0:
            mu = sum(self.ys) / len(self.ys) if self.ys else 0.0
            return mu, 0.0, math.sqrt(nearest)
        mu = num / den
        var = 0.0
        for xi, yi in zip(self.xs, self.ys):
            d2 = sum((p - q) * (p - q) for p, q in zip(x, xi))
            w = math.exp(-d2 / (2.0 * h * h))
            var += w * (yi - mu) * (yi - mu)
        var /= den
        return mu, math.sqrt(max(var, 0.0)), math.sqrt(nearest)


def fit(
    history: Sequence[Trial], objectives: Sequence[str], bandwidth: float = 0.6
) -> SurrogateModel:
    """用历史试验拟合代理。

    标量化用**固定的**一组权重（取 ``seed`` 无关的确定性权重）—— 若每次 fit
    都随机换权重，代理的目标就在漂移，LCB 的排序没有意义。权重的作用是把多目标
    压成单目标；多目标的多样性由**前沿**去维护，不是由代理去维护。
    """
    obj = tuple(objectives)
    ideal = ideal_point(history, obj)
    # 确定性权重：均匀权重 + 一点倾斜，避免某个目标权重为 0
    n = len(obj)
    weights = tuple((i + 1.0) / (n * (n + 1) / 2.0) for i in range(n))
    xs = [encode(t.spec) for t in history]
    ys = [scalarize(t, obj, weights, ideal) for t in history]
    # 带宽：特征维度的平均最近邻距离 × 一个常数。太窄 = 只信最近点，太宽 = 全都平均
    if len(xs) > 1:
        nn = []
        for i, x in enumerate(xs):
            d = min(_distance(x, y) for j, y in enumerate(xs) if j != i)
            nn.append(d)
        scale = sum(nn) / len(nn)
        h = max(bandwidth * scale, 1e-3)
    else:
        h = bandwidth
    return SurrogateModel(xs=xs, ys=ys, bandwidth=h, source_trials=list(history))


def lcb(model: SurrogateModel, spec: ArchSpec, kappa: float) -> float:
    """下置信界（越小越值得试）：``mu - kappa·sigma - 距离惩罚``。

    距离惩罚这一项是**对 NW 回归不能外推的补偿**：NW 在远离数据处会退化成
    全局平均（方差反而变小），于是纯 LCB 会反复推荐「和已知点一样的点」。
    减去 ``kappa·距离`` 把「没探过的地方」重新变得有吸引力。
    """
    mu, sigma, nearest = model.predict(spec)
    return mu - kappa * sigma - kappa * nearest


def random_scalarization(
    history: Sequence[Trial], objectives: Sequence[str], rng: Any
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """随机权重 + 理想点。给 ParEGO 式的多目标进化用（当前未接进搜索器，留作扩展点）。"""
    return random_weights(rng, len(objectives)), ideal_point(history, objectives)
