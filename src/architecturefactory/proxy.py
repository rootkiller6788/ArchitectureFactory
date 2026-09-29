"""代理训练：把「架构」变成一个可比较的 (质量, 成本) 点。

**为什么叫「代理」。** 真实系统会用完整训练 + 验证集来评价一个架构，代价极高。
搜索要跑几十上百次评估，只能用一个**廉价但同向**的代理：固定小步数、小批量的
训练，测出来的准确率当作「这个结构好不好」的排序信号。

代理的可信度取决于**所有候选共用同一套训练协议**（步数 / 学习率 / batch / 初始化
尺度 / 数据顺序全部由配置与种子决定）。这一点由 :func:`protocol_digest` 固化 ——
协议变了，历史试验就不可比了，必须换版本号。

实现：动量 SGD + 全局范数梯度裁剪。**手写**，因为要让「一步更新」逐位可复现，
并且不引入任何第三方依赖（见 :mod:`architecturefactory.autograd` 的说明）。
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

from .autograd import Tensor
from .builder import Model
from .rng import make_rng, sha256_text
from .schema import Evaluation
from .task import Dataset, Task, evaluate, train_step_loss

__all__ = ["TrainConfig", "train_proxy", "protocol_digest", "sgd_momentum_step"]

#: 不可逐位复现的字段（实测耗时）。见 schema.TIMING_METRICS。
NON_REPRODUCIBLE = ("train_seconds",)


class TrainConfig:
    """训练协议。**它不进版本摘要**（版本由任务+空间+种子决定），但进试验记录。"""

    __slots__ = (
        "steps",
        "lr",
        "batch_size",
        "momentum",
        "init_scale",
        "grad_clip",
        "eval_every",
    )

    def __init__(self, cfg: dict[str, Any]) -> None:
        p = cfg.get("proxy", {})
        self.steps = int(p.get("steps", 150))
        self.lr = float(p.get("lr", 0.05))
        self.batch_size = int(p.get("batch_size", 32))
        self.momentum = float(p.get("momentum", 0.9))
        self.init_scale = float(p.get("init_scale", 0.08))
        self.grad_clip = float(p.get("grad_clip", 1.0))
        self.eval_every = int(p.get("eval_every", 25))

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


def protocol_digest(cfg: dict[str, Any]) -> str:
    """训练协议的指纹。**改了它，历史试验就不可比** —— 必须换版本号。"""
    import json

    return sha256_text(json.dumps(TrainConfig(cfg).to_dict(), sort_keys=True))[:12]


# ----------------------------------------------------------------------
# 优化器：动量 SGD + 全局范数裁剪
# ----------------------------------------------------------------------
def _iter_flat(grad: list[Any]):
    """遍历梯度的所有元素（1D 或 2D），产出 ``(行索引, 列索引或 None, 值)``。"""
    if grad and isinstance(grad[0], list):
        for i, row in enumerate(grad):
            for j, v in enumerate(row):
                yield i, j, v
    else:
        for i, v in enumerate(grad):
            yield i, None, v


def _apply_flat(param: list[Any], idx: tuple[int, int | None], delta: float) -> None:
    i, j = idx
    if j is None:
        param[i] += delta
    else:
        param[i][j] += delta


def sgd_momentum_step(
    params: Sequence[Tensor],
    velocity: dict[int, list[Any]],
    lr: float,
    momentum: float,
    grad_clip: float,
) -> float:
    """一步更新。返回**裁剪前**的全局梯度范数（用于诊断训练是否稳定）。

    先按全局范数算一个统一缩放系数再更新 —— 逐参数各自裁剪会改变梯度方向。
    """
    total = 0.0
    for t in params:
        for _, _, g in _iter_flat(t.grad):
            total += g * g
    norm = math.sqrt(total)
    k = 1.0
    if grad_clip > 0.0 and norm > grad_clip:
        k = grad_clip / norm

    for t in params:
        key = id(t)
        vel = velocity.get(key)
        if vel is None:
            vel = _zeros_like(t.grad)
            velocity[key] = vel
        for i, j, g in _iter_flat(t.grad):
            if j is None:
                vel[i] = momentum * vel[i] + g * k
                t.data[i] -= lr * vel[i]
            else:
                vel[i][j] = momentum * vel[i][j] + g * k
                t.data[i][j] -= lr * vel[i][j]
    return norm


def _zeros_like(x: list[Any]) -> list[Any]:
    if x and isinstance(x[0], list):
        return [[0.0] * len(row) for row in x]
    return [0.0] * len(x)


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
def _accuracy(model: Model, ds: Dataset) -> float:
    acc, _ = evaluate(model, ds)
    return acc


def train_proxy(
    model: Model, task: Task, cfg: dict[str, Any], seed: int
) -> Evaluation:
    """在 ``task.train`` 上训练 ``model``，返回 :class:`Evaluation`。

    **一步 = 一个 batch 的一次更新**（标准 SGD，不是「攒完整个 epoch 再更新」）。
    攒 epoch 会做 ``n/batch`` 倍的重复前向反向，却只换来同样多的参数更新 ——
    纯粹浪费，而代理训练的全部预算都花在这里。

    ``params`` / ``flops`` 由调用方用 :mod:`architecturefactory.profile` 补齐 ——
    这里只负责「训练」与「质量」，成本是结构属性、不该由训练器算。

    ``curve`` 记录**测试准确率**在若干检查点的值（末点即最终准确率），
    是给报告画图用的；排序只认 ``accuracy`` 这一个标量。
    """
    tc = TrainConfig(cfg)
    params = model.parameter_list()
    velocity: dict[int, list[Any]] = {}
    curve: list[float] = []
    n = len(task.train)
    rng = make_rng(seed, "proxy-order")
    order = list(range(n))
    rng.shuffle(order)
    t0 = time.perf_counter()
    for step in range(tc.steps):
        start = (step * tc.batch_size) % n
        idxs = [order[(start + k) % n] for k in range(tc.batch_size)]
        model.zero_grad()
        loss, _, _ = train_step_loss(model, task.train, idxs)
        loss.backward()
        sgd_momentum_step(params, velocity, tc.lr, tc.momentum, tc.grad_clip)
        if tc.eval_every > 0 and (step + 1) % tc.eval_every == 0:
            curve.append(round(_accuracy(model, task.test), 6))
    elapsed = time.perf_counter() - t0
    acc, loss = evaluate(model, task.test)
    if not curve or curve[-1] != round(acc, 6):
        curve.append(round(acc, 6))
    return Evaluation(
        accuracy=acc,
        loss=loss,
        params=0,
        flops=0,
        train_seconds=elapsed,
        curve=curve,
    )
