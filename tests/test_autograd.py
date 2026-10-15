"""autograd 引擎：每个算子都要过「解析梯度 == 数值梯度」。

梯度写错**不会报错**，只会让训练静默变差 —— 所以这一层必须用中心差分逐元素核对，
而不是靠「训练 loss 降了」这种间接信号。
"""

from __future__ import annotations

import math
import random

import pytest

from architecturefactory import autograd as ag


def _sum_all(x):
    if x and isinstance(x[0], list):
        return sum(sum(r) for r in x)
    return sum(x)


def _iter_slots(x):
    if x and isinstance(x[0], list):
        for i in range(len(x)):
            for j in range(len(x[i])):
                yield i, j
    else:
        for i in range(len(x)):
            yield i, None


def check_grad(fn, tensors, tol=2e-4, h=1e-5):
    """核对 ``sum(fn(*tensors))`` 对每个输入元素的梯度。"""
    out = fn(*tensors)
    for t in tensors:
        t.zero_grad()
    out.backward()

    def total():
        return _sum_all(fn(*tensors).data)

    for t in tensors:
        for i, j in _iter_slots(t.data):
            orig = t.data[i] if j is None else t.data[i][j]
            if j is None:
                t.data[i] = orig + h
                fp = total()
                t.data[i] = orig - h
                fm = total()
                t.data[i] = orig
                ng = (fp - fm) / (2 * h)
                got = t.grad[i]
            else:
                t.data[i][j] = orig + h
                fp = total()
                t.data[i][j] = orig - h
                fm = total()
                t.data[i][j] = orig
                ng = (fp - fm) / (2 * h)
                got = t.grad[i][j]
            assert abs(ng - got) < tol, f"梯度不符 @{i},{j}: 数值={ng:.6f} 解析={got:.6f}"


def rnd2(m, n, rng, lo=-1.0, hi=1.0):
    return [[rng.uniform(lo, hi) for _ in range(n)] for _ in range(m)]


def rnd1(n, rng, lo=-1.0, hi=1.0):
    return [rng.uniform(lo, hi) for _ in range(n)]


@pytest.fixture
def rng():
    return random.Random(7)


def test_matmul(rng):
    a, b = ag.constant(rnd2(3, 4, rng)), ag.constant(rnd2(4, 2, rng))
    check_grad(lambda x, y: ag.matmul(x, y), [a, b])


def test_matvec(rng):
    v, w = ag.constant(rnd1(4, rng)), ag.constant(rnd2(4, 3, rng))
    check_grad(lambda x, y: ag.matvec(x, y), [v, w])


def test_transpose(rng):
    a = ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x: ag.transpose(x), [a])


def test_add_and_mul(rng):
    a, b = ag.constant(rnd2(3, 4, rng)), ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x, y: ag.add(x, y), [a, b])
    a2, b2 = ag.constant(rnd2(3, 4, rng)), ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x, y: ag.mul(x, y), [a2, b2])
    v, w = ag.constant(rnd1(3, rng)), ag.constant(rnd1(3, rng))
    check_grad(lambda x, y: ag.add(x, y), [v, w])


def test_add_bias(rng):
    a, b = ag.constant(rnd2(3, 4, rng)), ag.constant(rnd1(4, rng))
    check_grad(lambda x, y: ag.add_bias(x, y), [a, b])


def test_scale_rowscale(rng):
    a = ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x: ag.scale(x, 2.5), [a])
    a2, v = ag.constant(rnd2(3, 4, rng)), ag.constant(rnd1(3, rng))
    check_grad(lambda x, y: ag.row_scale(x, y), [a2, v])


def test_take_col_row(rng):
    a = ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x: ag.take_col(x, 2), [a])
    a2 = ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x: ag.take_row(x, 1), [a2])


def test_split_concat(rng):
    a = ag.constant(rnd2(3, 6, rng))
    check_grad(lambda x: ag.concat_cols(ag.split_cols(x, [2, 4])), [a])
    a2 = ag.constant(rnd2(3, 6, rng))
    parts = [ag.constant(rnd2(3, 2, rng)), ag.constant(rnd2(3, 4, rng))]
    check_grad(lambda p, q: ag.concat_cols([p, q]), parts)


def test_stack_sum_mean(rng):
    vs = [ag.constant(rnd1(4, rng)) for _ in range(3)]
    check_grad(lambda *xs: ag.stack_rows(list(xs)), vs)
    a = ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x: ag.sum_rows(x), [a])
    a2 = ag.constant(rnd2(3, 4, rng))
    check_grad(lambda x: ag.mean_rows(x), [a2])


def test_activations(rng):
    a = ag.constant(rnd2(3, 4, rng, -3, 3))
    check_grad(lambda x: ag.sigmoid(x), [a])
    a2 = ag.constant(rnd2(3, 4, rng, -3, 3))
    check_grad(lambda x: ag.silu(x), [a2])
    a3 = ag.constant(rnd2(3, 4, rng, -3, 3))
    check_grad(lambda x: ag.gelu(x), [a3])


def test_layernorm(rng):
    a = ag.constant(rnd2(4, 5, rng))
    w, b = ag.constant(rnd1(5, rng)), ag.constant(rnd1(5, rng))
    check_grad(lambda x, y, z: ag.layernorm(x, y, z, 1e-5), [a, w, b])


def test_rmsnorm(rng):
    a = ag.constant(rnd2(4, 5, rng))
    w = ag.constant(rnd1(5, rng))
    check_grad(lambda x, y: ag.rmsnorm(x, y, 1e-6), [a, w])


def test_embed(rng):
    w = ag.constant(rnd2(6, 3, rng))
    ids = [1, 4, 4, 0]
    check_grad(lambda x: ag.embed(x, ids), [w])


def test_softmax_rows(rng):
    a = ag.constant(rnd2(3, 5, rng))
    check_grad(lambda x: ag.softmax_rows(x), [a])
    a2 = ag.constant(rnd2(3, 5, rng))
    mask = [[j > i for j in range(5)] for i in range(3)]
    check_grad(lambda x: ag.softmax_rows(x, mask), [a2])


def test_softmax_cross_entropy(rng):
    logits = ag.constant(rnd2(4, 6, rng))
    targets = [1, 5, 0, 3]
    check_grad(lambda x: ag.softmax_cross_entropy(x, targets), [logits])


def test_softmax_rows_sums_to_one(rng):
    a = ag.constant(rnd2(3, 5, rng))
    p = ag.softmax_rows(a).data
    for row in p:
        assert abs(sum(row) - 1.0) < 1e-9
