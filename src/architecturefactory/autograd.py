"""微型张量 autograd 引擎（零第三方依赖）。

**为什么手写。** 本项目要回答「同一个 IR 空间里，哪种装配更好」。这需要一个能被
完全拆开、逐位复现、且**真的能训练**的观测对象。用 PyTorch 建模型，得到的是
「某个框架版本在某一天的行为」—— 梯度、初始化、算子实现都不可复核。生产环境
当然该用 PyTorch，因此留了 ``torch`` extra 做一致性对照。

**设计取舍。**

- 张量 = 嵌套 ``list``（1D: ``list[float]``；2D: ``list[list[float]]``）。够小、够快、无依赖。
- 反向传播 = 每个算子自带 backward 闭包 + 反向拓扑遍历（micrograd 的矢量化版本）。
- **只实现本项目组件用得到的算子**（见 ``__all__``），其余一律不提供 —— 少即是稳。
  每个算子都配一条「解析梯度 == 数值梯度」的测试（见 ``tests/test_autograd.py``），
  因为梯度写错**不会报错**，只会让训练静默变差。
- **内积一律走 :func:`math.sumprod`**（CPython 3.12+ 的标准库，C 实现）。
  纯 Python 三重循环的矩阵乘是这个项目唯一的性能瓶颈 —— 实测 matmul 的前向+反向
  占整条代理训练 **71%** 的时间，换成 ``sumprod`` 后前向快 2.6×、反向快 2.3×，
  而**没有引入任何第三方依赖**。这是本项目最重要的一处性能决策，见 README 附录 C。

约定：``Tensor.data`` 与 ``Tensor.grad`` 同形；叶子用 :func:`constant` 造。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

#: C 速度的内积。``math.sumprod`` 自 CPython 3.12 起提供（见模块 docstring）。
_dot = math.sumprod

__all__ = [
    "Tensor",
    "constant",
    "ones",
    "zeros",
    "matmul",
    "matvec",
    "add",
    "add_bias",
    "mul",
    "scale",
    "row_scale",
    "take_col",
    "take_row",
    "gather_rows",
    "transpose",
    "split_cols",
    "concat_cols",
    "stack_rows",
    "softmax_rows",
    "softmax_cross_entropy",
    "layernorm",
    "rmsnorm",
    "silu",
    "gelu",
    "sigmoid",
    "embed",
    "sum_rows",
    "mean_rows",
]


# ----------------------------------------------------------------------
# 形状工具
# ----------------------------------------------------------------------
def _is2d(x: Sequence[Any]) -> bool:
    return bool(x) and isinstance(x[0], list)


def _zeros_like(x: Sequence[Any]) -> list[Any]:
    if _is2d(x):
        return [[0.0] * len(row) for row in x]  # type: ignore[union-attr]
    return [0.0] * len(x)


def _ones_like(x: Sequence[Any]) -> list[Any]:
    if _is2d(x):
        return [[1.0] * len(row) for row in x]  # type: ignore[union-attr]
    return [1.0] * len(x)


def _shape(x: Sequence[Any]) -> tuple[int, ...]:
    if _is2d(x):
        return (len(x), len(x[0]))  # type: ignore[arg-type]
    return (len(x),)


# ----------------------------------------------------------------------
# 张量
# ----------------------------------------------------------------------
class Tensor:
    """一个值 + 它的梯度 + 它从哪些父张量算来。"""

    __slots__ = ("data", "grad", "_parents", "_backward")

    def __init__(
        self,
        data: list[Any],
        parents: tuple["Tensor", ...] = (),
        backward: Callable[[list[Any]], None] | None = None,
    ) -> None:
        self.data = data
        self.grad = _zeros_like(data)
        self._parents = parents
        self._backward = backward

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.data)

    def zero_grad(self) -> None:
        self.grad = _zeros_like(self.data)

    def backward(self) -> None:
        """从本张量出发做反向传播；本张量视作标量输出（grad=1）。"""
        topo: list[Tensor] = []
        seen: set[int] = set()

        def build(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        self.grad = _ones_like(self.data)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward(t.grad)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"Tensor(shape={self.shape})"


def constant(data: list[Any]) -> Tensor:
    """叶子张量（不接收梯度）。"""
    return Tensor(data)


def ones(n: int) -> Tensor:
    return Tensor([1.0] * n)


def zeros(n: int) -> Tensor:
    return Tensor([0.0] * n)


# ----------------------------------------------------------------------
# 线性代数
# ----------------------------------------------------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """``[m,k] @ [k,n] -> [m,n]``。

    前向与反向**都只由 :func:`math.sumprod` 的内积组成**（列优先取用，避免
    ``g[i][j]`` 这种跨行索引破坏局部性）。见模块 docstring 的性能说明。
    """
    A, B = a.data, b.data
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"matmul 维度不匹配：{_shape(A)} @ {_shape(B)}")
    bcols = list(zip(*B))
    C = [[_dot(A[i], bcols[j]) for j in range(n)] for i in range(m)]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        acols = list(zip(*A))
        gcols = list(zip(*g))
        # ga = g @ B^T  ⇒  ga[i][t] = ⟨g[i], B[t]⟩
        for i in range(m):
            gi = g[i]
            gai = a.grad[i]
            for t in range(k):
                gai[t] += _dot(gi, B[t])
        # gb = A^T @ g  ⇒  gb[t][j] = ⟨A[:,t], g[:,j]⟩
        for t in range(k):
            at = acols[t]
            gbt = b.grad[t]
            for j in range(n):
                gbt[j] += _dot(at, gcols[j])

    out._backward = bw
    return out


def matvec(v: Tensor, W: Tensor) -> Tensor:
    """``[k] @ [k,n] -> [n]``。"""
    V, Wd = v.data, W.data
    k, n = len(Wd), len(Wd[0])
    if len(V) != k:
        raise ValueError(f"matvec 维度不匹配：{len(V)} @ {_shape(Wd)}")
    wcols = list(zip(*Wd))
    y = [_dot(V, wcols[j]) for j in range(n)]
    out = Tensor(y, (v, W))

    def bw(g: list[Any]) -> None:
        for t in range(k):
            vt = V[t]
            wgt = W.grad[t]
            for j in range(n):
                wgt[j] += vt * g[j]
        for t in range(k):
            Wt = Wd[t]
            v.grad[t] += _dot(g, Wt)

    out._backward = bw
    return out


def transpose(a: Tensor) -> Tensor:
    A = a.data
    m, n = len(A), len(A[0])
    T = [[A[i][j] for i in range(m)] for j in range(n)]
    out = Tensor(T, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += g[j][i]

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 逐元素 / 广播
# ----------------------------------------------------------------------
def add(a: Tensor, b: Tensor) -> Tensor:
    """同形相加（2D 或 1D）。"""
    A, B = a.data, b.data
    if _is2d(A):
        m, n = len(A), len(A[0])
        C = [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]
    else:
        C = [A[i] + B[i] for i in range(len(A))]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            m, n = len(A), len(A[0])
            for i in range(m):
                for j in range(n):
                    a.grad[i][j] += g[i][j]
                    b.grad[i][j] += g[i][j]
        else:
            for i in range(len(A)):
                a.grad[i] += g[i]
                b.grad[i] += g[i]

    out._backward = bw
    return out


def add_bias(a: Tensor, b: Tensor) -> Tensor:
    """``[m,n] + [n]``（每行加同一个偏置）。"""
    A, B = a.data, b.data
    m, n = len(A), len(A[0])
    C = [[A[i][j] + B[j] for j in range(n)] for i in range(m)]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            gi = g[i]
            for j in range(n):
                a.grad[i][j] += gi[j]
                b.grad[j] += gi[j]

    out._backward = bw
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    """同形逐元素相乘。"""
    A, B = a.data, b.data
    if _is2d(A):
        m, n = len(A), len(A[0])
        C = [[A[i][j] * B[i][j] for j in range(n)] for i in range(m)]
    else:
        C = [A[i] * B[i] for i in range(len(A))]
    out = Tensor(C, (a, b))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            m, n = len(A), len(A[0])
            for i in range(m):
                for j in range(n):
                    a.grad[i][j] += g[i][j] * B[i][j]
                    b.grad[i][j] += g[i][j] * A[i][j]
        else:
            for i in range(len(A)):
                a.grad[i] += g[i] * B[i]
                b.grad[i] += g[i] * A[i]

    out._backward = bw
    return out


def scale(a: Tensor, s: float) -> Tensor:
    A = a.data
    if _is2d(A):
        C = [[v * s for v in row] for row in A]
    else:
        C = [v * s for v in A]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            for i, row in enumerate(g):
                for j, v in enumerate(row):
                    a.grad[i][j] += v * s
        else:
            for i, v in enumerate(g):
                a.grad[i] += v * s

    out._backward = bw
    return out


def row_scale(a: Tensor, v: Tensor) -> Tensor:
    """``[m,n]`` 的第 i 行乘标量 ``v[i]``。"""
    A, V = a.data, v.data
    m, n = len(A), len(A[0])
    C = [[A[i][j] * V[i] for j in range(n)] for i in range(m)]
    out = Tensor(C, (a, v))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            vi = V[i]
            s = 0.0
            for j in range(n):
                a.grad[i][j] += g[i][j] * vi
                s += g[i][j] * A[i][j]
            v.grad[i] += s

    out._backward = bw
    return out


def take_col(a: Tensor, j: int) -> Tensor:
    A = a.data
    m = len(A)
    C = [A[i][j] for i in range(m)]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            a.grad[i][j] += g[i]

    out._backward = bw
    return out


def take_row(a: Tensor, i: int) -> Tensor:
    A = a.data
    n = len(A[0])
    C = list(A[i])
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for j in range(n):
            a.grad[i][j] += g[j]

    out._backward = bw
    return out


def gather_rows(a: Tensor, idxs: Sequence[int]) -> Tensor:
    """按行索引取行：``[m,n] , [k] -> [k,n]``（允许重复、允许乱序）。

    监督只要最后一个位置时用 ``gather_rows(logits, [L-1])`` —— 于是 loss 的
    形状是 ``[1,V]``，不需要给 ``softmax_cross_entropy`` 塞一堆无意义的目标。
    """
    A = a.data
    n = len(A[0]) if A else 0
    C = [list(A[i]) for i in idxs]
    out = Tensor(C, (a,))
    k = len(idxs)

    def bw(g: list[Any]) -> None:
        for r in range(k):
            gi = a.grad[idxs[r]]
            gr = g[r]
            for j in range(n):
                gi[j] += gr[j]

    out._backward = bw
    return out


def split_cols(a: Tensor, sizes: Sequence[int]) -> list[Tensor]:
    """按列切成若干块。用于多头切分。"""
    A = a.data
    m = len(A)
    outs: list[Tensor] = []
    start = 0
    for size in sizes:
        C = [[A[i][start + c] for c in range(size)] for i in range(m)]
        out = Tensor(C, (a,))
        col0, width = start, size

        def make_bw(out_t: Tensor, c0: int, w: int) -> Callable[[list[Any]], None]:
            def bw(g: list[Any]) -> None:
                for i in range(m):
                    for c in range(w):
                        a.grad[i][c0 + c] += g[i][c]

            return bw

        out._backward = make_bw(out, col0, width)
        outs.append(out)
        start += size
    return outs


def concat_cols(ts: Sequence[Tensor]) -> Tensor:
    A0 = ts[0].data
    m = len(A0)
    C = [[] for _ in range(m)]
    for t in ts:
        for i in range(m):
            C[i].extend(t.data[i])
    out = Tensor(C, tuple(ts))

    def bw(g: list[Any]) -> None:
        start = 0
        for t in ts:
            n = len(t.data[0])
            for i in range(m):
                for c in range(n):
                    t.grad[i][c] += g[i][start + c]
            start += n

    out._backward = bw
    return out


def stack_rows(vs: Sequence[Tensor]) -> Tensor:
    C = [list(v.data) for v in vs]
    out = Tensor(C, tuple(vs))
    n = len(C[0]) if C else 0

    def bw(g: list[Any]) -> None:
        for i, v in enumerate(vs):
            for j in range(n):
                v.grad[j] += g[i][j]

    out._backward = bw
    return out


def sum_rows(a: Tensor) -> Tensor:
    A = a.data
    m, n = len(A), len(A[0])
    C = [0.0] * n
    for i in range(m):
        for j in range(n):
            C[j] += A[i][j]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += g[j]

    out._backward = bw
    return out


def mean_rows(a: Tensor) -> Tensor:
    A = a.data
    m, n = len(A), len(A[0])
    C = [0.0] * n
    for i in range(m):
        for j in range(n):
            C[j] += A[i][j]
    C = [v / m for v in C]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += g[j] / m

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 激活
# ----------------------------------------------------------------------
def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid(a: Tensor) -> Tensor:
    A = a.data
    if _is2d(A):
        S = [[_sigmoid_scalar(v) for v in row] for row in A]
    else:
        S = [_sigmoid_scalar(v) for v in A]
    out = Tensor(S, (a,))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            for i in range(len(A)):
                for j in range(len(A[0])):
                    s = S[i][j]
                    a.grad[i][j] += g[i][j] * s * (1.0 - s)
        else:
            for i in range(len(A)):
                s = S[i]
                a.grad[i] += g[i] * s * (1.0 - s)

    out._backward = bw
    return out


def silu(a: Tensor) -> Tensor:
    """``x * sigmoid(x)``。"""
    A = a.data
    if _is2d(A):
        S = [[_sigmoid_scalar(v) for v in row] for row in A]
        C = [[A[i][j] * S[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    else:
        S = [_sigmoid_scalar(v) for v in A]
        C = [A[i] * S[i] for i in range(len(A))]
    out = Tensor(C, (a,))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            for i in range(len(A)):
                for j in range(len(A[0])):
                    s = S[i][j]
                    d = s * (1.0 + A[i][j] * (1.0 - s))
                    a.grad[i][j] += g[i][j] * d
        else:
            for i in range(len(A)):
                s = S[i]
                d = s * (1.0 + A[i] * (1.0 - s))
                a.grad[i] += g[i] * d

    out._backward = bw
    return out


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(a: Tensor) -> Tensor:
    """tanh 近似的 GELU。"""
    A = a.data
    if _is2d(A):
        m, n = len(A), len(A[0])
        T = [[math.tanh(_GELU_C * (v + 0.044715 * v * v * v)) for v in row] for row in A]
        C = [[0.5 * A[i][j] * (1.0 + T[i][j]) for j in range(n)] for i in range(m)]
    else:
        m, n = 1, len(A)
        T = [[math.tanh(_GELU_C * (v + 0.044715 * v * v * v)) for v in A]]
        C = [[0.5 * A[i] * (1.0 + T[0][i]) for i in range(n)]]

    def dgelu(x: float, t: float) -> float:
        inner = _GELU_C * (x + 0.044715 * x * x * x)
        dinner = _GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
        sech2 = 1.0 - t * t
        return 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner

    out = Tensor(C if _is2d(A) else C[0], (a,))

    def bw(g: list[Any]) -> None:
        if _is2d(A):
            for i in range(m):
                for j in range(n):
                    a.grad[i][j] += g[i][j] * dgelu(A[i][j], T[i][j])
        else:
            for i in range(n):
                a.grad[i] += g[i] * dgelu(A[i], T[0][i])

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 归一化
# ----------------------------------------------------------------------
def layernorm(a: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    A = a.data
    m, n = len(A), len(A[0])
    C = [[0.0] * n for _ in range(m)]
    xhat = [[0.0] * n for _ in range(m)]
    inv = [0.0] * m
    for i in range(m):
        row = A[i]
        mu = sum(row) / n
        var = sum((v - mu) * (v - mu) for v in row) / n
        s = 1.0 / math.sqrt(var + eps)
        inv[i] = s
        for j in range(n):
            xh = (row[j] - mu) * s
            xhat[i][j] = xh
            C[i][j] = xh * w.data[j] + b.data[j]
    out = Tensor(C, (a, w, b))

    def bw(g: list[Any]) -> None:
        Wd = w.data
        for i in range(m):
            row = A[i]
            mu = sum(row) / n
            s = inv[i]
            # gxhat = g * w
            gxh = [g[i][j] * Wd[j] for j in range(n)]
            gvar = sum(gxh[j] * (row[j] - mu) for j in range(n)) * (-0.5) * (s ** 3)
            gmean = sum(gxh[j] for j in range(n)) * (-s)
            # Σ(x-μ)=0 ⇒ gvar 对 mean 的贡献项为 0
            for j in range(n):
                a.grad[i][j] += gxh[j] * s + gvar * 2.0 * (row[j] - mu) / n + gmean / n
                w.grad[j] += g[i][j] * xhat[i][j]
                b.grad[j] += g[i][j]

    out._backward = bw
    return out


def rmsnorm(a: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    A = a.data
    m, n = len(A), len(A[0])
    C = [[0.0] * n for _ in range(m)]
    xhat = [[0.0] * n for _ in range(m)]
    inv = [0.0] * m
    for i in range(m):
        row = A[i]
        ms = sum(v * v for v in row) / n
        s = 1.0 / math.sqrt(ms + eps)
        inv[i] = s
        for j in range(n):
            xh = row[j] * s
            xhat[i][j] = xh
            C[i][j] = xh * w.data[j]
    out = Tensor(C, (a, w))

    def bw(g: list[Any]) -> None:
        Wd = w.data
        for i in range(m):
            row = A[i]
            s = inv[i]
            gxh = [g[i][j] * Wd[j] for j in range(n)]
            gms = sum(gxh[j] * row[j] for j in range(n)) * (-0.5) * (s ** 3)
            for j in range(n):
                a.grad[i][j] += gxh[j] * s + gms * 2.0 * row[j] / n
                w.grad[j] += g[i][j] * xhat[i][j]

    out._backward = bw
    return out


# ----------------------------------------------------------------------
# 嵌入 / 注意力辅助
# ----------------------------------------------------------------------
def embed(weight: Tensor, ids: Sequence[int]) -> Tensor:
    """按 id 取行：``[V,d] , [L] -> [L,d]``。"""
    W = weight.data
    C = [list(W[i]) for i in ids]
    out = Tensor(C, (weight,))
    L = len(ids)
    d = len(W[0])

    def bw(g: list[Any]) -> None:
        for r in range(L):
            wi = weight.grad[ids[r]]
            for j in range(d):
                wi[j] += g[r][j]

    out._backward = bw
    return out


def softmax_rows(a: Tensor, mask: Sequence[Sequence[bool]] | None = None) -> Tensor:
    """逐行 softmax；``mask[i][j] == True`` 的位置置 ``-inf``（用于因果掩码）。"""
    A = a.data
    m, n = len(A), len(A[0])
    P = [[0.0] * n for _ in range(m)]
    for i in range(m):
        row = A[i]
        mx = -math.inf
        for j in range(n):
            if mask is not None and mask[i][j]:
                continue
            if row[j] > mx:
                mx = row[j]
        ex = [0.0] * n
        s = 0.0
        for j in range(n):
            if mask is not None and mask[i][j]:
                continue
            e = math.exp(row[j] - mx)
            ex[j] = e
            s += e
        if s == 0.0:
            s = 1.0
        for j in range(n):
            P[i][j] = ex[j] / s
    out = Tensor(P, (a,))

    def bw(g: list[Any]) -> None:
        for i in range(m):
            p = P[i]
            dot = 0.0
            for j in range(n):
                dot += g[i][j] * p[j]
            for j in range(n):
                a.grad[i][j] += p[j] * (g[i][j] - dot)

    out._backward = bw
    return out


def softmax_cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    """逐行 softmax + NLL，返回**标量**张量（形状 ``[[loss]]``）。"""
    A = logits.data
    m, V = len(A), len(A[0])
    probs: list[list[float]] = []
    loss = 0.0
    for i in range(m):
        row = A[i]
        mx = max(row)
        ex = [math.exp(v - mx) for v in row]
        s = sum(ex)
        p = [e / s for e in ex]
        probs.append(p)
        loss += -math.log(max(p[targets[i]], 1e-12))
    loss /= m
    out = Tensor([[loss]], (logits,))

    def bw(g: list[Any]) -> None:
        gs = g[0][0]
        for i in range(m):
            t = targets[i]
            for j in range(V):
                logits.grad[i][j] += gs * (probs[i][j] - (1.0 if j == t else 0.0)) / m

    out._backward = bw
    return out
