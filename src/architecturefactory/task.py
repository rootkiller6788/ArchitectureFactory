"""观测任务：**被搜索对象的一半**。

两个任务，都是「同一个 IR 空间里哪种装配更好」的探针，但**难度分层**：

``index-recall``（默认）
    序列 = ``[x₀ x₁ … x_{P-1} | q]``，目标 = ``x_q``。
    末尾的 ``q`` 是一个「下标 token」（取自词表尾部保留的一段）。模型要从位置 ``q``
    把值取回来。解它需要**按位置寻址**：查询 token 的内容必须与「第 q 个位置」绑定。
    一层注意力（配位置嵌入）就能学 —— 它的打分函数是 ``(词嵌入 ⊕ 位置嵌入)`` 的双线性型，
    可以让「下标 token = q」与「位置 = q」的键对上。而门控递推**结构上做不到**：
    它的状态是定长向量，没有「按位置随机读」这个操作。
    于是这个任务**便宜**（``L = P+1``）且**区分度大**（见附录 B 的实测表）。

``induction-recall``
    序列 = ``[k₁ v₁ k₂ v₂ … k_P v_P | k_q]``，目标 = ``v_q``。
    经典「归纳头」任务。但要注意：本项目的模型**没有位置信息时它在数学上不可解** ——
    最后一层的注意力输出只是查询 token 内容的函数，与上下文无关，因此无法随上下文
    给出不同的 ``v_q``（见附录 C 坑 46）。即使补上位置嵌入，也需要 ≥3 层才开始学到
    （实测 200 步到 0.656）。**留作难度上界**，默认不用它做搜索的判据。

**关于随机基线。** ``chance`` 是均匀猜测。但还有一个更强的「启发式基线」：
``index-recall`` 里如果模型只会输出最后一个 item，命中率是 ``1/P``；
``induction-recall`` 里如果只会输出最近见到的值，命中率也是 ``1/P``。
**测试必须断言架构显著高于启发式基线**，否则「学到了」和「蒙对了」分不开 ——
这正是本项目踩过的坑（见附录 C 坑 47）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .autograd import gather_rows, scale, softmax_cross_entropy
from .builder import Model
from .rng import make_rng

__all__ = [
    "Dataset",
    "Task",
    "build_task",
    "evaluate",
    "train_step_loss",
    "TASK_NAMES",
]

TASK_NAMES = ("index-recall", "induction-recall")


@dataclass
class Dataset:
    """定长序列 + 单目标。``inputs`` 每行长度相同（= ``Task.seq_len``）。"""

    inputs: list[list[int]] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {"n": len(self), "seq_len": len(self.inputs[0]) if self.inputs else 0}


@dataclass
class Task:
    """任务定义。**它是「被搜索对象」的一半**，所以整体进版本摘要。

    ``params`` 只放定义性的键值（vocab / 规模 / 种子），**不放数据集本体** ——
    数据由这些参数确定性重建，摘要只记定义（见 skill 坑 20）。
    """

    name: str
    params: dict[str, Any]
    seq_len: int
    seed: int
    train: Dataset = field(default_factory=Dataset)
    test: Dataset = field(default_factory=Dataset)

    @property
    def vocab_size(self) -> int:
        return int(self.params["vocab_size"])

    @property
    def chance(self) -> float:
        """均匀猜测的期望准确率（最弱的基线）。"""
        return 1.0 / self.vocab_size

    @property
    def heuristic(self) -> float:
        """「只会输出最后一个 item / 最近见到的值」这个启发式的命中率。

        这是**真正的**及格线：低于它说明连启发式都没学到，高于它才说明学到了结构。
        """
        return 1.0 / int(self.params["n_items" if self.name == "index-recall" else "n_pairs"])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, **self.params}


# ----------------------------------------------------------------------
# 生成器
# ----------------------------------------------------------------------
def _gen_index_recall(
    i: int, rng: Any, p: dict[str, Any]
) -> tuple[list[int], int]:
    """``[x₀ … x_{P-1} | q] -> x_q``。

    词表尾部保留 ``n_items`` 个 token 专做「下标」（值 0..P-1），
    前面的 token 做 item 值。下标与 item 词表不相交，模型无法靠词嵌入作弊。
    """
    vocab = int(p["vocab_size"])
    n_items = int(p["n_items"])
    n_vals = vocab - n_items
    items = rng.sample(range(n_vals), n_items)  # 互不相同的 item
    idx = rng.randrange(n_items)
    seq = items + [n_vals + idx]
    return seq, items[idx]


def _gen_induction_recall(
    i: int, rng: Any, p: dict[str, Any]
) -> tuple[list[int], int]:
    """``[k₁ v₁ … k_P v_P | k_q] -> v_q``。

    键与值的配对**每个样本独立随机** —— 所以没有固定的「键→值」表可背，
    必须真的做上下文内的归纳（这也是它难的原因，见模块 docstring）。
    """
    vocab = int(p["vocab_size"])
    n_pairs = int(p["n_pairs"])
    n_keys = vocab // 2
    keys = rng.sample(range(n_keys), n_pairs)
    vals = rng.sample(range(n_keys, vocab), n_pairs)
    seq: list[int] = []
    for k, v in zip(keys, vals):
        seq.append(k)
        seq.append(v)
    q = rng.randrange(n_pairs)
    seq.append(keys[q])
    return seq, vals[q]


_GENERATORS: dict[str, Callable[[int, Any, dict[str, Any]], tuple[list[int], int]]] = {
    "index-recall": _gen_index_recall,
    "induction-recall": _gen_induction_recall,
}


def _make_split(
    n: int, name: str, params: dict[str, Any], seed: int, split: str
) -> Dataset:
    """确定性生成一个 split。**每个样本一个独立 RNG** —— 与生成顺序无关。"""
    gen = _GENERATORS[name]
    ds = Dataset()
    for i in range(n):
        rng = make_rng(seed, "data", name, split, i)
        seq, target = gen(i, rng, params)
        ds.inputs.append(seq)
        ds.targets.append(target)
    return ds


def build_task(cfg: dict[str, Any]) -> Task:
    """从配置造任务（train / test 两个 split）。"""
    t = cfg["task"]
    name = str(t.get("name", TASK_NAMES[0]))
    if name not in _GENERATORS:
        raise ValueError(f"未知任务：{name!r}（可选：{list(_GENERATORS)}）")
    seed = int(cfg.get("seed", 0))
    vocab = int(t["vocab_size"])
    if name == "index-recall":
        n_items = int(t["n_items"])
        if vocab <= n_items + 1:
            raise ValueError(f"vocab_size({vocab}) 太小：需要 > n_items({n_items}) + 1")
        params: dict[str, Any] = {"vocab_size": vocab, "n_items": n_items}
        seq_len = n_items + 1
    else:
        n_pairs = int(t["n_pairs"])
        if vocab < 2 * n_pairs:
            raise ValueError(
                f"vocab_size({vocab}) 太小：需要 ≥ 2·n_pairs({2 * n_pairs})"
            )
        params = {"vocab_size": vocab, "n_pairs": n_pairs}
        seq_len = 2 * n_pairs + 1
    return Task(
        name=name,
        params=params,
        seq_len=seq_len,
        seed=seed,
        train=_make_split(int(t["n_train"]), name, params, seed, "train"),
        test=_make_split(int(t["n_test"]), name, params, seed, "test"),
    )


# ----------------------------------------------------------------------
# 评估 / 训练步
# ----------------------------------------------------------------------
def _add_scalar(a: Any, b: Any) -> Any:
    """两个 ``[[v]]`` 标量张量相加（``autograd`` 没有标量加法，这里包一层）。"""
    from .autograd import add as _add

    return _add(a, b)


def train_step_loss(
    model: Model, ds: Dataset, idxs: Sequence[int]
) -> tuple[Any, float, int]:
    """一个 batch 的 ``(loss_tensor, loss值, 正确数)``；``loss_tensor`` 供反向。

    **batch loss 取平均，不取和。** 求和会让梯度范数正比于 batch 大小，
    于是 ``grad_clip`` 几乎总是把梯度整体裁掉 —— 表现就是「训练完全不动」，
    而且 batch 一改行为就变。这个坑很隐蔽：loss 曲线看起来在降（因为它在降和），
    但参数几乎没更新。见 README 附录 C。
    """
    total: Any = None
    correct = 0
    for i in idxs:
        ids = ds.inputs[i]
        logits = model.forward(ids)
        last = gather_rows(logits, [len(ids) - 1])  # [1,V]
        loss = softmax_cross_entropy(last, [ds.targets[i]])
        total = loss if total is None else _add_scalar(total, loss)
        row = last.data[0]
        pred = max(range(len(row)), key=lambda j: row[j])
        if pred == ds.targets[i]:
            correct += 1
    assert total is not None
    n = len(idxs)
    if n != 1:
        total = scale(total, 1.0 / n)
    return total, total.data[0][0], correct


def evaluate(model: Model, ds: Dataset) -> tuple[float, float]:
    """返回 ``(accuracy, mean_loss)``。只前向，不做参数更新。"""
    total_loss = 0.0
    correct = 0
    n = len(ds)
    for i in range(n):
        ids = ds.inputs[i]
        logits = model.forward(ids)
        last = gather_rows(logits, [len(ids) - 1])
        loss = softmax_cross_entropy(last, [ds.targets[i]])
        total_loss += loss.data[0][0]
        row = last.data[0]
        pred = max(range(len(row)), key=lambda j: row[j])
        if pred == ds.targets[i]:
            correct += 1
    return (correct / n if n else 0.0, total_loss / n if n else 0.0)
