"""任务：确定性 / 形状 / 基线。**基线算错会让「学到了」变成「蒙对了」。**"""

from __future__ import annotations

import pytest

from architecturefactory.task import TASK_NAMES, build_task, evaluate, train_step_loss


def test_both_tasks_are_registered():
    assert set(TASK_NAMES) == {"index-recall", "induction-recall"}


def test_index_recall_shapes(cfg):
    t = build_task(cfg)
    assert t.name == "index-recall"
    assert t.seq_len == t.params["n_items"] + 1 == 5
    for row, target in zip(t.train.inputs, t.train.targets):
        assert len(row) == t.seq_len
        # 下标 token 取自词表尾部保留段；item 取自前缀段 —— 两段不相交
        n_vals = t.vocab_size - t.params["n_items"]
        assert row[-1] >= n_vals
        assert all(v < n_vals for v in row[:-1])
        assert target == row[row[-1] - n_vals]


def test_index_recall_items_are_distinct(cfg):
    """item 必须互不相同，否则「第 q 个」是有歧义的。"""
    t = build_task(cfg)
    for row in t.train.inputs:
        items = row[:-1]
        assert len(set(items)) == len(items)


def test_induction_recall_shapes(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c["task"]["name"] = "induction-recall"
    c["task"]["n_pairs"] = 3
    c["task"]["n_train"] = 16
    c["task"]["n_test"] = 8
    t = build_task(c)
    assert t.seq_len == 2 * 3 + 1 == 7
    for row, target in zip(t.train.inputs, t.train.targets):
        assert len(row) == 7
        keys = row[0::2][:3]
        vals = row[1::2][:3]
        assert row[-1] in keys
        assert target == vals[keys.index(row[-1])]


def test_task_is_deterministic(cfg):
    a = build_task(cfg)
    b = build_task(cfg)
    assert a.train.inputs == b.train.inputs
    assert a.train.targets == b.train.targets
    assert a.test.inputs == b.test.inputs


def test_train_and_test_splits_differ(cfg):
    t = build_task(cfg)
    assert t.train.inputs[:8] != t.test.inputs[:8]


def test_task_definition_excludes_dataset(cfg):
    """摘要只记定义、不记数据 —— 数据由 seed 重建（否则 version.json 会巨大）。"""
    d = build_task(cfg).to_dict()
    assert set(d) == {"name", "seed", "vocab_size", "n_items"}
    assert "inputs" not in d and "targets" not in d


def test_baselines(cfg):
    t = build_task(cfg)
    assert t.chance == pytest.approx(1 / 12)
    assert t.heuristic == pytest.approx(1 / 4)  # 「只输出最后一个 item」


def test_unknown_task_raises(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c["task"]["name"] = "nope"
    with pytest.raises(ValueError, match="未知任务"):
        build_task(c)


def test_vocab_too_small_raises(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c["task"]["vocab_size"] = 5
    c["task"]["n_items"] = 4
    with pytest.raises(ValueError, match="太小"):
        build_task(c)


# ----------------------------------------------------------------------
# 训练步：**batch loss 必须取平均**
# ----------------------------------------------------------------------
def test_batch_loss_is_averaged_not_summed(cfg, task):
    """同一个 batch 的 loss 不能随 batch 大小线性放大。

    若取和，梯度范数正比于 batch，``grad_clip`` 会把梯度整体裁掉 ——
    表现是「训练完全不动」，而且改 batch 大小行为就变。
    """
    from architecturefactory.builder import build_from_spec
    from architecturefactory.schema import ArchSpec

    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    m = build_from_spec(spec, task.vocab_size, max_seq_len=task.seq_len)
    idxs = list(range(8))
    loss_t, loss_v, _ = train_step_loss(m, task.train, idxs)
    assert loss_v == pytest.approx(loss_t.data[0][0])
    # 单个样本的 loss 与 8 个样本的平均 loss 应当同量级（而不是 8 倍）
    _, one, _ = train_step_loss(m, task.train, [idxs[0]])
    assert loss_v < one * 3


def test_evaluate_returns_accuracy_and_loss(cfg, task):
    from architecturefactory.builder import build_from_spec
    from architecturefactory.schema import ArchSpec

    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    m = build_from_spec(spec, task.vocab_size, max_seq_len=task.seq_len)
    acc, loss = evaluate(m, task.test)
    assert 0.0 <= acc <= 1.0
    assert loss > 0.0


def test_untrained_accuracy_is_near_chance(cfg, task):
    """未训练时应当接近随机 —— 若明显高于 chance，说明基线或数据泄漏了。"""
    from architecturefactory.builder import build_from_spec
    from architecturefactory.schema import ArchSpec

    spec = ArchSpec("attention", "mlp", "rmsnorm", 1, 12, n_heads=2)
    m = build_from_spec(spec, task.vocab_size, max_seq_len=task.seq_len)
    acc, _ = evaluate(m, task.test)
    assert acc <= 4 * task.chance
