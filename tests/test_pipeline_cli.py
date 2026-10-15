"""流水线契约 + CLI 只读契约。

``test_readonly_commands_do_not_write`` 是本文件里最要紧的一条：它用**文件指纹**
证明读命令确实不落盘。不是「看一眼代码觉得没写」，而是跑完所有读命令后
逐个文件比对 md5 —— 这类契约一旦破了，用户看到的报告就可能对应另一份配置，
而错误结论会伪装成正常输出。
"""

from __future__ import annotations

import hashlib
import os
import re

import pytest

from architecturefactory.cli import main
from architecturefactory.pipeline import OUT_FILES, load_result

READONLY_CMDS = [
    ["summary"],
    ["frontier"],
    ["matrix", "--limit", "5"],
    ["compare"],
    ["lineage", "--limit", "8"],
    ["space"],
    ["task"],
    ["explain", "attention-swiglu-rmsnorm-l1-d12-h2"],
]


# ----------------------------------------------------------------------
# 产物契约
# ----------------------------------------------------------------------
def test_out_dir_has_all_required_files(out_dir):
    for name in OUT_FILES:
        assert os.path.exists(os.path.join(out_dir, name)), f"缺产物 {name}"
    assert os.path.exists(os.path.join(out_dir, "report.md"))
    assert os.path.exists(os.path.join(out_dir, "lineage.dot"))


def test_load_result_returns_none_when_incomplete(tmp_path):
    """缺任何一个必需文件都必须返回 ``None`` —— 不能「读一半」。"""
    assert load_result(str(tmp_path)) is None
    (tmp_path / "trials.json").write_text("[]", encoding="utf-8")
    assert load_result(str(tmp_path)) is None


def test_load_result_roundtrips(result, out_dir):
    r = load_result(out_dir)
    assert r is not None
    assert r.frontier == result.frontier
    assert r.hypervolume == pytest.approx(result.hypervolume)
    assert len(r.trials) == len(result.trials)
    assert len(r.matrix) == len(result.matrix)
    assert r.objectives == result.objectives


def test_frontier_entries_are_non_dominated(result):
    """前沿里的每个点都不该被任何试验支配 —— 这是「前沿」的定义。"""
    from architecturefactory.pareto import dominates

    front = result.frontier_trials()
    for f in front:
        fv = [f.metric(o) for o in result.objectives]
        for t in result.trials:
            tv = [t.metric(o) for o in result.objectives]
            if t.trial_id == f.trial_id:
                continue
            assert not dominates(tv, fv, result.objectives), (
                f"{f.trial_id} 被 {t.trial_id} 支配，却在前沿上"
            )


def test_frontier_has_no_duplicate_spec_ids(result):
    """**实测踩过的坑**：不同搜索器重复评估同一个架构会让前沿出现重复点。"""
    ids = [t.trial_id for t in result.trials]
    assert len(set(ids)) == len(ids), "同一个 spec_id 被评估了多次"


def test_ranks_are_consistent(result):
    for t in result.trials:
        assert t.rank >= 0
        assert t.on_frontier == (t.rank == 0)
        assert t.on_frontier == (t.trial_id in set(result.frontier))


def test_matrix_covers_all_trials(result):
    assert len(result.matrix) == len(result.trials)
    assert {r["trial_id"] for r in result.matrix} == {t.trial_id for t in result.trials}


def test_version_counts_match(result):
    v = result.version
    assert v is not None
    assert v.counts["trials"] == len(result.trials)
    assert v.counts["frontier"] == len(result.frontier)
    assert v.version_id.endswith(v.digest)


def test_reference_point_is_worse_than_every_trial(result):
    from architecturefactory.schema import METRIC_DIRECTIONS

    ref = result.counters["reference_point"]
    for t in result.trials:
        for i, o in enumerate(result.objectives):
            v = t.metric(o)
            if METRIC_DIRECTIONS[o] == "max":
                assert ref[i] <= v
            else:
                assert ref[i] >= v


def test_dag_covers_every_trial(result):
    assert result.dag is not None
    assert len(result.dag.nodes) == 2 * len(result.trials)


def test_report_renders(result):
    from architecturefactory.report import render_report

    md = render_report(result)
    assert md.startswith("# ArchitectureFactory 报告")
    assert "帕累托前沿" in md
    assert "可复现性边界" in md
    assert "启发式基线" in md
    # 报告里不该出现「按耗时排序」的结论
    assert "train_seconds" in md


# ----------------------------------------------------------------------
# 只读契约
# ----------------------------------------------------------------------
def _fingerprint(d: str) -> dict[str, str]:
    out = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                out[name] = hashlib.md5(fh.read()).hexdigest()
    return out


def test_readonly_commands_do_not_write(out_dir, capsys):
    """跑完所有读命令，产物目录必须**逐字节不变**。"""
    before = _fingerprint(out_dir)
    for cmd in READONLY_CMDS:
        rc = main(["--out", out_dir, *cmd])
        assert rc == 0, f"{cmd} 退出码 {rc}"
    capsys.readouterr()
    after = _fingerprint(out_dir)
    assert after == before, (
        "读命令改动了产物目录："
        + str({k: (before.get(k), after.get(k)) for k in set(before) | set(after)
               if before.get(k) != after.get(k)})
    )


def test_readonly_commands_produce_output(out_dir, capsys):
    """读命令必须真的有输出 —— 不能静默通过。"""
    for cmd in READONLY_CMDS:
        main(["--out", out_dir, *cmd])
        captured = capsys.readouterr()
        assert captured.out.strip(), f"{cmd} 没有输出任何内容"


def test_visualize_is_the_write_command(out_dir, capsys):
    """``visualize`` 是唯一的写命令：它会新增 report.html，且内容是自包含的。"""
    html_path = os.path.join(out_dir, "report.html")
    if os.path.exists(html_path):
        os.remove(html_path)
    assert not os.path.exists(html_path)
    assert main(["--out", out_dir, "visualize"]) == 0
    capsys.readouterr()
    assert os.path.exists(html_path)
    body = open(html_path, encoding="utf-8").read()
    assert body.startswith("<!DOCTYPE html>")
    assert "<svg" in body
    assert "http://" not in body and "https://" not in body, "HTML 不该引用外部资源"


def test_explain_is_readonly_and_needs_no_pipeline(cfg, capsys):
    """``explain`` 只从 spec_id 展开 IR —— 不该触发任何训练。"""
    rc = main(["--config", "configs/default.json", "--out", "nonexistent_out", "explain",
               "attention-moe-rmsnorm-l2-d16-h4-e2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "block0" in out and "block1" in out
    assert "params=" in out


def test_explain_rejects_illegal_spec(capsys):
    """非法架构（d_model 不被头数整除）必须报错并返回非零。"""
    rc = main(["--out", "nonexistent_out", "explain", "attention-mlp-rmsnorm-l1-d12-h5"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "整除" in out


def _strip_timing(obj):
    """递归剔除实测耗时字段 —— 它们是**唯一**允许不可复现的部分。"""
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items() if k != "train_seconds"}
    if isinstance(obj, list):
        return [_strip_timing(v) for v in obj]
    return obj


def test_run_is_deterministic(two_runs):
    """同一份配置跑两次，产物必须逐位相同（剔除耗时字段后）。

    两次运行来自 session 级夹具 ``two_runs`` —— 复现性是**一对**运行的性质，
    没必要每个用例各自重跑一对（那正是套件一度跑 5 分钟的原因之一）。
    """
    import json

    a, b = two_runs
    for name in OUT_FILES:
        if name == "version.json":
            continue
        da = json.load(open(os.path.join(a, name), encoding="utf-8"))
        db = json.load(open(os.path.join(b, name), encoding="utf-8"))
        assert _strip_timing(da) == _strip_timing(db), f"{name} 两次运行不一致"
    va = json.load(open(os.path.join(a, "version.json"), encoding="utf-8"))
    vb = json.load(open(os.path.join(b, "version.json"), encoding="utf-8"))
    assert va["digest"] == vb["digest"]
    assert va["version_id"] == vb["version_id"]


def test_timing_fields_are_the_only_nonreproducible_ones(two_runs):
    """明确钉住：``train_seconds`` 允许不同，其余字段不允许。"""
    import json

    from architecturefactory.schema import TIMING_METRICS

    a, b = two_runs
    ta = json.load(open(os.path.join(a, "trials.json"), encoding="utf-8"))
    tb = json.load(open(os.path.join(b, "trials.json"), encoding="utf-8"))
    assert TIMING_METRICS == frozenset({"train_seconds"})
    for x, y in zip(ta, tb):
        assert x["evaluation"]["accuracy"] == y["evaluation"]["accuracy"]
        assert x["evaluation"]["params"] == y["evaluation"]["params"]
        assert x["evaluation"]["curve"] == y["evaluation"]["curve"]


def test_rejected_trials_do_not_kill_the_pipeline(cfg, task):
    """预算超限必须变成一条拒绝记录，而不是异常冒泡 —— 否则搜索器的探索会被砍掉。"""
    from architecturefactory.pipeline import evaluate_spec
    from architecturefactory.schema import ArchSpec

    spec = ArchSpec("attention", "moe", "rmsnorm", 2, 16, n_heads=4, n_experts=2)
    outcome = evaluate_spec(spec, task, cfg, 0, max_params=10)
    assert outcome.trial is None
    assert "超过预算" in outcome.reason


def test_report_markdown_has_no_placeholders(result):
    """报告里不该漏出 ``None`` / ``nan`` —— 那是渲染没兜住的痕迹。"""
    from architecturefactory.report import render_report

    md = render_report(result)
    assert "nan" not in md.lower()
    assert " None" not in md
    assert "→" in md  # 散点图的坐标轴标注在
