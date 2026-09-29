"""流水线：``Spec → Parse → IR → Validate → Build → Train Proxy → Profile → Compare → Search ↺``。

对应 README 那条链路。每一步都有一处**硬约束**，写在这里而不是散在代码里：

1. **Parse**：``ArchSpec → ArchIR``。架构变成数据，之后所有环节只认 IR。
2. **Validate**：构建**之前**校验（含 ``max_params`` 预算）。坏结构绝不进装配。
3. **Build**：只查表（:mod:`architecturefactory.builder`）。没有 ``if 架构 == ...``。
4. **Train Proxy**：所有候选共享同一份训练协议。协议指纹进试验记录。
5. **Profile**：成本是**结构属性**，由公式算出，不测量 —— 于是成本轴逐位可复现。
6. **Compare**：非支配排序 + 超体积。产物是**前沿**，不是「冠军」。
7. **Search ↺**：三种搜索器共用同一份 ``trials`` 预算。

**一次试验的失败不该杀死整条流水线。** 预算超限、IR 非法、构建异常 —— 都记成
一条 ``status="rejected"`` 的试验并继续（见 :class:`Trial` 的 payload 约定）。
搜索本来就是「大量提议里筛出少数可行」，让异常冒泡等于把搜索器的探索能力砍掉。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from .builder import build
from .config import expand_specs, objectives_of, protocol_tag, version_digest
from .ir import parse, validate
from .lineage import build_lineage
from .pareto import (
    frontier_ids,
    hypervolume,
    ideal_point,
    non_dominated_sort,
    reference_point,
)
from .profile import profile_model
from .proxy import TrainConfig, train_proxy
from .schema import ArchResult, ArchSpec, Trial, Version, utcnow
from .search import make_searcher
from .task import Task, build_task

__all__ = ["run_pipeline", "evaluate_spec", "write_outputs", "load_result", "OUT_FILES"]

#: 写盘产物。``load_result`` 需要它们全部在场才认账。
OUT_FILES = (
    "trials.json",
    "frontier.json",
    "lineage.json",
    "version.json",
    "matrix.json",
)


@dataclass
class EvalOutcome:
    """一次评估的结果。``trial`` 为 ``None`` 表示被拒绝（原因在 ``reason``）。"""

    trial: Trial | None
    reason: str = ""


def evaluate_spec(
    spec: ArchSpec,
    task: Task,
    cfg: dict[str, Any],
    seed: int,
    max_params: int | None = None,
) -> EvalOutcome:
    """评估一个架构：校验 → 装配 → 训练 → 画像。**任何失败都变成一条拒绝记录。**"""
    try:
        ir = parse(spec, task.vocab_size)
    except Exception as exc:  # noqa: BLE001 - 解析失败要变成数据，不能冒泡
        return EvalOutcome(None, f"parse: {type(exc).__name__}: {exc}")

    errs = validate(ir, max_params, max_seq_len=task.seq_len)
    if errs:
        return EvalOutcome(None, "validate: " + "; ".join(errs))

    try:
        model = build(ir, init_scale=TrainConfig(cfg).init_scale, seed=seed,
                      max_seq_len=task.seq_len)
    except Exception as exc:  # noqa: BLE001
        return EvalOutcome(None, f"build: {type(exc).__name__}: {exc}")

    try:
        ev = train_proxy(model, task, cfg, seed)
    except Exception as exc:  # noqa: BLE001
        return EvalOutcome(None, f"train: {type(exc).__name__}: {exc}")

    prof = profile_model(model, task.seq_len)
    ev.params = prof.params
    ev.flops = prof.flops
    model.spec = spec.canonical()
    trial = Trial(
        trial_id=spec.spec_id,
        spec=spec.canonical(),
        evaluation=ev,
        searcher="",
        rank=-1,
        on_frontier=False,
    )
    return EvalOutcome(trial)


def run_pipeline(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> ArchResult:
    """跑完整条流水线。**确定性**：同一份 cfg 给出同一份 result（除耗时字段）。"""
    say = progress or (lambda _m: None)
    task = build_task(cfg)
    objectives = objectives_of(cfg)
    seed = int(cfg.get("seed", 0))
    search_cfg = cfg["search"]
    trials_per = int(search_cfg.get("trials", 32))
    max_params = search_cfg.get("max_params")
    space = cfg["space"]
    pool = expand_specs(cfg)

    result = ArchResult(objectives=objectives, cfg=cfg)
    result.counters["pool_size"] = len(pool)
    result.counters["protocol"] = protocol_tag(cfg)
    result.counters["task"] = task.to_dict()
    result.counters["chance"] = task.chance
    result.counters["heuristic"] = task.heuristic

    #: 跨搜索器共享的「已评估」集合 —— 架构数据库只有一份，同一 spec_id 只评估一次。
    seen: set[str] = set()

    for name in search_cfg["searchers"]:
        searcher = make_searcher(
            name, space, objectives, search_cfg, seed, task.vocab_size, seen
        )
        say(f"[{name}] 开始，预算 {trials_per} 次")
        rejected = 0
        for k in range(trials_per):
            spec = searcher.propose(result.trials)
            if spec is None:
                say(f"[{name}] 空间耗尽，提前结束于第 {k} 次")
                break
            parents = list(searcher.pending_parents)
            outcome = evaluate_spec(spec, task, cfg, seed, max_params)
            if outcome.trial is None:
                rejected += 1
                result.counters.setdefault("rejected", []).append(
                    {"searcher": name, "spec_id": spec.spec_id, "reason": outcome.reason}
                )
                continue
            trial = outcome.trial
            if result.find(trial.trial_id) is not None:
                # 防御性去重：搜索器已共享 seen，这里兜住任何漏网的重复
                continue
            trial.searcher = name
            trial.parents = parents
            trial.propose_node = f"p:{trial.trial_id}"
            trial.evaluate_node = f"e:{trial.trial_id}"
            searcher.observe(trial)
            result.trials.append(trial)
            if progress is not None:
                say(
                    f"[{name}] {k + 1}/{trials_per} {trial.spec_id} "
                    f"acc={trial.evaluation.accuracy:.3f} params={trial.evaluation.params}"
                )
        say(f"[{name}] 完成：{trials_per - rejected} 条有效，{rejected} 条被拒")
        result.searchers[name] = {"trials": trials_per, "rejected": rejected}

    _finalize(result, cfg, task)
    return result


def _finalize(result: ArchResult, cfg: dict[str, Any], task: Task) -> None:
    """排序 / 前沿 / 超体积 / 血统 / 版本。**所有派生量在这里一次算清。**"""
    objectives = result.objectives
    fronts = non_dominated_sort(result.trials, objectives)
    rank_of: dict[str, int] = {}
    for i, front in enumerate(fronts):
        for tid in front:
            rank_of[tid] = i
    result.frontier = frontier_ids(result.trials, objectives)
    front_set = set(result.frontier)
    for t in result.trials:
        t.rank = rank_of.get(t.trial_id, -1)
        t.on_frontier = t.trial_id in front_set

    ref = reference_point(result.trials, objectives)
    result.reference = ref[0] if len(ref) == 1 else 0.0
    result.counters["reference_point"] = list(ref)
    if result.trials:
        pts = [[t.metric(o) for o in objectives] for t in result.frontier_trials()]
        result.hypervolume = hypervolume(pts, ref, objectives)
        result.counters["ideal_point"] = list(ideal_point(result.trials, objectives))

    result.dag = build_lineage(result.trials)

    digest = version_digest(cfg)
    result.version = Version(
        version_id=f"{cfg.get('version_tag', 'v1')}-{digest}",
        created_at=utcnow(),
        digest=digest,
        tag=str(cfg.get("version_tag", "v1")),
        counts={
            "pool": result.counters["pool_size"],
            "trials": len(result.trials),
            "frontier": len(result.frontier),
            "rejected": len(result.counters.get("rejected", [])),
        },
    )
    result.matrix = build_matrix(result)


def build_matrix(result: ArchResult) -> list[dict[str, Any]]:
    """架构数据库：每行一个试验，含 spec 全字段 + 全部目标 + 前沿标记。"""
    rows: list[dict[str, Any]] = []
    for t in result.trials:
        row: dict[str, Any] = {
            "trial_id": t.trial_id,
            "searcher": t.searcher,
            "rank": t.rank,
            "on_frontier": t.on_frontier,
        }
        row.update({k: v for k, v in t.spec.to_dict().items() if k != "spec_id"})
        row.update(
            {
                "accuracy": round(t.evaluation.accuracy, 6),
                "loss": round(t.evaluation.loss, 6),
                "params": t.evaluation.params,
                "flops": t.evaluation.flops,
                "train_seconds": round(t.evaluation.train_seconds, 4),
            }
        )
        rows.append(row)
    rows.sort(key=lambda r: (r["rank"], -r["accuracy"], r["params"], r["trial_id"]))
    return rows


# ----------------------------------------------------------------------
# 读写
# ----------------------------------------------------------------------
def write_outputs(result: ArchResult, out_dir: str) -> None:
    """写盘。``report.md`` 由 :mod:`architecturefactory.report` 生成。"""
    from .report import render_report

    os.makedirs(out_dir, exist_ok=True)

    def dump(name: str, obj: Any) -> None:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)

    dump("trials.json", [t.to_dict() for t in result.trials])
    dump("frontier.json", {
        "frontier": list(result.frontier),
        "hypervolume": result.hypervolume,
        "reference_point": result.counters.get("reference_point", []),
        "objectives": list(result.objectives),
        "rows": [r for r in result.matrix if r["on_frontier"]],
    })
    dump("lineage.json", (result.dag.to_dict() if result.dag else {"nodes": [], "edges": []}))
    dump("version.json", (result.version.to_dict() if result.version else {}))
    dump("matrix.json", {
        "objectives": list(result.objectives),
        "counters": result.counters,
        "searchers": result.searchers,
        "rows": result.matrix,
    })
    with open(os.path.join(out_dir, "lineage.dot"), "w", encoding="utf-8") as fh:
        fh.write(result.dag.to_dot(set(result.frontier)) if result.dag else "digraph {}\n")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_report(result))


def load_result(out_dir: str) -> ArchResult | None:
    """**只读重建**。任何一个必需文件缺失就返回 ``None``（调用方决定要不要重跑）。"""
    needed = list(OUT_FILES)
    for name in needed:
        if not os.path.exists(os.path.join(out_dir, name)):
            return None

    def read(name: str) -> Any:
        with open(os.path.join(out_dir, name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    trials = [Trial.from_dict(d) for d in read("trials.json")]
    front = read("frontier.json")
    matrix = read("matrix.json")
    version = read("version.json")
    result = ArchResult(
        trials=trials,
        frontier=list(front.get("frontier", [])),
        hypervolume=float(front.get("hypervolume", 0.0)),
        searchers=dict(matrix.get("searchers", {})),
        matrix=list(matrix.get("rows", [])),
        objectives=tuple(matrix.get("objectives", ["accuracy", "params"])),
        version=Version.from_dict(version) if version else None,
        counters=dict(matrix.get("counters", {})),
    )
    result.dag = build_lineage(trials)
    return result
