"""报告渲染：``ArchResult → markdown``。

**纯文本产物。** 不依赖任何绘图库 —— 帕累托前沿用 ASCII 散点画出来（:func:`ascii_scatter`），
因为「看得见前沿」这件事不该以「装了 matplotlib」为前提。要看矢量图就打开
``lineage.dot``，要看交互就上 ``report.html``（见 CLI 的 ``visualize``）。

报告里**必须写明哪些数字不可逐位复现**（``train_seconds`` 及其派生的任何东西），
否则读者会以为两次运行的差异是搜索不稳定 —— 见 :data:`schema.TIMING_METRICS`。
"""

from __future__ import annotations

from typing import Sequence

from .schema import METRIC_DIRECTIONS, TIMING_METRICS, ArchResult, Trial

__all__ = ["render_report", "ascii_scatter", "frontier_table"]

_W = 52
_H = 16


def ascii_scatter(
    points: Sequence[tuple[float, float]],
    x_label: str,
    y_label: str,
    highlight: Sequence[bool] | None = None,
    width: int = _W,
    height: int = _H,
) -> str:
    """把 ``(x, y)`` 画成 ASCII 散点。``highlight[i]`` 为真时用 ``*``（前沿点）。

    刻度只用真实极值标注 —— 不引入「好看的」圆整，避免读者对精度产生错觉。
    """
    if not points:
        return "(无数据)\n"
    hl = list(highlight) if highlight is not None else [False] * len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    grid = [[" "] * width for _ in range(height)]
    for (x, y), star in zip(points, hl):
        cx = int((x - x0) / (x1 - x0) * (width - 1))
        cy = int((y - y0) / (y1 - y0) * (height - 1))
        grid[height - 1 - cy][cx] = "*" if star else "."
    lines = [f"  {y_label} ↑"]
    for i, row in enumerate(grid):
        v = y1 - (y1 - y0) * i / (height - 1)
        lines.append(f"  {v:>11.4g} │{''.join(row)}")
    lines.append(f"  {'':>11s} └" + "─" * width)
    lines.append(f"  {'':>11s}  {x0:<{width // 2}.4g}{x1:>{width - width // 2}.4g}  {x_label} →")
    lines.append("  ( * = 帕累托前沿   . = 被支配 )")
    return "\n".join(lines) + "\n"


def _fmt(v: float, digits: int = 4) -> str:
    """整数目标（参数量 / FLOPs）按整数印，避免出现 ``3120.0000`` 这种假精度。"""
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.{digits}f}"


def frontier_table(result: ArchResult, digits: int = 4) -> str:
    """前沿表格：每行一个非支配架构。"""
    objs = list(result.objectives)
    head = ["spec_id", *objs, "searcher", "rank"]
    rows = []
    for tid in result.frontier:
        t = result.find(tid)
        if t is None:
            continue
        rows.append(
            [tid, *[_fmt(t.metric(o), digits) for o in objs], t.searcher, str(t.rank)]
        )
    if not rows:
        return "(前沿为空)\n"
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    out = [line(head), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out) + "\n"


def _per_searcher(result: ArchResult) -> str:
    names = list(result.searchers) or sorted({t.searcher for t in result.trials})
    lines = ["| searcher | 有效 | 被拒 | 前沿数 | 最好 accuracy | 该点 params | 最好 loss |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        ts = [t for t in result.trials if t.searcher == n]
        front = [t for t in ts if t.on_frontier]
        best = max(ts, key=lambda t: t.evaluation.accuracy) if ts else None
        info = result.searchers.get(n, {})
        lines.append(
            f"| {n} | {len(ts)} | {info.get('rejected', 0)} | {len(front)} | "
            f"{_fmt(best.evaluation.accuracy) if best else '-'} | "
            f"{best.evaluation.params if best else '-'} | "
            f"{_fmt(best.evaluation.loss) if best else '-'} |"
        )
    return "\n".join(lines) + "\n"


def _best_of_each_family(result: ArchResult) -> str:
    """按 mixer 家族分组看最好成绩 —— **架构是决策变量**最直观的一处证据。"""
    fams: dict[str, list[Trial]] = {}
    for t in result.trials:
        fams.setdefault(t.spec.mixer, []).append(t)
    lines = ["| mixer | 试验数 | 最好 accuracy | 中位 accuracy | 最好点 params |",
             "|---|---|---|---|---|"]
    for mixer in sorted(fams):
        ts = fams[mixer]
        accs = sorted(t.evaluation.accuracy for t in ts)
        best = max(ts, key=lambda t: t.evaluation.accuracy)
        med = accs[len(accs) // 2]
        lines.append(
            f"| {mixer} | {len(ts)} | {_fmt(best.evaluation.accuracy)} | {_fmt(med)} | "
            f"{best.evaluation.params} |"
        )
    return "\n".join(lines) + "\n"


def render_report(result: ArchResult) -> str:
    """整份报告。"""
    v = result.version
    c = result.counters
    task = c.get("task", {})
    objs = list(result.objectives)
    objs_txt = ", ".join(
        f"{o}({'越大越好' if METRIC_DIRECTIONS[o] == 'max' else '越小越好'})" for o in objs
    )
    pts = [(t.metric(objs[0]), t.metric(objs[1])) for t in result.trials] if len(objs) >= 2 else []
    hl = [t.on_frontier for t in result.trials]

    parts: list[str] = []
    parts.append(f"# ArchitectureFactory 报告\n")
    parts.append(
        f"- 版本：`{v.version_id if v else '(未生成)'}`  （digest `{v.digest if v else '-'}`）\n"
        f"- 创建时间：{v.created_at if v else '-'}（**不可逐位复现**）\n"
        f"- 任务：`{task.get('name', '?')}`  {task}\n"
        f"- 随机基线（均匀猜测）：{_fmt(float(c.get('chance', 0.0)))}\n"
        f"- **启发式基线**（只会输出最后一个 item）：{_fmt(float(c.get('heuristic', 0.0)))}\n"
        f"- 目标：{objs_txt}\n"
        f"- 训练协议指纹：`{c.get('protocol', '-')}`\n"
        f"- 设计空间大小：{c.get('pool_size', '-')}，有效试验：{len(result.trials)}，"
        f"被拒：{c.get('rejected', []) and len(c['rejected']) or 0}\n"
        f"- 参考点：{c.get('reference_point', [])}\n"
        f"- 理想点：{c.get('ideal_point', [])}\n"
        f"- **超体积：{result.hypervolume:.6g}**（参考点固定，可跨运行比较）\n"
    )

    parts.append("\n## 1. 帕累托前沿\n")
    parts.append(f"前沿规模 **{len(result.frontier)}** / {len(result.trials)}。\n\n")
    parts.append(frontier_table(result))

    if pts:
        parts.append(f"\n### 前沿散点（{objs[0]} × {objs[1]}）\n\n```\n")
        parts.append(ascii_scatter(pts, objs[0], objs[1], hl))
        parts.append("```\n")

    parts.append("\n## 2. 搜索器对比（同一份预算）\n")
    parts.append(_per_searcher(result))

    parts.append("\n## 3. 按 mixer 家族\n")
    parts.append(
        "同一套组件、不同的 mixer 装配。**这是「架构是决策变量」最直接的证据**：\n"
        "如果同一任务下各家族的成绩没有系统性差异，那这个项目的命题就不成立。\n\n"
    )
    parts.append(_best_of_each_family(result))

    parts.append("\n## 4. 血统图\n")
    if result.dag is not None:
        parts.append(
            f"节点 {len(result.dag.nodes)}，边 {len(result.dag.edges)}，"
            f"最长链 {result.dag.depth()}。完整图见 `lineage.dot`。\n\n```\n"
        )
        parts.append(result.dag.to_ascii(max_rows=24))
        parts.append("```\n")

    parts.append("\n## 5. 全部试验（按 rank 排序）\n")
    head = ["spec_id", "searcher", "rank", "前沿", *objs, "flops", "train_s"]
    rows = []
    for r in result.matrix:
        rows.append([
            r["trial_id"], r["searcher"], str(r["rank"]),
            "✓" if r["on_frontier"] else "",
            *[_fmt(float(r[o])) for o in objs],
            str(r["flops"]), f"{r['train_seconds']:.2f}",
        ])
    widths = [max(len(str(x[i])) for x in [head, *rows]) for i in range(len(head))]
    parts.append("| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |\n")
    parts.append("|" + "|".join("-" * (w + 2) for w in widths) + "|\n")
    for r in rows:
        parts.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |\n")

    parts.append("\n## 6. 可复现性边界\n")
    parts.append(
        "- **可逐位复现**：``spec_id`` / IR 指纹 / 参数量 / FLOPs / 试验顺序 / "
        "前沿集合 / 超体积 / 训练得到的 accuracy（同一 Python 版本、同一机器）。\n"
        "- **不可逐位复现**："
        + ", ".join(f"`{m}`" for m in sorted(TIMING_METRICS))
        + "，以及 `version.json` 的 `created_at`。它们是实测耗时，天然带噪声。\n"
        "- 因此报告里**没有**任何以耗时排序的结论 —— 耗时只作为参考列出现。\n"
    )
    return "".join(parts)
