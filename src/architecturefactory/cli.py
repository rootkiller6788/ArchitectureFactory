"""命令行入口。

**读命令不写盘。** ``summary`` / ``frontier`` / ``matrix`` / ``show`` / ``compare`` /
``lineage`` / ``explain`` / ``space`` / ``task`` 全部是只读的：优先用
:func:`pipeline.load_result` 从 ``out/`` 重建（快），``out/`` 不存在时退回
**在内存里跑一遍**（慢，但不落盘）。唯一会写盘的是 ``run`` 与 ``visualize``。

这个区分不是洁癖：``out/`` 里那份结果可能对应另一份配置，而读命令如果顺手重跑，
用户看到的就是「当前配置」的结果 —— 两者混在一起会得出错误结论。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import expand_specs, load_config, protocol_tag, version_digest
from .ir import parse, validate
from .pareto import ideal_point, reference_point
from .pipeline import load_result, run_pipeline, write_outputs
from .report import frontier_table
from .schema import ArchResult, ArchSpec

DEFAULT_CFG = "configs/default.json"
DEFAULT_OUT = "out"


def _load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config)


def _result(args: argparse.Namespace) -> tuple[ArchResult, dict[str, Any]]:
    """取结果：先读盘，读不到就内存里跑（**不写盘**）。"""
    cfg = _load_cfg(args)
    cached = load_result(args.out)
    if cached is not None:
        cached.cfg = cfg
        return cached, cfg
    print(f"[提示] {args.out}/ 没有可用结果，正在内存里跑一遍（不写盘，约 2-3 分钟）…",
          file=sys.stderr)
    return run_pipeline(cfg), cfg


def _fmt_table(head: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(空)\n"
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out) + "\n"


def _num(v: float) -> str:
    """整数按整数印（参数量 / FLOPs 不该出现 ``3120.0000`` 这种假精度）。"""
    f = float(v)
    return str(int(f)) if f == int(f) and abs(f) < 1e15 else f"{f:.4f}"


def _public(d: dict[str, Any]) -> dict[str, Any]:
    """去掉 ``_comment`` 之类的说明键 —— 它们是给人看配置的，不是数据。"""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ----------------------------------------------------------------------
# 命令
# ----------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    if args.trials is not None:
        cfg["search"]["trials"] = int(args.trials)
    if args.searchers:
        cfg["search"]["searchers"] = list(args.searchers)
    result = run_pipeline(cfg, progress=lambda m: print(m, flush=True))
    write_outputs(result, args.out)
    print(f"\n写出 {args.out}/：{', '.join(sorted(os.listdir(args.out)))}")
    print(f"版本 {result.version.version_id if result.version else '-'}")
    print(f"前沿 {len(result.frontier)} 个，超体积 {result.hypervolume:.6g}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    res, cfg = _result(args)
    v = res.version
    c = res.counters
    print(f"版本        : {v.version_id if v else '-'}")
    print(f"digest      : {v.digest if v else '-'}")
    print(f"创建时间    : {v.created_at if v else '-'}  （不可逐位复现）")
    print(f"任务        : {c.get('task')}")
    print(f"随机基线    : {c.get('chance')}")
    print(f"启发式基线  : {c.get('heuristic')}  （只会输出最后一个 item）")
    print(f"目标        : {res.objectives}")
    print(f"协议指纹    : {c.get('protocol')}")
    print(f"空间 / 有效 / 被拒 : {c.get('pool_size')} / {len(res.trials)} / "
          f"{len(c.get('rejected', []))}")
    print(f"参考点      : {c.get('reference_point')}")
    print(f"理想点      : {c.get('ideal_point')}")
    print(f"前沿        : {len(res.frontier)} 个")
    print(f"超体积      : {res.hypervolume:.6g}")
    for name, info in res.searchers.items():
        print(f"  - {name}: {info}")
    return 0


def _cmd_frontier(args: argparse.Namespace) -> int:
    res, _ = _result(args)
    sys.stdout.write(frontier_table(res))
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    res, _ = _result(args)
    objs = list(res.objectives)
    head = ["spec_id", "searcher", "rank", "前沿", *objs, "flops", "train_s"]
    rows = []
    for r in res.matrix[: args.limit]:
        rows.append([
            r["trial_id"], r["searcher"], str(r["rank"]),
            "✓" if r["on_frontier"] else "",
            *[_num(float(r[o])) for o in objs],
            str(r["flops"]), f"{r['train_seconds']:.2f}",
        ])
    sys.stdout.write(_fmt_table(head, rows))
    if len(res.matrix) > args.limit:
        print(f"… 共 {len(res.matrix)} 行，用 --limit 调整")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    res, _ = _result(args)
    t = res.find(args.spec_id) or res.by_spec(args.spec_id)
    if t is None:
        print(f"没有找到 {args.spec_id!r}", file=sys.stderr)
        return 2
    print(f"spec_id   : {t.trial_id}")
    print(f"searcher  : {t.searcher}   rank={t.rank}  前沿={t.on_frontier}")
    print(f"spec      : {json.dumps(t.spec.to_dict(), ensure_ascii=False)}")
    print(f"父本      : {t.parents}")
    print(f"evaluation: {json.dumps(t.evaluation.to_dict(), ensure_ascii=False)}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    res, _ = _result(args)
    names = list(res.searchers) or sorted({t.searcher for t in res.trials})
    head = ["searcher", "有效", "被拒", "前沿数", "最好 acc", "中位 acc", "最好点 params"]
    rows = []
    for n in names:
        ts = [t for t in res.trials if t.searcher == n]
        accs = sorted(t.evaluation.accuracy for t in ts)
        best = max(ts, key=lambda t: t.evaluation.accuracy) if ts else None
        rows.append([
            n, str(len(ts)), str(res.searchers.get(n, {}).get("rejected", 0)),
            str(len([t for t in ts if t.on_frontier])),
            f"{best.evaluation.accuracy:.4f}" if best else "-",
            f"{accs[len(accs) // 2]:.4f}" if accs else "-",
            str(best.evaluation.params) if best else "-",
        ])
    sys.stdout.write(_fmt_table(head, rows))
    return 0


def _cmd_lineage(args: argparse.Namespace) -> int:
    res, _ = _result(args)
    if res.dag is None:
        print("(无血统图)")
        return 0
    sys.stdout.write(res.dag.to_ascii(max_rows=args.limit))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """把一个 spec_id 展开成 IR 并解释 —— **只读，不需要跑流水线**。"""
    from .builder import build
    from .profile import profile_model
    from .task import build_task

    cfg = _load_cfg(args)
    spec = ArchSpec.from_spec_id(args.spec_id)
    task = build_task(cfg)
    ir = parse(spec, task.vocab_size)
    errs = validate(ir, cfg["search"].get("max_params"), max_seq_len=task.seq_len)
    print(f"spec_id : {spec.spec_id}")
    print(f"canonical: {json.dumps(spec.to_dict(), ensure_ascii=False)}")
    print(f"IR 指纹 : {ir.fingerprint()[:16]}")
    print(f"blocks  : {ir.n_layers}")
    for i, b in enumerate(ir.blocks):
        print(f"  block{i}: mixer={b.mixer:9s} ffn={b.ffn:6s} norm={b.norm:9s} "
              f"residual={b.residual:3s} d={b.d_model} heads={b.n_heads} experts={b.n_experts}")
    print(f"校验    : {'通过' if not errs else '失败 -> ' + '; '.join(errs)}")
    if not errs:
        m = build(ir, max_seq_len=task.seq_len)
        p = profile_model(m, task.seq_len)
        print(f"画像    : params={p.params} flops={p.flops} "
              f"activation_elems={p.activation_elems}")
    return 0 if not errs else 1


def _cmd_space(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    pool = expand_specs(cfg)
    print(f"空间大小 : {len(pool)}")
    print(f"轴       : {json.dumps(_public(cfg['space']), ensure_ascii=False)}")
    by_mixer: dict[str, int] = {}
    for s in pool:
        by_mixer[s.mixer] = by_mixer.get(s.mixer, 0) + 1
    print(f"按 mixer : {dict(sorted(by_mixer.items()))}")
    print("前 8 个  :")
    for s in pool[:8]:
        print(f"  {s.spec_id}")
    return 0


def _cmd_task(args: argparse.Namespace) -> int:
    from .task import build_task

    cfg = _load_cfg(args)
    t = build_task(cfg)
    print(f"任务      : {t.name}")
    print(f"定义      : {json.dumps(t.to_dict(), ensure_ascii=False)}")
    print(f"序列长度  : {t.seq_len}")
    print(f"随机基线  : {t.chance:.4f}")
    print(f"启发式基线: {t.heuristic:.4f}")
    print(f"train/test: {len(t.train)} / {len(t.test)}")
    print(f"样本      : {t.train.inputs[0]} -> {t.train.targets[0]}")
    print(f"协议指纹  : {protocol_tag(cfg)}")
    print(f"版本 digest: {version_digest(cfg)}")
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    """**写命令**：生成自包含的 ``report.html``（内联 SVG，无外部依赖）。"""
    res, _ = _result(args)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(res))
    print(f"写出 {path}")
    return 0


# ----------------------------------------------------------------------
# HTML（内联，无外部依赖）
# ----------------------------------------------------------------------
def render_html(res: ArchResult) -> str:
    objs = list(res.objectives)
    pts = [(t.metric(objs[0]), t.metric(objs[1]), t.on_frontier, t.spec_id) for t in res.trials]
    W, H, PAD = 760, 420, 60
    xs = [p[0] for p in pts] or [0.0]
    ys = [p[1] for p in pts] or [1.0]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    fx = lambda v: PAD + (v - x0) / (x1 - x0) * (W - 2 * PAD)  # noqa: E731
    fy = lambda v: H - PAD - (v - y0) / (y1 - y0) * (H - 2 * PAD)  # noqa: E731
    marks = []
    for x, y, front, sid in pts:
        color = "#2b7a78" if front else "#9aa5b1"
        r = 7 if front else 4
        marks.append(
            f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="{r}" fill="{color}">'
            f"<title>{sid}: {objs[0]}={x:.4f} {objs[1]}={y}</title></circle>"
        )
    rows = "".join(
        "<tr class='{}'><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            "front" if r["on_frontier"] else "",
            r["trial_id"], r["searcher"], r["rank"],
            f"{float(r[objs[0]]):.4f}", f"{float(r[objs[1]]):.0f}",
        )
        for r in res.matrix
    )
    v = res.version
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ArchitectureFactory {v.version_id if v else ''}</title>
<style>
 body{{font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f7f8fa;color:#1a202c}}
 .wrap{{max-width:1100px;margin:0 auto;padding:28px}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 8px;color:#2d3748}}
 .meta{{color:#4a5568;font-size:13px}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin-top:14px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border-bottom:1px solid #edf2f7;padding:6px 8px;text-align:left}}
 th{{background:#f7fafc;font-weight:600}}
 tr.front td{{background:#e6fffa;font-weight:600}}
 .kpi{{display:flex;gap:24px;flex-wrap:wrap}} .kpi div{{min-width:120px}}
 .kpi b{{display:block;font-size:18px;color:#2b7a78}}
 text{{font:11px sans-serif;fill:#4a5568}}
</style></head><body><div class="wrap">
<h1>ArchitectureFactory — 架构搜索报告</h1>
<div class="meta">版本 {v.version_id if v else '-'} · digest {v.digest if v else '-'} · 创建 {v.created_at if v else '-'}</div>
<div class="card"><div class="kpi">
 <div><b>{len(res.trials)}</b>有效试验</div>
 <div><b>{len(res.frontier)}</b>前沿点</div>
 <div><b>{res.hypervolume:.4g}</b>超体积</div>
 <div><b>{res.counters.get('pool_size')}</b>空间大小</div>
 <div><b>{float(res.counters.get('chance',0)):.3f}</b>随机基线</div>
 <div><b>{float(res.counters.get('heuristic',0)):.3f}</b>启发式基线</div>
</div></div>
<div class="card"><h2>{objs[0]} × {objs[1]}</h2>
<svg viewBox="0 0 {W} {H}" width="100%" height="{H}">
 <line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#cbd5e0"/>
 <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#cbd5e0"/>
 <text x="{W/2}" y="{H-14}" text-anchor="middle">{objs[0]} →</text>
 <text x="16" y="{PAD-14}">{objs[1]} ↑</text>
 <text x="{PAD}" y="{H-PAD+18}">{x0:.3g}</text>
 <text x="{W-PAD}" y="{H-PAD+18}" text-anchor="end">{x1:.3g}</text>
 <text x="{PAD-8}" y="{H-PAD}" text-anchor="end">{y0:.3g}</text>
 <text x="{PAD-8}" y="{PAD+4}" text-anchor="end">{y1:.3g}</text>
 {''.join(marks)}
</svg>
<div class="meta">深色大点 = 帕累托前沿（互不支配）。悬停看 spec_id。</div></div>
<div class="card"><h2>架构数据库（按 rank）</h2>
<table><thead><tr><th>spec_id</th><th>searcher</th><th>rank</th>
<th>{objs[0]}</th><th>{objs[1]}</th></tr></thead><tbody>{rows}</tbody></table></div>
</div></body></html>
"""


# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------
#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` / `--root` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    for name in ("config", "out", "root"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="architecturefactory",
        description="把「架构」当决策变量的搜索闭环：产物是 (质量, 成本) 的帕累托前沿。",
    )
    p.add_argument("--config", default=DEFAULT_CFG, help="配置文件路径；相对路径按项目根解析")
    p.add_argument("--out", default=DEFAULT_OUT, help="产物目录；相对路径按项目根解析")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑流水线并写盘")
    r.add_argument("--trials", type=int, default=None, help="每个搜索器的预算（覆盖配置）")
    r.add_argument("--searchers", nargs="*", default=None, help="覆盖搜索器列表")
    r.set_defaults(fn=_cmd_run)

    sub.add_parser("summary", help="版本 / 计数 / 超体积").set_defaults(fn=_cmd_summary)
    sub.add_parser("frontier", help="打印帕累托前沿").set_defaults(fn=_cmd_frontier)

    m = sub.add_parser("matrix", help="打印架构数据库")
    m.add_argument("--limit", type=int, default=40)
    m.set_defaults(fn=_cmd_matrix)

    s = sub.add_parser("show", help="看某个试验的详情")
    s.add_argument("spec_id")
    s.set_defaults(fn=_cmd_show)

    sub.add_parser("compare", help="按搜索器对比").set_defaults(fn=_cmd_compare)

    ln = sub.add_parser("lineage", help="打印血统图")
    ln.add_argument("--limit", type=int, default=60)
    ln.set_defaults(fn=_cmd_lineage)

    e = sub.add_parser("explain", help="把一个 spec_id 展开成 IR（只读，不跑流水线）")
    e.add_argument("spec_id")
    e.set_defaults(fn=_cmd_explain)

    sub.add_parser("space", help="设计空间概览").set_defaults(fn=_cmd_space)
    sub.add_parser("task", help="任务定义概览").set_defaults(fn=_cmd_task)
    sub.add_parser("visualize", help="生成自包含 report.html（写命令）").set_defaults(fn=_cmd_visualize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _anchor_args(args)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
