"""搜索血统 DAG：**每一次「提议」和「评估」都留痕**。

为什么搜索也需要血统。因为搜索的产物是**前沿**，而前沿是「哪些点被生成过、
被谁生成、评估结果如何」的函数。只存最终前沿，就丢掉了「为什么没找到更好的」——
是空间里没有，还是搜索器没走到？这两件事的修法完全不同。

图结构（每个 trial 两个节点）：

    propose(t)  --evaluates-->  evaluate(t)
        ↑            ↑
        │ derives    │ (跨 trial 的父子关系 = 变异/交叉/代理推荐)

- ``derives`` 边：搜索的**探索结构**。随机搜索没有这种边（互不相关）；
  进化搜索的每条边指向两个父本；贝叶斯搜索的边指向「代理模型是根据哪些点拟合的」
  （取最近的 k 个）。
- ``evaluates`` 边：``propose`` 到 ``evaluate`` 的落地点。**把「提议」与「实测」
  分开记**，是为了让「提议了但评估失败」这件事在图上是可见的（evaluate 节点缺失）。

``to_dot`` 输出 Graphviz；``to_ascii`` 输出纯文本（不依赖任何外部工具，测试用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import Trial

__all__ = ["Node", "Edge", "Lineage", "build_lineage"]


@dataclass
class Node:
    node_id: str
    kind: str  # "propose" | "evaluate"
    label: str
    trial_id: str
    searcher: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "trial_id": self.trial_id,
            "searcher": self.searcher,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        return cls(
            node_id=d["node_id"],
            kind=d["kind"],
            label=d["label"],
            trial_id=d["trial_id"],
            searcher=d.get("searcher", ""),
            payload=dict(d.get("payload", {})),
        )


@dataclass
class Edge:
    a: str  # 源
    b: str  # 目标
    kind: str  # "derives" | "evaluates"

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Edge":
        return cls(a=d["a"], b=d["b"], kind=d["kind"])


@dataclass
class Lineage:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Lineage":
        return cls(
            nodes=[Node.from_dict(n) for n in d.get("nodes", [])],
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
        )

    # -- 查询 --------------------------------------------------------
    def by_trial(self, trial_id: str) -> list[Node]:
        return [n for n in self.nodes if n.trial_id == trial_id]

    def children(self, node_id: str) -> list[Node]:
        idx = {n.node_id: n for n in self.nodes}
        return [idx[e.b] for e in self.edges if e.a == node_id and e.b in idx]

    def derive_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.kind == "derives"]

    def roots(self) -> list[Node]:
        targets = {e.b for e in self.edges}
        return [n for n in self.nodes if n.node_id not in targets]

    def depth(self) -> int:
        """最长链长度（节点数）。随机搜索应当很浅；进化/贝叶斯应当更深。"""
        succ: dict[str, list[str]] = {}
        for e in self.edges:
            succ.setdefault(e.a, []).append(e.b)
        memo: dict[str, int] = {}

        def walk(nid: str, seen: frozenset[str]) -> int:
            if nid in memo:
                return memo[nid]
            if nid in seen:  # 理论上不该有环；有环就截断，别死循环
                return 0
            best = 1
            for c in succ.get(nid, ()):
                best = max(best, 1 + walk(c, seen | {nid}))
            memo[nid] = best
            return best

        return max((walk(n.node_id, frozenset()) for n in self.nodes), default=0)

    # -- 渲染 --------------------------------------------------------
    def to_dot(self, highlight: set[str] | None = None) -> str:
        hl = highlight or set()
        lines = ["digraph lineage {", '  rankdir="LR";', '  node [shape=box, fontsize=10];']
        for n in self.nodes:
            color = "#2b7a78" if n.kind == "propose" else "#c46a2f"
            shape = "box" if n.kind == "propose" else "ellipse"
            pen = "2.4" if n.trial_id in hl else "1.0"
            lines.append(
                f'  "{n.node_id}" [label="{_esc(n.label)}", shape={shape}, '
                f'color="{color}", penwidth={pen}];'
            )
        for e in self.edges:
            style = "solid" if e.kind == "derives" else "dashed"
            lines.append(f'  "{e.a}" -> "{e.b}" [style={style}];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_ascii(self, max_rows: int = 60) -> str:
        """纯文本渲染：``节点  类型  标签`` + ``父 -> 子`` 列表。**无外部依赖。**"""
        out = [f"lineage: {len(self.nodes)} nodes, {len(self.edges)} edges, depth={self.depth()}"]
        for n in self.nodes[:max_rows]:
            out.append(f"  [{n.kind:8s}] {n.node_id:16s} {n.label}")
        if len(self.nodes) > max_rows:
            out.append(f"  … 另有 {len(self.nodes) - max_rows} 个节点")
        for e in self.edges[:max_rows]:
            out.append(f"  {e.a:16s} --{e.kind}--> {e.b}")
        if len(self.edges) > max_rows:
            out.append(f"  … 另有 {len(self.edges) - max_rows} 条边")
        return "\n".join(out) + "\n"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_lineage(trials: list[Trial], suggest_edges: dict[str, list[str]] | None = None) -> Lineage:
    """从试验列表造血统图。

    ``suggest_edges``：``trial_id -> [父 trial_id]``。由各搜索器在 ``observe`` 时
    自己填（进 :attr:`Trial.parents`）。这里只负责把它翻译成图。
    """
    del suggest_edges  # 父本信息已经落在 Trial.parents 上，单一来源
    lin = Lineage()
    for t in trials:
        pnode = f"p:{t.trial_id}"
        enode = f"e:{t.trial_id}"
        lin.nodes.append(
            Node(
                node_id=pnode,
                kind="propose",
                label=f"{t.searcher}:{t.spec_id}",
                trial_id=t.trial_id,
                searcher=t.searcher,
                payload={"rank": t.rank, "on_frontier": t.on_frontier},
            )
        )
        lin.nodes.append(
            Node(
                node_id=enode,
                kind="evaluate",
                label=f"acc={t.evaluation.accuracy:.3f} p={t.evaluation.params}",
                trial_id=t.trial_id,
                searcher=t.searcher,
                payload=t.evaluation.to_dict(),
            )
        )
        lin.edges.append(Edge(a=pnode, b=enode, kind="evaluates"))
        for parent in t.parents:
            lin.edges.append(Edge(a=f"p:{parent}", b=pnode, kind="derives"))
    return lin
