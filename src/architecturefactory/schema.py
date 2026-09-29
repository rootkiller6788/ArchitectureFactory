"""核心对象：ArchSpec（决策变量）/ ArchIR（可组合图）/ Profile / Evaluation / Trial / Version。

**这是本项目的分水岭。** README 第二节点名了要避免的写法：

    if architecture == "transformer": ...
    elif architecture == "mamba": ...

所以「架构」在这里是一个**数据对象**（:class:`ArchSpec`），它被解析成一张图
（:class:`ArchIR`），图再被装配成模型。没有一处 ``if architecture == ...``
的分支散落在组件实现里 —— 组件只认「我是 Mixer / FFN / Norm」，装配由 IR 驱动。

``ArchSpec.spec_id`` 是**可逆**的：``from_spec_id(spec_id(s)) == s``。
搜索器只搬 ``spec_id``，产物里只存 ``spec_id``，重建时不需要额外信息。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .rng import sha256_text

#: 组件枚举（设计空间的轴取值）
MIXERS: tuple[str, ...] = ("attention", "ssm", "hybrid")
FFNS: tuple[str, ...] = ("mlp", "swiglu", "moe")
NORMS: tuple[str, ...] = ("layernorm", "rmsnorm")
RESIDUALS: tuple[str, ...] = ("pre",)

#: 哪些 mixer 用得到 ``n_heads``（其余取 1，且不进 spec_id —— 避免同架构两个 id）
ATTENTIVE_MIXERS: tuple[str, ...] = ("attention", "hybrid")

#: 目标方向表。质量越大越好，成本越小越好。
METRIC_DIRECTIONS: dict[str, str] = {
    "accuracy": "max",
    "params": "min",
    "flops": "min",
    "train_seconds": "min",
}

#: 含实测成分、**不可逐位复现**的指标（见 skill 坑 25/33）
TIMING_METRICS: frozenset[str] = frozenset({"train_seconds"})


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# 决策变量：ArchSpec
# ----------------------------------------------------------------------
@dataclass
class ArchSpec:
    """一次搜索里的「架构」—— 即 IR 搜索空间里的一个点。

    字段与设计空间的轴一一对应。``n_heads`` 只在注意力型 mixer 上有意义，
    ``n_experts`` 只在 ``moe`` 上有意义；无关时分别归一为 1 / 0，
    且**不进 spec_id** —— 否则「同一架构」会有多个 id，前沿上会出现重复点。
    """

    mixer: str
    ffn: str
    norm: str
    n_layers: int
    d_model: int
    n_heads: int = 1
    n_experts: int = 0
    residual: str = "pre"

    def canonical(self) -> "ArchSpec":
        """把「无关字段」归一，保证同架构同 id。"""
        n_heads = self.n_heads if self.mixer in ATTENTIVE_MIXERS else 1
        n_experts = self.n_experts if self.ffn == "moe" else 0
        return ArchSpec(
            mixer=self.mixer,
            ffn=self.ffn,
            norm=self.norm,
            n_layers=int(self.n_layers),
            d_model=int(self.d_model),
            n_heads=int(n_heads),
            n_experts=int(n_experts),
            residual=self.residual,
        )

    @property
    def spec_id(self) -> str:
        """可逆短标识，例如 ``attention-swiglu-rmsnorm-l2-d24-h4``。"""
        s = self.canonical()
        parts = [s.mixer, s.ffn, s.norm, f"l{s.n_layers}", f"d{s.d_model}"]
        if s.mixer in ATTENTIVE_MIXERS:
            parts.append(f"h{s.n_heads}")
        if s.ffn == "moe":
            parts.append(f"e{s.n_experts}")
        return "-".join(parts)

    @classmethod
    def from_spec_id(cls, spec_id: str) -> "ArchSpec":
        toks = spec_id.split("-")
        if len(toks) < 5:
            raise ValueError(f"spec_id 格式不符：{spec_id!r}")
        mixer, ffn, norm = toks[0], toks[1], toks[2]
        n_layers = d_model = n_heads = n_experts = None
        for t in toks[3:]:
            if t.startswith("l"):
                n_layers = int(t[1:])
            elif t.startswith("d"):
                d_model = int(t[1:])
            elif t.startswith("h"):
                n_heads = int(t[1:])
            elif t.startswith("e"):
                n_experts = int(t[1:])
        if n_layers is None or d_model is None:
            raise ValueError(f"spec_id 缺 l/d：{spec_id!r}")
        return cls(
            mixer=mixer,
            ffn=ffn,
            norm=norm,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads or 1,
            n_experts=n_experts or 0,
        ).canonical()

    def mixer_at(self, layer: int) -> str:
        """第 ``layer`` 层实际用哪个 mixer（hybrid 逐层交替）。"""
        if self.mixer == "hybrid":
            return "attention" if layer % 2 == 0 else "ssm"
        return self.mixer

    def to_dict(self) -> dict[str, Any]:
        s = self.canonical()
        return {
            "spec_id": s.spec_id,
            "mixer": s.mixer,
            "ffn": s.ffn,
            "norm": s.norm,
            "residual": s.residual,
            "n_layers": s.n_layers,
            "d_model": s.d_model,
            "n_heads": s.n_heads,
            "n_experts": s.n_experts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchSpec":
        return cls(
            mixer=d["mixer"],
            ffn=d["ffn"],
            norm=d["norm"],
            n_layers=d["n_layers"],
            d_model=d["d_model"],
            n_heads=d.get("n_heads", 1),
            n_experts=d.get("n_experts", 0),
            residual=d.get("residual", "pre"),
        ).canonical()

    @property
    def digest(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))


# ----------------------------------------------------------------------
# IR：可组合图
# ----------------------------------------------------------------------
@dataclass
class BlockIR:
    """一个 Transformer-ish block：Mixer → Residual → FFN → Residual。"""

    mixer: str
    ffn: str
    norm: str
    residual: str
    d_model: int
    n_heads: int
    n_experts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixer": self.mixer,
            "ffn": self.ffn,
            "norm": self.norm,
            "residual": self.residual,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_experts": self.n_experts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlockIR":
        return cls(**d)


@dataclass
class ArchIR:
    """``ArchSpec`` 解析出来的可组合图（可序列化）。

    形状：``Embedding → Block × N → Head``。每个 block 的组件都只是「一个名字」，
    真正的实现由 :mod:`architecturefactory.components` 按名字查表装配 ——
    于是「架构」不再是代码分支，而是数据。
    """

    vocab_size: int
    d_model: int
    blocks: list[BlockIR] = field(default_factory=list)
    head: str = "linear"

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "head": self.head,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchIR":
        return cls(
            vocab_size=d["vocab_size"],
            d_model=d["d_model"],
            head=d.get("head", "linear"),
            blocks=[BlockIR.from_dict(b) for b in d["blocks"]],
        )

    def fingerprint(self) -> str:
        """IR 的稳定指纹（用于「同 IR 不同 spec_id」这类检查）。"""
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))


# ----------------------------------------------------------------------
# 度量
# ----------------------------------------------------------------------
@dataclass
class Profile:
    """结构画像：参数量 / FLOPs / 激活元素数。**全部可手算、可逐位复现。**"""

    params: int
    flops: int
    activation_elems: int
    n_blocks: int
    d_model: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "flops": self.flops,
            "activation_elems": self.activation_elems,
            "n_blocks": self.n_blocks,
            "d_model": self.d_model,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Profile":
        return cls(**d)


@dataclass
class Evaluation:
    """一次代理评估的产物。质量 + 成本都记下来。"""

    accuracy: float
    loss: float
    params: int
    flops: int
    train_seconds: float
    curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 6),
            "loss": round(self.loss, 6),
            "params": self.params,
            "flops": self.flops,
            "train_seconds": round(self.train_seconds, 6),
            "curve": [round(v, 6) for v in self.curve],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evaluation":
        return cls(
            accuracy=d["accuracy"],
            loss=d["loss"],
            params=d["params"],
            flops=d["flops"],
            train_seconds=d["train_seconds"],
            curve=list(d.get("curve", [])),
        )


# ----------------------------------------------------------------------
# 试验 / 版本 / 结果
# ----------------------------------------------------------------------
@dataclass
class Trial:
    """搜索里的一次试验。``spec_id`` 是决策变量，``evaluation`` 自带结果（见 skill 坑 15）。"""

    trial_id: str
    spec: ArchSpec
    evaluation: Evaluation
    searcher: str
    parents: list[str] = field(default_factory=list)
    propose_node: str = ""
    evaluate_node: str = ""
    rank: int = -1
    on_frontier: bool = False

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id

    def metric(self, name: str) -> float:
        if name == "accuracy":
            return self.evaluation.accuracy
        if name == "params":
            return float(self.evaluation.params)
        if name == "flops":
            return float(self.evaluation.flops)
        if name == "train_seconds":
            return self.evaluation.train_seconds
        raise KeyError(f"未知指标：{name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "spec": self.spec.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "searcher": self.searcher,
            "parents": list(self.parents),
            "propose_node": self.propose_node,
            "evaluate_node": self.evaluate_node,
            "rank": self.rank,
            "on_frontier": self.on_frontier,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trial":
        return cls(
            trial_id=d["trial_id"],
            spec=ArchSpec.from_dict(d["spec"]),
            evaluation=Evaluation.from_dict(d["evaluation"]),
            searcher=d["searcher"],
            parents=list(d.get("parents", [])),
            propose_node=d.get("propose_node", ""),
            evaluate_node=d.get("evaluate_node", ""),
            rank=int(d.get("rank", -1)),
            on_frontier=bool(d.get("on_frontier", False)),
        )


@dataclass
class Version:
    """内容寻址版本：digest 由**被搜索对象的定义**决定（任务 + 空间 + 种子），
    不由搜索配置决定 —— 换一个搜索器只该产生新的试验，不该产生新的版本号。
    """

    version_id: str
    created_at: str
    digest: str
    tag: str
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "digest": self.digest,
            "tag": self.tag,
            "counts": dict(sorted(self.counts.items())),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Version":
        return cls(
            version_id=d["version_id"],
            created_at=d["created_at"],
            digest=d["digest"],
            tag=d["tag"],
            counts=dict(d.get("counts", {})),
        )


@dataclass
class ArchResult:
    """整条流水线的产物。"""

    trials: list[Trial] = field(default_factory=list)
    frontier: list[str] = field(default_factory=list)
    hypervolume: float = 0.0
    reference: float = 0.0
    searchers: dict[str, Any] = field(default_factory=dict)
    matrix: list[dict[str, Any]] = field(default_factory=list)
    objectives: tuple[str, ...] = ("accuracy", "params")
    dag: Any = None
    version: Version | None = None
    counters: dict[str, Any] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    # -- 便捷查询 ----------------------------------------------------
    def find(self, trial_id: str) -> Trial | None:
        for t in self.trials:
            if t.trial_id == trial_id:
                return t
        return None

    def by_spec(self, spec_id: str) -> Trial | None:
        for t in self.trials:
            if t.spec_id == spec_id:
                return t
        return None

    def frontier_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.trial_id in set(self.frontier)]

    def best_by(self, metric: str) -> Trial | None:
        if not self.trials:
            return None
        direction = METRIC_DIRECTIONS[metric]
        key = (lambda t: t.metric(metric))  # noqa: E731
        return (min if direction == "min" else max)(self.trials, key=key)
