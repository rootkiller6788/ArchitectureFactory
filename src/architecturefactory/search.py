"""搜索器：随机 / 进化 / 贝叶斯。**同一个空间，同一份预算，三种走法。**

搜索的接口只有两个动作：

- :meth:`Searcher.propose` —— 给出下一个要评估的 :class:`ArchSpec`（空间耗尽返回 ``None``）；
- :meth:`Searcher.observe` —— 把评估结果回灌给搜索器（进化/贝叶斯据此调整下一步）。

**三者必须共享同一个 ``trials`` 预算**，否则比较的是「谁的预算多」而不是「谁的策略好」。
这是本项目最容易做错的地方之一（见 README 附录 C 坑 48）。

三种策略的分工：

``random``
    在展开后的空间里确定性洗牌后依次取。**它是基线** —— 任何「智能」搜索器
    如果打不过随机，就说明它的先验是错的。

``evolution``
    种群 = 非支配排序的第一梯队 + 拥挤距离补齐。后代由变异（改一个字段）与
    交叉（两个父本各取一部分字段）产生。选择压力来自**帕累托支配**而不是加权和，
    所以它天然维护前沿的多样性。

``bayesian``
    用 Nadaraya–Watson 核回归当代理（见 :mod:`architecturefactory.surrogate`），
    在候选池上取 ``mu - kappa·sigma`` 最小的点。前 ``n_init`` 个点纯随机 ——
    没有数据时任何代理都是噪声。

**变异必须能改变「有效字段」。** 比如把 ``mixer`` 从 ``attention`` 改成 ``ssm`` 时，
``n_heads`` 必须一起归一（否则 spec_id 会变但 IR 不变，搜索就会在同一个点上反复
「发现新架构」）。这由 :meth:`ArchSpec.canonical` 兜住。
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from .ir import expand_space
from .pareto import (
    crowding_distance,
    ideal_point,
    non_dominated_sort,
    random_weights,
    scalarize,
)
from .rng import make_rng
from .schema import (
    ATTENTIVE_MIXERS,
    FFNS,
    MIXERS,
    NORMS,
    ArchSpec,
    Trial,
)

__all__ = ["Searcher", "RandomSearcher", "EvolutionSearcher", "BayesianSearcher", "make_searcher", "SEARCHERS"]


class Searcher:
    """搜索器基类。``propose`` / ``observe`` 是唯一的扩展点。"""

    name = "base"

    def __init__(
        self,
        space: dict[str, Any],
        objectives: Sequence[str],
        search_cfg: dict[str, Any],
        seed: int,
        vocab_size: int,
        seen: set[str] | None = None,
    ) -> None:
        self.space = space
        self.objectives = tuple(objectives)
        self.cfg = search_cfg
        self.seed = seed
        self.vocab_size = vocab_size
        self.pool = expand_space(space, vocab_size)
        if not self.pool:
            raise ValueError("设计空间展开为空，检查 configs 的 space 段")
        self.rng = make_rng(seed, "search", self.name)
        #: **跨搜索器共享**的「已提议」集合。三个搜索器各跑各的，但架构数据库只有一份：
        #: 同一个 ``spec_id`` 只该被评估一次。否则前沿上会出现重复点、
        #: 而且白烧掉三分之一的训练预算。见 README 附录 C 坑 49。
        self.proposed: set[str] = seen if seen is not None else set()
        #: 上一次 ``propose`` 的父本（spec_id 列表）。流水线读它填 ``Trial.parents``。
        #: 放在这里而不是让流水线猜，是因为**只有搜索器自己知道**它的探索结构。
        self.pending_parents: list[str] = []

    # -- 扩展点 ------------------------------------------------------
    def propose(self, history: list[Trial]) -> ArchSpec | None:
        raise NotImplementedError

    def observe(self, trial: Trial) -> None:
        """默认什么都不做（随机搜索不需要反馈）。"""

    # -- 工具 --------------------------------------------------------
    def _fresh(self, spec: ArchSpec) -> ArchSpec | None:
        """规范化后判重：**同一个 spec_id 只提议一次**。"""
        s = spec.canonical()
        if s.spec_id in self.proposed:
            return None
        self.proposed.add(s.spec_id)
        return s

    def _mutate(self, spec: ArchSpec) -> ArchSpec:
        """改一个随机字段。取值一律来自 ``space`` —— 变异不越界。

        **返回值一律规范化。** 比如把 mixer 从 ``attention`` 改成 ``ssm`` 时，
        ``n_heads`` 必须一起归一 —— 否则 spec_id 变了而 IR 没变，搜索会以为
        自己发现了新架构。这条与 :meth:`_crossover` 保持一致。
        """
        s = spec.canonical()
        fields = ["mixer", "ffn", "norm", "n_layers", "d_model"]
        if s.mixer in ATTENTIVE_MIXERS:
            fields.append("n_heads")
        if s.ffn == "moe":
            fields.append("n_experts")
        f = self.rng.choice(fields)
        if f == "mixer":
            return ArchSpec(self.rng.choice(MIXERS), s.ffn, s.norm, s.n_layers,
                            s.d_model, s.n_heads, s.n_experts).canonical()
        if f == "ffn":
            return ArchSpec(s.mixer, self.rng.choice(FFNS), s.norm, s.n_layers,
                            s.d_model, s.n_heads,
                            self.rng.choice(self.space["n_experts"])).canonical()
        if f == "norm":
            return ArchSpec(s.mixer, s.ffn, self.rng.choice(NORMS), s.n_layers,
                            s.d_model, s.n_heads, s.n_experts).canonical()
        if f == "n_layers":
            return ArchSpec(s.mixer, s.ffn, s.norm, self.rng.choice(self.space["n_layers"]),
                            s.d_model, s.n_heads, s.n_experts).canonical()
        if f == "d_model":
            return ArchSpec(s.mixer, s.ffn, s.norm, s.n_layers,
                            self.rng.choice(self.space["d_models"]), s.n_heads,
                            s.n_experts).canonical()
        if f == "n_heads":
            return ArchSpec(s.mixer, s.ffn, s.norm, s.n_layers, s.d_model,
                            self.rng.choice(self.space["n_heads"]),
                            s.n_experts).canonical()
        return ArchSpec(s.mixer, s.ffn, s.norm, s.n_layers, s.d_model, s.n_heads,
                        self.rng.choice(self.space["n_experts"])).canonical()

    def _crossover(self, a: ArchSpec, b: ArchSpec) -> ArchSpec:
        """逐字段从两个父本里挑（均匀交叉），然后规范化。"""
        pick = lambda x, y: x if self.rng.random() < 0.5 else y  # noqa: E731
        return ArchSpec(
            pick(a.mixer, b.mixer), pick(a.ffn, b.ffn), pick(a.norm, b.norm),
            pick(a.n_layers, b.n_layers), pick(a.d_model, b.d_model),
            pick(a.n_heads, b.n_heads), pick(a.n_experts, b.n_experts),
        ).canonical()

    def _random_from_pool(self) -> ArchSpec | None:
        """从池子里随机取一个没提过的点（池子耗尽则返回 ``None``）。"""
        for _ in range(len(self.pool) * 2):
            s = self.rng.choice(self.pool).canonical()
            if s.spec_id not in self.proposed:
                self.proposed.add(s.spec_id)
                return s
        for s in self.pool:  # 兜底：顺序扫一遍
            c = s.canonical()
            if c.spec_id not in self.proposed:
                self.proposed.add(c.spec_id)
                return c
        return None


class RandomSearcher(Searcher):
    """确定性洗牌后依次取。**基线**。"""

    name = "random"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.order = list(self.pool)
        self.rng.shuffle(self.order)
        self.cursor = 0

    def propose(self, history: list[Trial]) -> ArchSpec | None:
        self.pending_parents = []  # 随机搜索没有父本：点与点之间互不相关
        while self.cursor < len(self.order):
            spec = self.order[self.cursor].canonical()
            self.cursor += 1
            if spec.spec_id not in self.proposed:
                self.proposed.add(spec.spec_id)
                return spec
        return None


class EvolutionSearcher(Searcher):
    """NSGA-II 风格：非支配排序选精英 + 拥挤距离补多样，变异/交叉产生后代。"""

    name = "evolution"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.population = int(self.cfg.get("population", 6))
        self.mutation_rate = float(self.cfg.get("mutation_rate", 0.35))
        self.elite = int(self.cfg.get("elite", 2))
        self._queue: list[ArchSpec] = []

    def _parents(self, history: list[Trial]) -> list[ArchSpec]:
        """按「前沿优先、其次拥挤距离」挑父本。"""
        if not history:
            return []
        fronts = non_dominated_sort(history, self.objectives)
        dist = crowding_distance(history, self.objectives)
        by_id = {t.trial_id: t for t in history}
        chosen: list[ArchSpec] = []
        for front in fronts:
            ordered = sorted(front, key=lambda tid: (-dist[tid], tid))
            for tid in ordered:
                chosen.append(by_id[tid].spec)
                if len(chosen) >= self.population:
                    return chosen
        return chosen

    def _refill(self, history: list[Trial], parents: list[ArchSpec]) -> None:
        """补一批后代进队列。``parents`` 由调用方算好传进来（别在循环里重复算）。"""
        if not parents:
            return
        target = max(self.population, 2)
        for _ in range(target * 3):
            if len(self._queue) >= target:
                break
            if self.rng.random() < self.mutation_rate or len(parents) < 2:
                child = self._mutate(self.rng.choice(parents))
            else:
                a, b = self.rng.sample(parents, 2)
                child = self._crossover(a, b)
            c = self._fresh(child)
            if c is not None:
                self._queue.append(c)

    def propose(self, history: list[Trial]) -> ArchSpec | None:
        # 前 population 个点纯随机 —— 没有种群就没法选择
        if len(history) < self.population:
            self.pending_parents = []
            return self._random_from_pool()
        parents = self._parents(history)
        if not self._queue:
            self._refill(history, parents)
        if self._queue:
            child = self._queue.pop(0)
            pset = {p.spec_id for p in parents}
            self.pending_parents = [t.spec_id for t in history if t.spec_id in pset][:2]
            return child
        self.pending_parents = []
        return self._random_from_pool()


class BayesianSearcher(Searcher):
    """Nadaraya–Watson 代理 + LCB 采集。前 ``n_init`` 个点随机。

    采集准则是 ``mu(x) - kappa·sigma(x)``（在标量化后的最小化目标上）：
    ``mu`` 负责利用、``sigma`` 负责探索，``kappa`` 调两者配比。
    """

    name = "bayesian"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.n_init = int(self.cfg.get("n_init", 8))
        self.kappa = float(self.cfg.get("kappa", 2.0))
        self.n_parents = int(self.cfg.get("n_parents", 4))

    def propose(self, history: list[Trial]) -> ArchSpec | None:
        if len(history) < self.n_init:
            self.pending_parents = []
            return self._random_from_pool()
        from .surrogate import fit, lcb

        model = fit(history, self.objectives)
        candidates = [s.canonical() for s in self.pool if s.spec_id not in self.proposed]
        if not candidates:
            return None
        best = min(candidates, key=lambda s: lcb(model, s, self.kappa))
        self.proposed.add(best.spec_id)
        # 代理是拿哪些点拟合的 —— 落进血统图的 derives 边
        self.pending_parents = [t.trial_id for t in model.source_trials[: self.n_parents]]
        return best


SEARCHERS: dict[str, type[Searcher]] = {
    "random": RandomSearcher,
    "evolution": EvolutionSearcher,
    "bayesian": BayesianSearcher,
}


def make_searcher(
    name: str,
    space: dict[str, Any],
    objectives: Sequence[str],
    search_cfg: dict[str, Any],
    seed: int,
    vocab_size: int,
    seen: set[str] | None = None,
) -> Searcher:
    try:
        cls = SEARCHERS[name]
    except KeyError:
        raise KeyError(f"未注册的搜索器：{name!r}（已注册：{sorted(SEARCHERS)}）") from None
    return cls(space, objectives, search_cfg, seed, vocab_size, seen)
