"""验收：**本项目的命题是否成立。**

README 的承诺是两条：

1. 同一套 IR 能装配出 ≥4 种结构（Transformer / Mamba / MoE / Hybrid）；
2. 搜索能产出**可比较的架构数据库**（质量 vs 成本的前沿）。

但这两条都是「结构上做到了」。真正要证的是一条更硬的命题：

    **架构是决策变量 —— 在同一个任务、同一份训练预算下，
    不同装配的质量差异是系统性、可复现的，而不是噪声。**

所以本文件的核心是 :func:`test_attention_family_beats_the_heuristic_baseline` 与
:func:`test_ssm_stalls_at_the_heuristic_baseline`。它们**不用被削过的配置**
（``fast_cfg``），而是用默认协议各训练一对架构 —— 因为这是命题本身，
不能用打折的实验去证。

**启发式基线是关键。** ``index-recall`` 里「只会输出最后一个 item」的命中率是
``1/P``。ssm 恰好停在这个数上，说明它**没学到结构**、只学到了启发式；
attention 显著超过它，才说明它真的在做「按位置寻址」。
如果只跟 ``1/vocab`` 比，两者都「远超随机」，结论就成了幻觉。
"""

from __future__ import annotations

import pytest

from architecturefactory.builder import build_from_spec
from architecturefactory.ir import canonical_demo_specs
from architecturefactory.profile import profile_model
from architecturefactory.proxy import train_proxy
from architecturefactory.schema import ArchSpec


@pytest.fixture(scope="session")
def trained(cfg, task):
    """用**默认协议**训练一对对照架构：attention 1 层 vs ssm 1 层。

    除 mixer 外全部相同（d_model / 层数 / ffn / norm / 头数），
    于是差异只可能来自 mixer —— 这是单变量对照。

    取 ``d_model=16``：实测它在默认协议下 5 个种子的 attention 是
    mean 0.925 / min 0.844（见 README 附录 B 的标定表）。更小的 d12 会退化到
    mean 0.588，那是**初始化噪声**而不是架构差异 —— 用它做验收会让这条断言
    变成掷骰子。
    """
    out = {}
    for name, spec in (
        ("attention", ArchSpec("attention", "swiglu", "rmsnorm", 1, 16, n_heads=2)),
        ("ssm", ArchSpec("ssm", "swiglu", "rmsnorm", 1, 16)),
    ):
        m = build_from_spec(spec, task.vocab_size, seed=cfg["seed"], max_seq_len=task.seq_len)
        ev = train_proxy(m, task, cfg, cfg["seed"])
        prof = profile_model(m, task.seq_len)
        ev.params, ev.flops = prof.params, prof.flops
        out[name] = (spec, ev)
    return out


def test_attention_family_beats_the_heuristic_baseline(trained, task):
    """attention 必须**显著超过**启发式基线，否则它没学到按位置寻址。"""
    spec, ev = trained["attention"]
    assert ev.accuracy > task.heuristic + 0.3, (
        f"{spec.spec_id} 只到 {ev.accuracy:.3f}，启发式基线是 {task.heuristic:.3f} —— "
        "没有证据表明它学到了结构"
    )


def test_ssm_stalls_at_the_heuristic_baseline(trained, task):
    """ssm 应当停在启发式基线附近：它只能记住「最近的值」，做不到按位置读。"""
    spec, ev = trained["ssm"]
    assert ev.accuracy < 0.6, f"{spec.spec_id} 达到 {ev.accuracy:.3f} —— 预期它做不到"


def test_the_gap_between_families_is_large(trained):
    """**核心断言**：同一任务、同一预算，两个 mixer 的差距必须是压倒性的。

    这条一旦失败，本项目的命题（架构是决策变量）就不成立 —— 那时候该改的是
    任务设计，不是把阈值调低。
    """
    acc_a = trained["attention"][1].accuracy
    acc_s = trained["ssm"][1].accuracy
    assert acc_a - acc_s > 0.3, f"attention={acc_a:.3f} ssm={acc_s:.3f} 差距不足"


def test_attention_and_ssm_have_comparable_cost(trained):
    """差距不能是「贵得多所以更好」—— 两者参数量必须同量级。"""
    pa = trained["attention"][1].params
    ps = trained["ssm"][1].params
    assert 0.5 < pa / ps < 2.0, f"参数量不可比：attention={pa} ssm={ps}"


def test_training_is_reproducible_with_same_seed(trained, cfg, task):
    """同种子重跑，accuracy 必须**逐位相同** —— 搜索可比性的前提。

    复用 ``trained`` 里那一次 attention 运行当「第一次」，只补跑一次：两次都走
    **默认协议**（不是削过的配置），断言因此有分量，同时省掉一次完整训练。
    这比「随便挑个种子跑两次」更强 —— 它把核心断言依赖的那一次运行本身
    钉在了可复现性上。
    """
    spec, a = trained["attention"]
    b = train_proxy(
        build_from_spec(spec, task.vocab_size, seed=cfg["seed"], max_seq_len=task.seq_len),
        task, cfg, cfg["seed"],
    )
    assert a.accuracy == b.accuracy
    assert a.loss == b.loss
    assert a.curve == b.curve
    # 耗时是唯一允许不同的字段
    assert a.train_seconds >= 0.0


def test_all_four_canonical_architectures_train_without_error(cfg, task):
    """README 承诺的四种结构都要能真的跑完训练（不只是能前向）。"""
    import copy

    small = copy.deepcopy(cfg)
    small["proxy"]["steps"] = 4
    small["proxy"]["eval_every"] = 4
    for name, spec in canonical_demo_specs().items():
        spec = ArchSpec(spec.mixer, spec.ffn, spec.norm, 1, 16,
                        n_heads=spec.n_heads or 2, n_experts=spec.n_experts)
        m = build_from_spec(spec, task.vocab_size, seed=0, max_seq_len=task.seq_len)
        ev = train_proxy(m, task, small, 0)
        assert 0.0 <= ev.accuracy <= 1.0, f"{name} 训练异常"
        assert ev.params == 0  # 训练器不负责成本，由调用方补
