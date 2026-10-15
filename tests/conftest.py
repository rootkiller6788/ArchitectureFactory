"""测试夹具。

**一条纪律：贵的流水线只跑一次。** 代理训练是这个项目里唯一昂贵的部分，
所以 ``result`` / ``out_dir`` 都是 **session 级** 的：整套测试共享一次运行，
后续所有断言都从产物上读。这和 L1 的做法一致 —— 让「跑一次、查很多遍」
成为默认，而不是每个测试各自重跑。

``fast_cfg`` 把预算压到最小（3 次/搜索器、12 训练步、32 训练样本），只为验证
**结构与契约**；**「attention 明显强于 ssm」这条核心断言不用它**，而是由
``test_acceptance.py`` 用默认协议单独训练一对架构来验证 —— 那是本项目的命题本身，
不能用一个被削过的配置去证明。

**关于 ``fast_cfg`` 的一个实测教训**：它最初只削了样本数（``n_train 48``），却把
``proxy.steps=80`` / ``batch_size=24`` 原样留着 —— 而代理训练的开销是
``trials × steps × batch`` 的乘积，只削一个因子等于没削。结果 156 条测试跑了
**301 秒**，因为契约测试（只读指纹、两次运行一致性）各自把近满预算的流水线
重跑了 5 遍。削预算要削**乘积**，并且把「跑两次」这类用例的产物**共享**出去
（见 ``two_runs``）：现在同样的测试集合约 1 分钟。
"""

from __future__ import annotations

import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session", autouse=True)
def _chdir_root():
    """所有相对路径（configs/、out/）都以项目根为基准。"""
    old = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def cfg() -> dict:
    from architecturefactory.config import load_config

    return load_config(os.path.join(ROOT, "configs", "default.json"))


@pytest.fixture(scope="session")
def fast_cfg(cfg: dict) -> dict:
    """削过的配置：只验证结构，不验证质量。

    三个因子一起削（次数 × 步数 × 批大小），因为训练开销是它们的乘积；
    只削一个因子等于没削（见模块 docstring 里的实测教训）。
    """
    c = copy.deepcopy(cfg)
    c["search"]["trials"] = 3
    c["search"]["searchers"] = ["random", "bayesian"]
    c["search"]["n_init"] = 2
    c["task"]["n_train"] = 32
    c["task"]["n_test"] = 16
    c["proxy"]["steps"] = 12
    c["proxy"]["batch_size"] = 8
    c["proxy"]["eval_every"] = 12
    return c


@pytest.fixture(scope="session")
def out_dir(fast_cfg: dict, tmp_path_factory) -> str:
    """跑一次流水线并把产物落到一个 session 级临时目录，供读命令与只读契约测试用。"""
    from architecturefactory.pipeline import run_pipeline, write_outputs

    d = str(tmp_path_factory.mktemp("af_out"))
    write_outputs(run_pipeline(fast_cfg), d)
    return d


@pytest.fixture(scope="session")
def two_runs(fast_cfg: dict, tmp_path_factory) -> tuple[str, str]:
    """**同一份配置独立跑两次**，落到两个目录，供所有「复现性」用例共享。

    复现性断言天然需要一对运行，而每次运行都要真训练。若每个用例各自跑两次，
    开销就随用例数线性增长 —— 共享一对运行既省时间，又让断言之间**可比**：
    两个用例看的是同一对产物，而不是各自的一对。
    """
    from architecturefactory.pipeline import run_pipeline, write_outputs

    a = str(tmp_path_factory.mktemp("af_run_a"))
    b = str(tmp_path_factory.mktemp("af_run_b"))
    write_outputs(run_pipeline(fast_cfg), a)
    write_outputs(run_pipeline(fast_cfg), b)
    return a, b


@pytest.fixture(scope="session")
def result(out_dir: str):
    from architecturefactory.pipeline import load_result

    r = load_result(out_dir)
    assert r is not None, "夹具写出的产物应当能被 load_result 读回"
    return r


@pytest.fixture(scope="session")
def task(cfg: dict):
    from architecturefactory.task import build_task

    return build_task(cfg)
