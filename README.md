# ArchitectureFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L2 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | ArchitectureFactory |
| 类型 | Factory |
| 真正有意思的核心 | Transformer / Mamba / MoE 组件组合 |

---

## 二、核心命题

**这是 9 个里面我认为架构价值最高的一个。**

千万别写成：

```
if architecture == "transformer":
    ...
elif architecture == "mamba":
    ...
```

应该建立一个 **Architecture IR**——让架构本身成为可解析、可组合、可搜索的数据结构。

## 三、整体架构

```
                   Architecture Spec
                          │
                          ▼
                  Architecture IR
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
    Attention           Mamba               MoE
       │                  │                  │
       ├──── Norm ────────┼──── FFN ─────────┤
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                   Compute Graph
                          │
                          ▼
                       Builder
                          │
                          ▼
                    PyTorch Model
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           Train       Benchmark     Profile
             │            │            │
             └────────────┼────────────┘
                          ▼
                      Evaluator
                          │
                          ▼
                    Architecture DB
```

## 四、Architecture Spec

```
Architecture
├── Embedding
├── Block × N
│   ├── Mixer
│   │   ├── Attention
│   │   ├── SSM
│   │   └── Hybrid
│   │
│   ├── FFN
│   │   ├── MLP
│   │   ├── SwiGLU
│   │   └── MoE
│   │
│   ├── Norm
│   └── Residual
│
└── Head
```

于是：

```
Transformer = Attention + SwiGLU + RMSNorm
Mamba       = SSM       + SwiGLU + RMSNorm
MoE         = Attention + MoE     + RMSNorm
Hybrid      = Attention/SSM + MoE + ...
```

**关键洞察**：这些"不同架构"本质上是同一套组件的不同装配方案。把组件抽象成 IR 节点后，架构设计空间就变成一个可枚举、可组合的搜索空间。

## 五、主流水线

```
Architecture Spec
 ↓
Parse
 ↓
Architecture IR
 ↓
Validate
 ↓
Build Compute Graph
 ↓
Instantiate
 ↓
Train Proxy
 ↓
Profile
 ↓
Benchmark
 ↓
Compare
 ↓
Mutate/Search
 ↺
```

## 六、做到这里，它已经开始像一个微型 NAS 系统

| 能力 | 说明 |
| --- | --- |
| IR 化 | 架构不再是代码分支，而是可序列化的图 |
| 组合 | Mixer × FFN × Norm × Residual 自由装配 |
| 验证 | Build 前做结构合法性校验（维度、连接、参数预算） |
| 代理训练 | 小规模 Train Proxy 快速筛掉劣质结构 |
| Profile | 参数量 / FLOPs / 显存 / 吞吐统一度量 |
| 搜索 | 在 IR 空间上做变异与搜索，结果入库成 Architecture DB |

## 七、阶段性落地计划

- **M0｜Spec 与 IR**：定义 Architecture Spec 的 schema 与 IR 表示，跑通"写 Spec → Parse → 实例化一个 Transformer"（非自动）。
- **M1｜组件库与 Builder**：实现 Attention / SSM / MoE / SwiGLU / RMSNorm 等组件与 Compute Graph Builder。
- **M2｜代理训练与搜索**：接入 Train Proxy / Profile / Benchmark，在 IR 空间做 Mutate + Search。
- **M3｜Architecture DB 与发布**：结果入库、对比可视化、示例与文档。

## 八、验收标准

能端到端完成「ArchitectureFactory」，并在其「核心」（Transformer / Mamba / MoE 组件组合）上给出可复现的量化结果：

- 同一套 IR 能装配出 Transformer / Mamba / MoE / Hybrid 至少四种结构
- 搜索过程可复现，产出可比较的 Architecture DB（质量 vs 成本）

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L2 深化设计稿；可按实际进展修订。_

---

# 附录 A｜骨架与使用

## A.1 零依赖

核心流水线**只用标准库**。唯一的版本要求是 **CPython ≥ 3.12** —— 因为
`math.sumprod`（C 实现的内积）自 3.12 起提供，它是本项目唯一但关键的
性能决策（见附录 C 坑 8）。`torch` 只作为可选 extra 存在，核心一行都不 import。

```
L2-architecturefactory/
├── main.py                      # 零安装入口（把 src/ 加进 sys.path）
├── configs/default.json         # 任务 / 设计空间 / 训练协议 / 搜索预算
├── src/architecturefactory/
│   ├── autograd.py              # 微型张量 autograd（全部算子 + 反向闭包）
│   ├── rng.py                   # sha256 派生确定性随机（禁用内置 hash）
│   ├── schema.py                # ArchSpec / ArchIR / Profile / Evaluation / Trial / Version
│   ├── ir.py                    # Spec→IR 解析、校验、参数量公式、空间展开
│   ├── components.py            # Mixer/FFN/Norm 注册表（按名字装配）
│   ├── builder.py               # IR→Model（只查表，无架构分支）
│   ├── profile.py               # 参数量 / FLOPs / 激活元素数
│   ├── task.py                  # 观测任务 index-recall / induction-recall
│   ├── proxy.py                 # 代理训练（动量 SGD + 全局范数裁剪）
│   ├── pareto.py                # 支配 / 非支配排序 / 拥挤距离 / 超体积 / 标量化
│   ├── surrogate.py             # Nadaraya–Watson 核回归（GP 的零依赖替身）
│   ├── search.py                # random / evolution / bayesian
│   ├── lineage.py               # 搜索血统 DAG（DOT + ASCII）
│   ├── config.py                # 配置校验 + 版本摘要
│   ├── pipeline.py              # 编排 + 读写产物
│   ├── report.py                # markdown 报告 + ASCII 散点
│   └── cli.py                   # 命令行（含自包含 report.html）
└── tests/                       # 162 条测试
```

## A.2 快速开始

```bash
cd L2-architecturefactory

python main.py task        # 看任务定义与两条基线（不需要跑流水线）
python main.py space       # 看设计空间大小与轴
python main.py explain attention-moe-rmsnorm-l2-d16-h4-e2   # 把一个 spec 展开成 IR

python main.py run         # 跑搜索并写 out/（约 4.5 分钟）
python main.py summary     # 版本 / 计数 / 超体积
python main.py frontier    # 帕累托前沿
python main.py compare     # 三个搜索器的对比
python main.py matrix      # 架构数据库
python main.py lineage     # 搜索血统图（ASCII）
python main.py visualize   # 生成自包含 out/report.html

python -m pytest -q        # 162 条测试，约 26 秒
```

## A.3 CLI 契约

**读命令不写盘。** 这不是洁癖：`out/` 里那份结果可能对应另一份配置，
读命令若顺手重跑，用户看到的报告就可能与手上的配置不是一回事 —— 而错误结论
会伪装成正常输出。所以读命令优先从 `out/` 重建（`load_result`），
`out/` 不存在时退回**在内存里跑一遍**（慢，但绝不落盘）。

| 命令 | 写盘 | 需要 out/ | 说明 |
| --- | --- | --- | --- |
| `run` | ✅ | — | 跑流水线，写 `trials/frontier/lineage/version/matrix/report` |
| `visualize` | ✅ | 否（可自跑） | 写自包含 `report.html`（内联 SVG，不引外部资源） |
| `summary` | — | 是（否则重跑） | 版本 / 基线 / 参考点 / 理想点 / 超体积 |
| `frontier` | — | 同上 | 前沿表格 |
| `matrix` | — | 同上 | 架构数据库（`--limit`） |
| `show <spec_id>` | — | 同上 | 单个试验的 spec / 父本 / evaluation |
| `compare` | — | 同上 | 按搜索器对比 |
| `lineage` | — | 同上 | 血统图 ASCII（`--limit`） |
| `space` | — | 否 | 设计空间概览 |
| `task` | — | 否 | 任务定义 + 协议指纹 + 版本摘要 |
| `explain <spec_id>` | — | 否 | spec → IR + 画像；**不训练**，非法架构返回码 1 |

## A.4 产物

| 文件 | 内容 |
| --- | --- |
| `trials.json` | 每条试验的 spec / evaluation / searcher / 父本 |
| `frontier.json` | 前沿 id 列表 + 超体积 + 参考点 + 前沿行 |
| `matrix.json` | 架构数据库（含 counters / searchers / 全部行） |
| `lineage.json` / `lineage.dot` | 搜索血统图（JSON + Graphviz） |
| `version.json` | 内容寻址版本号 + 计数 |
| `report.md` | 完整报告（含 ASCII 散点） |
| `report.html` | 自包含可视化（`visualize` 生成） |

---

# 附录 B｜实测结果

## B.1 环境

Windows / CPython 3.12.5 / 单机。全部数字由 `python main.py run` 一次产出，
产物在 `out/`。

| 项 | 值 |
| --- | --- |
| 版本 | `v1-ed040d679f00`（digest `ed040d679f00`）|
| 任务 | `index-recall`，vocab 12，n_items 4，序列长度 5 |
| 随机基线（均匀猜测） | 0.0833 |
| **启发式基线**（只会输出最后一个 item） | **0.25** |
| 训练协议指纹 | `504492b7fcc3` |
| 设计空间 | 120 个点 |
| 有效试验 / 被拒 | 24 / 0 |
| 前沿规模 | **7** |
| 超体积 | **6712.81** |
| 参考点 | `[0.1141, 11119.4]` |
| 理想点 | `[1.0, 3152]` |

## B.2 代理协议的标定（为什么是这组数）

代理训练是本项目唯一昂贵的部分，协议不能拍脑袋定。下表是**同一配置、5 个种子**
的 attention（d16 / 1 层 / swiglu / rmsnorm）accuracy：

| steps | batch | mean | min | max | 单次耗时 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 80 | 16 | 0.719 | 0.562 | 0.844 | 3.6 s | 可用但方差大 |
| 48 | 24 | 0.263 | 0.188 | 0.312 | 2.8 s | **失败**（停在基线） |
| 64 | 24 | 0.769 | 0.531 | 0.969 | 4.4 s | 勉强 |
| **80** | **24** | **0.925** | **0.844** | **1.000** | **5.5 s** | **采用** |
| 160 | 16 | 0.906 | 0.688 | 1.000 | 7.1 s | 好但更贵 |

三条结论：

1. **batch 比 steps 更关键。** `b24/s80` 的 mean 0.925 好过 `b16/s160` 的 0.906，
   而计算量只有 **75%** —— 梯度方差小，模型才能可靠地「开窍」。
2. **lr 不能加大。** lr=0.3 已在边缘：0.5 → mean 0.319，0.8 → 0.156。加大学习率
   会让训练发散而不是更快收敛。
3. **d_model 不能太小。** d12 的 mean 只有 0.588（min 0.438），而 d16 是 0.925。
   用 d12 做搜索会把**初始化噪声**混进「架构差异」里 —— 这是本项目最危险的一类
   错误（见附录 C 坑 11）。

## B.3 家族对比：**架构确实是决策变量**

| mixer | 试验数 | 最好 accuracy | 中位 | 最差 | 最好点 params |
| --- | --- | --- | --- | --- | --- |
| attention | 6 | **1.0000** | 0.9375 | 0.8125 | 8942 |
| hybrid | 15 | **1.0000** | 0.8438 | 0.6875 | 8942 |
| ssm | 3 | 0.2812 | 0.2188 | **0.1562** | 10740 |

**两个家族完全没有重叠**：attention/hybrid 的**最差**成绩（0.6875）也远高于
ssm 的**最好**成绩（0.2812）。而 ssm 的全部三次试验都落在启发式基线 0.25 附近
（0.156 / 0.219 / 0.281）—— 说明它只学到了「输出最近见过的值」，**没有**学到
按位置寻址。

这条差距不是「贵所以好」：见 B.4，前沿最便宜的点（3152 参数）是 hybrid 而不是
ssm，而 ssm 在 10740 参数上也只有 0.2812。

## B.4 帕累托前沿（7 个点）

| spec_id | accuracy | params | 来源 |
| --- | --- | --- | --- |
| `attention-moe-rmsnorm-l1-d20-h2-e2` | 1.0000 | 8942 | evolution |
| `hybrid-moe-rmsnorm-l1-d20-h2-e2` | 1.0000 | 8942 | random |
| `attention-swiglu-layernorm-l1-d20-h2` | 0.9688 | 4740 | evolution |
| `hybrid-swiglu-layernorm-l1-d20-h2` | 0.9688 | 4740 | evolution |
| `attention-swiglu-rmsnorm-l1-d20-h2` | 0.9375 | 4700 | random |
| `hybrid-mlp-layernorm-l1-d16-h4` | 0.8750 | 3744 | evolution |
| `hybrid-swiglu-layernorm-l1-d16-h4` | 0.8438 | **3152** | evolution |

前沿横跨 accuracy **0.844 → 1.000**、params **3152 → 8942** —— 这就是「质量 vs
成本」的权衡曲线：花 2.8 倍参数换 0.156 的准确率。

```text
  accuracy ↑
     1.000 ┐  *        *                                （8942 参数的两个点）
     0.969 │      *        *
     0.938 │          *
     0.875 │              *
     0.844 │                  *                        （3152 参数）
           └──────────────────────────────────────→ params
           3152                          8942
```

`report.md` 里的 ASCII 散点包含全部 24 个点（前沿 `*` / 被支配 `.`）。

## B.5 搜索器对比（同一份预算，8 次各）

| searcher | 有效 | 前沿数 | 最好 acc | 中位 acc | 最好点 params |
| --- | --- | --- | --- | --- | --- |
| random | 8 | 2 | 1.0000 | 0.9375 | 10200 |
| **evolution** | 8 | **5** | 1.0000 | 0.8750 | 8942 |
| bayesian | 8 | 0 | 0.9375 | 0.8750 | 5640 |

有意思的是三者**风格不同**，而不是「谁全面更强」：

- **random** 靠运气撞到了两个极值点（1.0000 与 0.9375），中位数最高，
  但前沿只占 2 个 —— 它没有**收拢**能力。
- **evolution** 贡献了 7 个前沿点里的 5 个。非支配排序 + 拥挤距离天然维护
  前沿的覆盖度，所以它最擅长「把曲线补密」。
- **bayesian** 一个前沿点都没占：它的最好成绩 0.9375 / 5640 参数被
  `attention-swiglu-layernorm-l1-d20-h2`（0.9688 / 4740）支配了。它的代理
  收敛到了「便宜且不错」的区域，但没有去够两端 —— 8 次预算里 6 次花在热身，
  只剩 2 次给代理，样本太薄。

**这正是需要三种搜索器的理由**：它们探索的是前沿的不同部位。

## B.6 复现

```bash
python main.py run       # 应当给出同样的 version_id / 前沿 / 超体积
```

**可逐位复现**：spec_id、IR 指纹、参数量、FLOPs、试验顺序、前沿集合、超体积、
训练得到的 accuracy（同 Python 版本、同机器）。
**不可逐位复现**：`train_seconds` 与 `version.json` 的 `created_at`。
因此报告里**没有任何以耗时排序的结论** —— 耗时只作为参考列出现。

---

# 附录 C｜踩过的坑

编号接续 `python-pipeline-scaffold` skill（该 skill 已记录到 45 号）。

### 46. 没有位置信息时，「归纳」任务在数学上不可解

第一版任务用 `induction-recall`（`[k₁v₁…k_Pv_P k_q] → v_q`），所有架构都卡在
0.42 附近，怎么调 lr / 层数都不动。原因不是训练不足，而是**结构性不可能**：

最后一层的注意力输出是 `Σ_j α_j·v_j`，而 `α_j = softmax((emb(k_q)W_q)·(emb(x_j)W_k))`
**只是查询 token 内容的函数**。键→值的配对每个样本随机，所以不存在一个固定的
双线性型能把「查询键」映到「它后面那个值」。因此该输出与上下文无关，
不可能随上下文给出不同的 `v_q`。**一层注意力在此任务上不可解，无位置嵌入时
多层也不可解**（缺位置就无法表达「往后挪一位」）。

修法两条：补位置嵌入 + 换任务。最终选了 `index-recall`（按位置寻址）：
它需要「查询 token 的值 ↔ 第 q 个位置」的绑定，一层注意力配位置嵌入就能学，
而门控递推结构上做不到。实测 attention 0.95 / ssm 0.22。

**教训**：当所有架构都卡在同一个数上时，先问「这个任务在我的模型类里可解吗」，
再去调超参。

### 47. 基线选错，「学到了」会变成「蒙对了」

`index-recall` 里「只会输出最后一个 item」的命中率是 `1/P = 0.25`，
而均匀猜测是 `1/12 = 0.083`。ssm 恰好停在 0.22 附近。

如果只跟 0.083 比，ssm 也「远超随机」，结论就成了幻觉。所以 `Task` 同时给出
`chance` 与 `heuristic` 两条基线，验收测试断言的是**启发式基线**。

### 48. batch loss 取和 → 梯度被整体裁掉 → 「训练完全不动」

`train_step_loss` 最初把所有样本的 loss **相加**。于是梯度范数正比于 batch 大小，
而 `grad_clip=1.0` 几乎总是把梯度整体缩放到范数 1 —— 参数几乎不更新。

这个坑特别隐蔽：**loss 曲线看起来在降**（因为它在降和），loss 值也「合理」，
但模型什么也没学到。改成取平均后，同样的超参立刻开始收敛。

**教训**：batch loss 一律取平均。取和会让「改 batch 大小」变成改学习率。

### 49. 一步攒完整个 epoch → 白烧 16 倍算力

`train_proxy` 最初每步遍历**全部** `n/batch` 个 batch，累积 loss 后只做一次反向。
结果：同样多的参数更新，却做了 `n/batch` 倍的重复前向反向。

改成「一步 = 一个 batch 的一次更新」后，同样的步数、同样的更新次数，
计算量降到 1/16。**代理训练的全部预算都花在这里**，这一处就决定了整个搜索能跑多大。

### 50. 不同搜索器重复评估同一个架构 → 前沿出现重复点

三个搜索器各自持有 `proposed` 集合，于是它们会各自提议同一个 `spec_id`。
后果有两个：前沿上出现重复点（`frontier.json` 里同一个 id 出现两次），
以及白白烧掉三分之一的训练预算。

修法：三个搜索器**共享**同一个 `seen` 集合。架构数据库只有一份。

### 51. `_mutate` 没有规范化 → 「发现了新架构」是假的

`_crossover` 返回 `.canonical()`，但 `_mutate` 没有。于是把 mixer 从 `attention`
改成 `ssm` 时，`n_heads` 仍是 4 —— `spec_id` 变了（因为 canonical 只在算 id 时
生效），但 IR 完全一样。搜索会以为自己发现了新架构。

**教训**：凡是搜索器吐出去的 spec，出口处一律规范化。这条由测试钉住
（`test_evolution_mutates_within_space`）。

### 52. matmul 占 71% 运行时间 —— 而标准库里有 C 速度的内积

cProfile 显示 `matmul` 的前向 + 反向占整条代理训练 **71%** 的时间。纯 Python
三重循环的矩阵乘是这个项目唯一的性能瓶颈。

用 `math.sumprod`（**CPython 3.12 起提供的标准库函数**，C 实现）重写前向与反向，
列优先取用以保证局部性：

| | 三重循环 | `math.sumprod` | 加速 |
| --- | --- | --- | --- |
| 前向 `[9,24]@[24,48]` | 435.6 µs | 167.8 µs | **2.6×** |
| 反向（两个梯度） | 845.0 µs | 361.6 µs | **2.3×** |

关键点是**没有引入任何第三方依赖** —— 这保住了整个工作区「零依赖脚手架」的性质。
代价是要求 CPython ≥ 3.12。

### 53. 参数量公式与实例化漂移

`ir.estimate_params`（构建前校验用）与 `Model.num_params`（实例化后实测）是
分开写的两份实现。加了位置嵌入之后，公式那份没有跟着加 —— 于是校验说没超预算，
装配出来却超了。

这类漂移是**静默的**：没有报错，只是预算约束失效。修法是让测试对**空间里全部
120 个点**逐点核对 `实测 == 公式`，而不是抽查几个。

### 54. 用不稳定的配置去证明架构差异 → 把初始化噪声当成了结论

d12 在默认协议下 5 个种子的 attention accuracy 是 mean 0.588 / min 0.438，
而 d16 是 0.925 / 0.844。用 d12 做验收，断言「attention > 基线 + 0.3」会
**随机地失败** —— 那不是架构差异，是掷骰子。

**教训**：验收测试必须跑在**标定过的**配置上，而且标定表要写进文档。
否则「测试通过」只说明这次运气好。

### 55. 测试自己算错了超体积

手算两点超体积时我把并集算成了 10，代码给出 8。复核后是**我对代码错**：
两个支配盒重叠 3，并集 = 5 + 6 − 3 = 8。

**教训**：超体积是对外唯一的单一数字，必须用独立可验的算例钉住
（`test_hypervolume_single_point_hand_computed` / `..._two_points_hand_computed`），
不能只做「跑一遍看看」。

### 56. 确定性测试把耗时字段也算进去了

`trials.json` 里带 `train_seconds`，于是「同配置重跑产物逐位相同」这条断言
必然失败。正确做法是**递归剔除**耗时字段再比，而不是放弃这条断言 ——
因为「除耗时外全部可复现」正是本项目对外承诺的口径。

### 57. 测试夹具「只削了一个因子」

`fast_cfg` 一开始只把 `n_train` 从 96 削到 48，却把 `proxy.steps=80` /
`batch_size=24` 原样留着。代理训练的开销是 `trials × steps × batch` 的**乘积**，
只削一个因子等于没削 —— 156 条测试跑了 **301 秒**（5:01），其中真正在算的是
契约测试反复重跑的那 5 遍近满预算流水线。

两处修改把同一批测试压到 **26 秒**（≈12×）：

1. **削乘积**：`fast_cfg` 同时降到 `steps=12 / batch_size=8 / n_train=32 / n_test=16`
   （外加 `eval_every=12`，否则曲线只有一个点）；
2. **共享「一对运行」**：复现性断言天然需要一对产物，新增 session 级
   `two_runs` 夹具跑一次、两个用例共用；验收侧的同种子复现测试改为复用
   `trained` 夹具里那次 attention 运行 + 补跑一次。

**教训**：削预算要削**乘积**，不是削某一项；而且凡是「需要 N 次运行才能断言」的
性质，产物就该提到 session 级夹具里共享 —— 否则套件耗时会随用例数线性增长。
省下来的时间要留给**不能被削**的那部分：验收测试仍跑默认协议（`trained` 8.1s、
同种子复现 5.0s），因为那正是本项目的命题，打折就等于没证。

## C.1 不变量清单

| # | 不变量 | 由哪条测试钉住 |
| --- | --- | --- |
| 1 | 参数量公式 == 实测（全 120 个点） | `test_params_formula_matches_instantiation_for_whole_space` |
| 2 | 装配只查表、不分支 | `test_assembler_uses_registry_not_branches` |
| 3 | `hybrid` 不在组件注册表里 | `test_hybrid_is_not_a_component` |
| 4 | 初始化种子 = (seed, IR 指纹, 参数名) | `test_init_stream_keyed_by_ir_and_name` |
| 5 | 归一化权重初始化为 1、偏置为 0 | `test_norm_weights_start_at_one_and_biases_at_zero` |
| 6 | 非法 IR 拒绝装配 | `test_illegal_ir_refuses_to_build` |
| 7 | batch loss 取平均不取和 | `test_batch_loss_is_averaged_not_summed` |
| 8 | 版本号**不**随 search/proxy 配置变 | `test_version_digest_ignores_search_config` |
| 9 | 协议指纹**随** proxy 配置变 | `test_protocol_tag_changes_with_proxy` |
| 10 | 跨搜索器共享去重集合 | `test_searchers_share_the_seen_set` |
| 11 | 前沿无重复 spec_id | `test_frontier_has_no_duplicate_spec_ids` |
| 12 | 前沿点不被任何试验支配 | `test_frontier_entries_are_non_dominated` |
| 13 | 空间展开确定且去重（120 个点） | `test_space_expansion_is_deterministic_and_deduped` |
| 14 | 超体积手算校验 | `test_hypervolume_single_point_hand_computed`、`..._two_points_...` |
| 15 | 加被支配点不改变超体积 | `test_hypervolume_dominated_point_adds_nothing` |
| 16 | 读命令不改产物目录（md5 指纹） | `test_readonly_commands_do_not_write` |
| 17 | `visualize` 是唯一写命令且 HTML 自包含 | `test_visualize_is_the_write_command` |
| 18 | 同配置重跑逐位一致（剔除耗时） | `test_run_is_deterministic` |
| 19 | 预算超限变成拒绝记录，不冒泡 | `test_rejected_trials_do_not_kill_the_pipeline` |
| 20 | 同种子重跑逐位一致（**默认协议**，非削过的配置） | `test_training_is_reproducible_with_same_seed` |
| 21 | 昂贵产物跨用例共享，不重复跑流水线 | `two_runs` / `out_dir` / `trained` 均为 session 级 |
| 20 | **attention 显著超过启发式基线** | `test_attention_family_beats_the_heuristic_baseline` |
| 21 | **ssm 停在启发式基线附近** | `test_ssm_stalls_at_the_heuristic_baseline` |
| 22 | **两家族差距 > 0.3 且成本可比** | `test_the_gap_between_families_is_large`、`..._comparable_cost` |
| 23 | 四种经典结构都能跑完训练 | `test_all_four_canonical_architectures_train_without_error` |

第 20–23 条是本项目的**命题本身**：如果它们失败，该改的是任务设计，
不是把阈值调低。

