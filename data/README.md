# data/ —— 外部输入

本项目**不读外部输入**：ArchitectureFactory 的世界 / 种群 / 证据都由 `configs/default.json` 的口径
在运行时自造，`src/architecturefactory/` 里没有任何地方引用 `data/`。
这个目录保留，是为了与工作区其余项目保持一致的目录结构。

如果你要接入真实输入：把 JSON 规格放进这个目录，并在 `configs/default.json` 里指向它。
