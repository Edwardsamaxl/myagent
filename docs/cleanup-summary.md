# 项目精简清理报告

本文记录本次清理所做的删除与简化。

## 1. 已删除的文档（已superseded）

| 文件 | 原因 |
|------|------|
| `docs/day1-summary.md` | 被 `project-progress-summary.md` 替代 |
| `docs/day2-daily-plan.md` | 被 `project-progress-summary.md` 替代 |
| `docs/day2-final-summary.md` | 被 `project-progress-summary.md` 替代 |
| `docs/day2-ingestion-notes.md` | 被 `project-progress-summary.md` + `coordination.md` 替代 |
| `docs/day3-daily-plan.md` | 被 `project-progress-summary.md` 替代 |
| `docs/week1-finance-agent-plan.md` | 被 `project-progress-summary.md` 替代 |

## 2. 已删除的运行时数据

| 目录/文件 | 原因 |
|-----------|------|
| `runtime/day2/` | 旧评估结果（Day2 ablation studies），不再需要 |
| `runtime/day3/` | 旧评估结果（Day3 ablation studies），不再需要 |
| `data/debug/` | 调试输出文件（chunks txt/jsonl 格式），不再需要 |

## 3. 精简后的项目结构

```
myagent/
├── docs/                      # 项目文档
│   ├── agent-design/          # 技术设计文档
│   ├── coordination.md        # 协调契约（权威版本）
│   ├── project-progress-summary.md  # 项目总览（权威版本）
│   └── ...
├── src/agent/                 # 源代码
│   ├── application/          # 应用层
│   ├── core/                  # 核心模块
│   ├── interfaces/           # Web接口
│   └── llm/                  # LLM providers
├── scripts/                   # 离线评估脚本
├── data/                      # 数据目录
│   ├── raw/finance/          # 原始金融文档
│   ├── eval/                 # 评估集
│   └── README.md
├── runtime/                   # 运行态数据（gitignored）
│   ├── traces.jsonl          # Trace日志
│   ├── eval_records.jsonl    # 评估记录
│   └── sessions.json         # 会话数据
└── main.py                    # 入口
```

## 4. 保留的评估脚本

以下脚本保留用于未来重新运行评估：

- `scripts/eval_week1.py` - Week1 评估集
- `scripts/eval_small_batch.py` - 小批量评估
- `scripts/eval_generation_*.py` - 生成策略评估
- `scripts/eval_rerank_ablation.py` - 重排ablation
- `scripts/eval_bm25_ablation.py` - BM25 ablation
- `scripts/eval_embedding_ab.py` - Embedding A/B
- `scripts/eval_query_rewrite_ablation.py` - query rewrite ablation

这些脚本依赖的 `runtime/day2/` 和 `runtime/day3/` 历史数据已删除，再次运行会产出新的结果文件。

## 5. 当前配置

默认模型已更新为 `MiniMax-M2.7`，使用 `anthropic_compatible` provider。

配置位置：`.env` 文件。
