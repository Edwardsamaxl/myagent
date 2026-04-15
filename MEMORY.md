# 项目记忆

## 项目概述
**Fin-agent**: 面向金融场景的 RAG + Agent 项目。
目标：展示 RAG pipeline 优化能力、Agent 架构设计能力、工程实现能力。

## 技术栈
- **Embedding**: `qwen3-embedding` (4096维, ollama) → 降级时自动切纯词频
- **检索**: 混合检索 lexical + tfidf + embedding + numeric（RRF / weighted_sum 可配）
- **重排**: HuggingFaceReranker (CrossEncoder, apply_softmax=True)
- **生成**: GroundedGenerator（基于证据生成 + 拒答）
- **Agent**: agent_loop.py（带 PLANNER_PROMPT + plan: 思考链 + MAX_LOOP_TURNS）
- **LLM**: anthropic_compatible (MiniMax-M2.7 via api.minimaxi.com)
- **Embedding Provider**: openai_compatible / ollama（支持 fallback）

## 关键文件
| 文件 | 作用 |
|------|------|
| `src/agent/core/retrieval.py` | 混合检索核心（lexical/tfidf/embedding + numeric数字通道 + set_weights动态调权） |
| `src/agent/core/rerank.py` | 重排（HuggingFaceReranker） |
## 路由测试套件 (2026-04-12)
| 文件 | 作用 |
|------|------|
| `tests/unit/test_routing_intent_classifier.py` | 路由测试：101 tests（intent分类/router决策/multi_agent链路/问题数据集） |
| `tests/ROUTING_TESTS_README.md` | 测试指南，供其他 agent 查阅 |
| `tests/unit/test_agent_router.py` | 原有 router 测试：17 tests |

**路由链路**: Query → classify_intent() → AgentRouter.decide() → Route(ReAct/Coordinator/Clarify) → Coordinator.run() → Worker.execute() → Synthesizer

**IntentTier**: TOOL_ONLY / KNOWLEDGE / MIXED / CHITCHAT / AMBIGUOUS / OOS
| `src/agent/core/agent_loop.py` | Agent规划层（PLANNER_PROMPT + plan:思考链 + MAX_LOOP_TURNS + 工具错误处理） |
| `src/agent/core/dialogue/query_rewrite.py` | 查询改写（rule/llm/hybrid/hyde/expand/hyde_expand 模式） |
| `src/agent/service/agent_service.py` | chat入口（RAG在agent.run()前调用，retrieval_hits统一注入） |
| `scripts/eval_retrieval.py` | 检索评估脚本 |
| `data/eval/eval_records/retrieval_results.jsonl` | 评估结果 |

## 当前 Recall@3: 0.36
- ✅ 实体类（股票代码、公司名、地址）: 100%
- ✅ 重大事项类（会计师事务所、董事会秘书）: 良好
- ❌ 数值类（营业收入、净利润、每股收益）: recall=0（根因：单位不匹配，元 vs 万元）

## 已解决问题
1. ✅ 数值召回通道：`_extract_numbers` + `_numeric_score` + `numeric_weight` + `QueryTypeClassifier`
2. ✅ Agent规划层：`agent_loop.py` 带 PLANNER_PROMPT + plan:思考链
3. ✅ RAG↔Agent桥接：`agent_service.py` 重构，RAG在agent.run()前调用
4. ✅ Query Rewrite增强：HyDE / Multi-Query / HyDE+Expand 模式

## 待解决问题
1. **P1**: 数值类recall=0根因：单位不匹配（expected=元，chunk=万元），可在query rewrite时规范化数字单位
2. **P1**: MiniMax API 当前可能500，等恢复

## 文档
- `docs/RAG_CHAIN_FIX_REPORT_2026-04-08.md` - 昨日修复详情
- `docs/RAG_AGENT_FIX_PLAN_2026-04-09.md` - 今日计划与进度
- `docs/thesis/00-thesis-structure.md` - 论文结构（用户撰写中）

## 配置关键
```env
RETRIEVAL_LEXICAL_WEIGHT=0.35
RETRIEVAL_TFIDF_WEIGHT=0.25
RETRIEVAL_EMBEDDING_WEIGHT=0.40   # embedding不可用时自动降级
RETRIEVAL_NUMERIC_WEIGHT=0.0     # 可设0.1-0.3启用数值通道
RETRIEVAL_TOP_K=6
RERANK_TOP_K=3
SPARSE_MODE=tfidf  # 或 bm25 / tfidf_bm25
FUSION_MODE=weighted_sum  # 或 rrf
MAX_LOOP_TURNS=3  # Agent规划层循环上限
```
