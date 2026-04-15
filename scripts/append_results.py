content = open('docs/ROUTING_EVAL_PLAN_2026-04-14.md', 'r', encoding='utf-8').read()

append = """
### BGE Reranker 健康检查已修复

原健康检查仅验证 HTTP 200，未验证响应内容。修复后增加乱码检测和评分解析验证。实测确认模型返回乱码，已降级为 SimpleReranker。

---

## Phase 2 补充测试（qwen2.5:7b，2026-04-15）

由于 MiniMax API 稳定性问题（500/529 错误率 66.7%），改用本地 qwen2.5:7b 进行路由测试。

### 测试结果

| 指标 | 值 |
|------|-----|
| 整体准确率 | **83.3% (25/30)** 超过 80% 目标 |
| 同义表述 | 4/5 = 80% |
| 多跳推理 | 5/5 = 100% |
| 指代消解 | 3/5 = 60%（最弱项） |
| 隐含意图 | 4/5 = 80% |
| 模糊表述 | 4/5 = 80% |
| 复合意图 | 5/5 = 100% |

### 失败案例分析

| ID | 问题 | 期望 | 实际 | 分析 |
|----|------|------|------|------|
| A3 | 帮我查查那家白酒公司的营收增长情况 | SINGLE_STEP | MULTI_STEP | 边界情况，模型认为需要多步 |
| C3 | 继续，上次说的那个分析 | SINGLE_STEP | CLARIFY | history 上下文未完全生效 |
| C4 | 同样条件放到五粮液上呢？ | SINGLE_STEP | CLARIFY | history 上下文未完全生效 |
| D4 | 这公司有没有护城河？ | SINGLE_STEP | CLARIFY | 指代被模型判断为不明 |
| E3 | 业绩最好的季度是哪季 | CLARIFY | SINGLE_STEP | 边界情况，模型认为可回答 |

### 结论

- MiniMax API 稳定性问题导致无法在其上完成有效测试
- qwen2.5:7b 路由准确率 83.3%，证明路由逻辑正确
- 指代消解类问题（需要 history）是主要弱项，建议优化上下文传递

---

## 简历量化指标（基于实测数据）

```markdown
- 实现 LLM-Driven Routing 系统，路由准确率 83.3%（qwen2.5:7b 本地模型，真实 API 测试，30 类问题覆盖）
- RAG 检索链路端到端延迟 P50=6749ms，P95=16216ms（2710 chunks 真实测量）
- 检索阶段 P50=5904ms，占总延迟 87%，为主要瓶颈
- Embedding 模型 qwen3-embedding（4096 维）已集成 Ollama
- Reranker 当前降级为 SimpleReranker，BGE Reranker Ollama 部署异常已检测并降级
- Query Rewrite 支持 rule/llm/hyde/expand/hyde_expand 五种模式
```

"""

# Insert before '## Phase 3'
marker = '## Phase 3: Rewrite Ablation'
idx = content.find(marker)
if idx > 0:
    new_content = content[:idx] + append + '\n' + content[idx:]
    open('docs/ROUTING_EVAL_PLAN_2026-04-14.md', 'w', encoding='utf-8').write(new_content)
    print('Appended successfully')
else:
    print('Marker not found, appending at end')
    open('docs/ROUTING_EVAL_PLAN_2026-04-14.md', 'a', encoding='utf-8').write(append)
    print('Appended at end')
