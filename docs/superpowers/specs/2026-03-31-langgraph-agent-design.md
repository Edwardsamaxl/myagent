# P0: LangGraph Agent 规划层设计

## 目标

将 `SimpleAgent` 从"见到工具名就调用"的随机循环，改造为基于 **LangGraph** 的 **ReAct + Tool-calling** 架构，具备规划路由能力。

## 核心设计

### 状态机

```
start → planning → routing → executing → observing → routing (loop)
                              ↓
                         generation → end
```

- **planning**：LLM 决定是否需要工具调用，输出 `ActionPlan`
- **routing**：根据 planning 结果路由到工具或 generation
- **executing**：执行工具，注入 state
- **observing**：收集执行结果
- **generation**：无工具调用时直接生成回复

### Tool-calling 机制

- 使用 `create_react_agent` + MiniMax-M2.7 native tool-calling
- 模型输出结构化 `{name, arguments}`，LangGraph 解析执行
- 不依赖 action 字符串解析，稳定可靠

### 工具列表

- `retrieval(query)` → RAG 检索，返回证据列表
- `generate(answer)` → 最终生成回复
- 工具通过 `tool_registry.py` 统一注册

### 状态定义

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]           # 对话历史
    plan: str | None                     # 当前计划（可选）
    tool_calls: list[dict]               # 本轮工具调用记录
    observations: list[str]              # 工具执行结果
    step: int                            # 当前步数
    should_retrieve: bool                # 是否需要检索
```

### 关键配置

- `MAX_STEPS=6`，防止无限循环
- `temperature=0.2`，减少幻觉
- 每次循环 `step += 1`，超限强制生成回复

## 文件变更

- `src/agent/core/agent_loop.py` → 废弃（SimpleAgent 循环）
- `src/agent/core/planning/` → 新增 LangGraph 状态机
  - `agent_graph.py` → StateGraph 定义
  - `nodes.py` → planning/routing/executing/observing 节点
  - `state.py` → AgentState TypedDict
- `src/agent/tools/registry.py` → 工具注册（已有，补充文档）
- `src/agent/application/rag_agent_service.py` → 对接 LangGraph Agent

## 测试验证

- 端到端对话：问需要检索的问题，验证工具被调用
- 端到端对话：问常识问题，验证跳过检索直接生成
- 循环安全：验证 `MAX_STEPS` 限制生效
