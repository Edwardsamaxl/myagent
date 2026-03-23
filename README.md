# mini-evolving-agent

一个可本地运行、可视化操作的最小 Agent。核心目标：
- 默认使用本地 `ollama/qwen2.5:7b`；
- 通过 Web UI 聊天，不依赖 CLI 操作；
- 支持模型热切换（provider + model_name）；
- 保留 `MEMORY.md` 与 `skills/*.md`，让 Agent 可持续演化；
- 保留工具调用闭环（模型 -> 工具 -> 结果回灌 -> 回复）。

## 分层结构

```text
myagent/
├─ main.py                      # 启动入口（Web）
├─ requirements.txt
├─ .env.example
├─ runtime/                     # 运行时数据（会话等）
├─ workspace/                   # 可进化资产（MEMORY + skills）
└─ src/agent/
   ├─ config.py                 # 配置层
  ├─ core/                     # 领域核心层（loop + memory + skill + session + rag skeleton）
   ├─ llm/                      # 模型提供商层
   ├─ tools/                    # 工具注册层
   ├─ application/              # 应用服务层（编排）
   ├─ interfaces/               # 接口层（Web API + UI）
   ├─ agent.py                  # 兼容导出（指向 core）
   ├─ providers.py              # 兼容导出（指向 llm）
   ├─ service.py                # 兼容导出（指向 application）
   └─ web.py                    # 兼容导出（指向 interfaces）
```

## 快速启动

1) 安装依赖

```powershell
pip install -r requirements.txt
```

2) 配置环境变量

```powershell
copy .env.example .env
```

3) 启动

```powershell
python main.py
```

4) 打开页面

`http://127.0.0.1:7860`

## Web UI 功能

- 聊天窗口（支持工具调用链路）；
- Session ID 切换；
- Provider / Model 切换并立即生效；
- 在线编辑 `MEMORY.md`；
- 在线读取与保存 `skills/*.md`。
- RAG 文档摄入（手工输入文本 -> 入库）。

## 企业级骨架（已预置，待你逐模块加强）

- `core/ingestion.py`：清洗、切块、去重、元数据。
- `core/retrieval.py`：混合检索占位实现（词法 + 简化语义分数）。
- `core/rerank.py`：重排占位实现（可替换为真实 reranker）。
- `core/generation.py`：基于证据生成与拒答骨架。
- `core/evaluation.py`：在线评估记录与聚合指标。
- `core/observability.py`：trace 事件记录。
- `application/rag_agent_service.py`：RAG 编排层。

### 新增 API（框架级）

- `POST /api/ingest`：写入文档到检索库。
- `POST /api/rag`：走 RAG 链路回答问题。
- `GET /api/metrics`：查看在线评估聚合指标。

## 可进化机制

- **长期记忆**：写入 `workspace/MEMORY.md`；
- **技能学习**：保存到 `workspace/skills/*.md`；
- **工具调用**：模型可调用内置工具操作记忆、技能与 workspace 文件；
- **会话沉淀**：历史写入 `runtime/sessions.json`。

## 关键环境变量

- `MODEL_PROVIDER=ollama|openai_compatible|mock`
- `MODEL_NAME=qwen2.5:7b`
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OPENAI_BASE_URL=https://api.openai.com`
- `OPENAI_API_KEY=...`（仅 openai_compatible 需要）
- `WEB_HOST=127.0.0.1`
- `WEB_PORT=7860`
- `DATA_DIR=./runtime`
- `WORKSPACE_DIR=./workspace`
- `RAG_ENABLED=true|false`
- `CHUNK_SIZE=500`
- `CHUNK_OVERLAP=80`
- `RETRIEVAL_TOP_K=6`
- `RERANK_TOP_K=3`

