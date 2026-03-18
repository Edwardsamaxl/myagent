# mini-openclaw-agent

一个可在本地运行的最小智能体项目，目标是：
- 使用公开可用模型（优先免费/本地）；
- 提供可替换模型接口（不绑定单一厂商）；
- 保持结构简单，便于后续扩展到更复杂的任务编排。

## 可行性结论（先说结论）

完全可行，而且对个人开发者友好。

如果你只追求本地可运行 + 可换模型接口，技术难度是**中低**，关键挑战不在"能不能做"，而在"如何把边界定义清楚"：
- 最小阶段先做单代理循环、少量工具、可插拔模型适配层；
- 后续再加记忆、任务分解、并发执行、浏览器自动化等。

## 为什么它可行

- 现在有可直接本地跑的模型生态（如 Ollama），以及大量 OpenAI 兼容接口；
- 智能体核心本质是：`模型 + 提示词 + 工具调用循环`；
- 通过一个统一 `ModelProvider` 抽象，就可以在本地模型、云端免费额度模型之间切换。

## 这个项目当前实现了什么

- 可插拔模型层（`ollama` / `openai_compatible` / `mock`）；
- 一个最小 Agent Loop（支持工具调用）；
- 内置工具：
  - `get_time`：获取当前本地时间；
  - `calculate`：安全地做基础算术表达式计算；
- 命令行交互入口（`main.py`）。

## 项目结构

```text
mini-openclaw-agent/
├─ main.py
├─ requirements.txt
├─ .env.example
└─ src/
   └─ mini_openclaw_agent/
      ├─ __init__.py
      ├─ config.py
      ├─ providers.py
      ├─ tools.py
      └─ agent.py
```

## 快速开始

1) 创建并激活虚拟环境（可选但推荐）

```powershell
cd C:\Users\Ed\mini-openclaw-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安装依赖

```powershell
pip install -r requirements.txt
```

3) 配置环境变量

```powershell
copy .env.example .env
```

按需编辑 `.env`（示例见下文）。

4) 启动

```powershell
python main.py
```

输入 `exit` 可退出。

## 环境变量说明

- `MODEL_PROVIDER`
  - `ollama`：本地 Ollama（推荐起步）；
  - `openai_compatible`：任意 OpenAI 兼容接口；
  - `mock`：本地回显调试。
- `MODEL_NAME`
  - 例如 `qwen2.5:7b`（Ollama）；
  - 或兼容接口中的模型名。
- `OLLAMA_BASE_URL`
  - 默认 `http://localhost:11434`。
- `OPENAI_BASE_URL`
  - 默认 `https://api.openai.com`。
- `OPENAI_API_KEY`
  - 仅 `openai_compatible` 模式需要。
- `MAX_STEPS`
  - 单轮最多工具调用步数，默认 `6`。
- `TEMPERATURE`
  - 采样温度，默认 `0.2`。
- `MAX_TOKENS`
  - 单次生成最大 token，默认 `512`。

## 推荐起步配置（本地免费）

在 `.env` 中设置：

```env
MODEL_PROVIDER=ollama
MODEL_NAME=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

并确保你已安装并启动 Ollama，且本地已有对应模型。

## 后续扩展建议

- 增加文件读写、网页检索、命令执行等工具（注意安全沙箱）；
- 加入短期记忆（会话）与长期记忆（向量库）；
- 把 Agent Loop 拆成规划器 + 执行器；
- 加入任务状态机和失败重试策略；
- 用 FastAPI 封装成服务接口，方便接前端。

