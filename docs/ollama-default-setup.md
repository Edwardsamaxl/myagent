# Ollama 与 Qwen 默认配置手册（Windows）

本文目标：让项目默认使用本地 `qwen2.5:7b`，避免每次开机手动设置环境变量。

---

## 1. 结论先说

- **不用每次手动设置**，推荐把项目配置写入仓库根目录 `.env`。
- 如果 Ollama 模型在 `E:`，建议固定：
  - `OLLAMA_MODELS=E:\Ollama\models`（系统级或启动脚本里设置）
  - 项目侧 `OLLAMA_BASE_URL=http://127.0.0.1:11435`（若你用 11435 实例）

---

## 2. 项目默认配置（一次性）

在 `e:\CursorProject\myagent` 下新建 `.env`（可从 `.env.example` 复制），至少包含：

```env
MODEL_PROVIDER=ollama
MODEL_NAME=qwen2.5:7b
OLLAMA_BASE_URL=http://127.0.0.1:11435
EVAL_USE_MOCK=false
```

说明：
- `MODEL_PROVIDER/MODEL_NAME`：业务与评测默认都走真实模型；
- `OLLAMA_BASE_URL`：指向你可用的 Ollama 实例地址；
- `EVAL_USE_MOCK=false`：评测脚本不再默认写死 mock。

---

## 3. Ollama 模型目录在 E 盘时（推荐）

### 3.1 永久设置系统变量（一次）

在 PowerShell 执行：

```powershell
setx OLLAMA_MODELS "E:\Ollama\models"
```

执行后需要**新开终端**生效。

### 3.2 启动一个固定端口实例（示例 11435）

```powershell
$env:OLLAMA_MODELS="E:\Ollama\models"
$env:OLLAMA_HOST="127.0.0.1:11435"
ollama serve
```

验证：

```powershell
$env:OLLAMA_HOST="127.0.0.1:11435"
ollama list
```

应能看到 `qwen2.5:7b`。

---

## 4. 常见问题

### 4.1 为什么出现“model not found”？

通常是这两种情况：
- Ollama 正在用另一个模型目录（没指到 `E:\Ollama\models`）；
- 你连的是另一个 Ollama 端口（例如默认 11434，而模型在你开的 11435 实例里）。

### 4.2 为什么之前看到很多 mock？

之前一些离线评测脚本里有硬编码 mock。现已改为：
- 默认读取 `.env`（真实模型）；
- 仅当 `EVAL_USE_MOCK=true` 时才强制 mock。

---

## 5. 推荐日常启动顺序

1. 启动 Ollama（确保指向 E 盘模型目录）；  
2. 进入项目目录，确认 `.env` 已配置；  
3. 启动项目或运行评测脚本。  

这样开机后通常只需要“启动服务”，不用反复改变量。

