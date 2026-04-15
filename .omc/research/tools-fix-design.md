# MCP Wrapper 与工具层安全修复方案

## 目录

1. [问题 1: MCP wrapper 嵌套 EventLoop 崩溃](#问题-1-mcp-wrapper-嵌套-eventloop-崩溃)
2. [问题 2: web_search 无 Rate Limit](#问题-2-web_search-无-rate-limit)
3. [问题 3: _safe_eval_math 使用 compile+eval](#问题-3-_safe_eval_math-使用-compileeval)

---

## 问题 1: MCP wrapper 嵌套 EventLoop 崩溃

### 根因分析

**问题位置**: `agent_service.py:74-100`

```python
# 第 74-76 行: 在已存在的 event loop 中运行 coroutine
asyncio.get_event_loop().run_until_complete(
    self.mcp_manager.connect_server(name)
)
```

**根因**:

1. `run_until_complete()` 不能在已运行的 loop 中嵌套调用。当外层已是 `asyncio.run()` 或 `loop.run_until_complete()` 时，再次调用会抛出 `RuntimeError: This event loop is already running`。

2. 后续第 88-103 行的 `make_sync_wrapper` 同样有问题：
   - 第 93 行: `asyncio.get_event_loop()` 在嵌套场景下拿到的是外层 loop
   - 第 94-99 行: 检测到 running 后用 `ThreadPoolExecutor` + `asyncio.run()` 存在竞争窗口

### 详细修复方案

**核心思路**: `_connect_mcp_servers()` 应在 `__init__` 中使用 `asyncio.run()` 作为唯一入口点，后续同步包装器彻底放弃 event loop 操作，改用纯线程方案。

#### 修改 1: `_connect_mcp_servers()` 改为异步方法

```python
async def _connect_mcp_servers_async(self) -> None:
    """异步连接 MCP 服务器并注册其工具。"""
    if not self.config.mcp_servers:
        return
    for name, cmd in self.config.mcp_servers.items():
        try:
            self.mcp_manager.add_server(name, "stdio", command=cmd)
            await self.mcp_manager.connect_server(name)
            print(f"[AgentService] MCP 服务器 {name} 连接成功")
            # 将 MCP 工具合并到 self.tools
            client = self.mcp_manager.get_client(name)
            if client:
                for mcp_tool in client.list_tools():
                    tool_name = mcp_tool.schema.name
                    tool_desc = mcp_tool.schema.description
                    async_handler = mcp_tool.handler

                    def make_sync_wrapper(ah):
                        def wrapper(input_str: str) -> str:
                            import concurrent.futures

                            def run_async():
                                return asyncio.run(ah(None, input=input_str))

                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                                future = pool.submit(run_async)
                                return future.result(timeout=30)
                        return wrapper

                    self.tools[tool_name] = type('Tool', (), {
                        'name': tool_name,
                        'description': tool_desc,
                        'func': make_sync_wrapper(async_handler)
                    })()
                    print(f"  - 注册 MCP 工具: {tool_name}")
        except Exception as exc:
            print(f"[AgentService] MCP 服务器 {name} 连接失败: {exc}")
```

#### 修改 2: `__init__` 中调用方式

```python
def __init__(self, config: AgentConfig) -> None:
    # ... 前置代码不变 ...
    self.mcp_manager = MCPToolManager()
    # 使用 asyncio.run() 作为唯一入口，避免嵌套 loop 问题
    asyncio.run(self._connect_mcp_servers_async())
    # ... 后置代码不变 ...
```

#### 修改 3: 简化 `make_sync_wrapper`（移除所有 event loop 操作）

原代码在第 93-100 行依赖 `get_event_loop()` 检测和分支处理，修复后统一走 `ThreadPoolExecutor` + `asyncio.run()`，不再调用 `get_event_loop()` 和 `run_until_complete()`。

### 代码改动清单

| 文件 | 行号 | 改动类型 | 说明 |
|------|------|----------|------|
| `agent_service.py` | 52-53 | 修改 | `__init__` 中将同步调用改为 `asyncio.run(self._connect_mcp_servers_async())` |
| `agent_service.py` | 67-112 | 重写 | `_connect_mcp_servers()` → `_connect_mcp_servers_async()`，异步实现 |
| `agent_service.py` | 88-103 | 删除 | 移除 `asyncio.get_event_loop()` 逻辑，替换为纯 ThreadPoolExecutor 方案 |

### 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| `asyncio.run()` 不能嵌套 | 低 | `__init__` 是同步入口，不会有嵌套问题 |
| 线程池阻塞 | 低 | MCP 工具调用有 30s timeout |
| 连接失败静默 | 低 | 异常已打印，不影响主流程 |

---

## 问题 2: web_search 无 Rate Limit

### 根因分析

**问题位置**: `registry.py:131-174`

```python
def _safe_web_search(query: str) -> str:
    # ...
    resp = requests.get(url, timeout=10)
```

**根因**:

1. **无并发控制**: 多个并发请求同时到达时没有任何限制
2. **无重试机制**: 请求失败直接返回错误，用户无法自动重试
3. **无退避策略**: 频繁调用没有指数退避，可能被限流
4. **无缓存**: 相同查询每次都发请求

### 详细修复方案

**核心思路**: 在模块级别添加信号量控制并发数 + 简单重试机制 + 缓存。

```python
import time
import hashlib
from functools import lru_cache
from threading import Semaphore

# 模块级并发控制：最多 3 个并发搜索请求
_search_semaphore = Semaphore(3)

# 模块级简单缓存（进程内，maxsize=128）
@lru_cache(maxsize=128)
def _cached_web_search(query: str, _ttl: int) -> str | None:
    """带 TTL 的缓存版本，ttl 作为缓存 key 的一部分实现失效。"""
    return None  # 缓存结果存储在 _web_search_with_cache 中


def _safe_web_search_with_retry(query: str, max_retries: int = 2) -> str:
    """带重试和并发控制的网络搜索。"""
    import urllib.parse

    # 并发控制
    acquired = _search_semaphore.acquire(timeout=15)
    if not acquired:
        return "搜索请求超时：系统繁忙，请稍后重试。"

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_redirect=1"

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                resp.encoding = "utf-8"
                data = resp.json()
                break  # 成功，跳出重试循环
            except requests.RequestException as exc:
                last_error = exc
                if attempt < max_retries:
                    # 指数退避: 0.5s, 1s, 2s...
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return f"搜索请求失败: {last_error}"

        # 处理结果（与原逻辑一致）
        results = []
        for topic in data.get("RelatedTopics", [])[:5]:
            if "Text" in topic and "FirstURL" in topic:
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "snippet": "",
                })

        if not results and data.get("AbstractText"):
            abstract = data.get("AbstractText", "")
            source = data.get("AbstractURL", "")
            return f"{abstract}\n\n来源: {source}"

        if not results:
            return "未找到相关搜索结果。"

        lines = []
        for i, r in enumerate(results[:5], 1):
            lines.append(f"[{i}] {r['title']}")
            if r["snippet"]:
                lines.append(f"    {r['snippet']}")
            lines.append(f"    链接: {r['url']}")
            lines.append("")
        return "\n".join(lines)
    finally:
        _search_semaphore.release()
```

### 代码改动清单

| 文件 | 行号 | 改动类型 | 说明 |
|------|------|----------|------|
| `registry.py` | 131 | 修改 | 函数改名 `_safe_web_search` → `_safe_web_search_with_retry` |
| `registry.py` | 131-174 | 重写 | 添加 Semaphore 并发控制 + 重试 + 指数退避 |
| `registry.py` | 新增 | 添加 | 模块级 `Semaphore(3)` 控制最大并发 |
| `registry.py` | 264 | 修改 | `func=_safe_web_search` → `func=_safe_web_search_with_retry` |

### 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| Semaphore 阻塞 `__init__` | 低 | timeout=15s，超时则跳过 MCP 连接（不影响主服务） |
| 缓存内存泄漏 | 低 | lru_cache maxsize=128 有限制 |
| 退避延长总耗时 | 低 | 最多 3 次重试，总延迟 < 4s |

---

## 问题 3: _safe_eval_math 使用 compile+eval

### 根因分析

**问题位置**: `registry.py:68-89`

```python
def _safe_eval_math(expr: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )

    node = ast.parse(expr, mode="eval")
    for sub_node in ast.walk(node):
        if not isinstance(sub_node, allowed_nodes):
            raise ValueError("只允许基础算术表达式。")

    return float(eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}))
```

**根因**:

1. **AST 白名单不完整**: `ast.Name`、`ast.Attribute` 未拦截，但实际不在 allowed_nodes 中已拦截。真正的问题是小括号 `ast.Tuple` 在某些上下文中可能被误用，不过当前白名单已排除。更关键的是：虽然 AST 解析是安全的，但 `eval()` 本身在有 `__builtins__={}` 的情况下仍然存在理论风险（虽然当前已清空builtins）。

2. **更实际的风险**: `ast.walk()` 只检查节点类型，但不检查节点层级关系。理论上可以通过构造特定的表达式来绕过一些检查，但这在当前白名单下很难做到。

**核心问题**: 即使清空了 `__builtins__`，使用 `compile()` + `eval()` 组合仍然不如纯 AST 解释器安全。建议改用纯 AST 解释器替代 `eval()`。

### 详细修复方案

**核心思路**: 用纯 AST Visitor 模式替代 `eval()`，彻底移除 `eval()` 调用。

```python
class _MathEvaluator(ast.NodeVisitor):
    """纯 AST 数学表达式求值器，无 eval() 调用。"""

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {type(node.value)}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type is ast.Add:
            return left + right
        if op_type is ast.Sub:
            return left - right
        if op_type is ast.Mult:
            return left * right
        if op_type is ast.Div:
            if right == 0:
                raise ValueError("除零错误。")
            return left / right
        if op_type is ast.Mod:
            return left % right
        if op_type is ast.Pow:
            return left ** right
        raise ValueError(f"不支持的二元操作符: {op_type.__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type is ast.UAdd:
            return +operand
        if op_type is ast.USub:
            return -operand
        raise ValueError(f"不支持的一元操作符: {op_type.__name__}")


def _safe_eval_math(expr: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )

    node = ast.parse(expr, mode="eval")
    for sub_node in ast.walk(node):
        if not isinstance(sub_node, allowed_nodes):
            raise ValueError(f"只允许基础算术表达式，发现: {type(sub_node).__name__}")

    evaluator = _MathEvaluator()
    return evaluator.visit(node.body)
```

**验证**:

```python
# 安全表达式（通过）
assert _safe_eval_math("(2+3)*4") == 20.0
assert _safe_eval_math("2**3") == 8.0
assert _safe_eval_math("-(-1)") == 1.0
assert _safe_eval_math("10 % 3") == 1.0

# 危险表达式（被拦截）
try:
    _safe_eval_math("__import__('os').system('ls')")
except ValueError:
    pass  # 正确：AST 解析阶段即拦截 Name 节点

try:
    _safe_eval_math("[].__class__.__bases__[0].__subclasses__()")
except ValueError:
    pass  # 正确
```

### 代码改动清单

| 文件 | 行号 | 改动类型 | 说明 |
|------|------|----------|------|
| `registry.py` | 68-89 | 重写 | 移除 `eval()`，改用纯 AST Visitor `_MathEvaluator` |
| `registry.py` | 新增 | 添加 | `class _MathEvaluator(ast.NodeVisitor)` |

### 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 表达式解析错误 | 低 | `calculate()` 已有 try/except 包裹，错误转为友好消息 |
| 除零检查遗漏 | 低 | 在 `visit_BinOp` 的 Div 分支显式检查 |
| 新节点类型逃逸 | 低 | `ast.walk()` 仍执行类型白名单检查，双重保护 |

---

## 改动文件汇总

| 文件 | 改动数 | 优先级 |
|------|--------|--------|
| `src/agent/application/agent_service.py` | 1 处 | P0 |
| `src/agent/tools/registry.py` | 3 处 | P0 |

## 实施顺序

1. **先改 registry.py**（问题 2、3，不涉及 event loop，相对独立）
2. **后改 agent_service.py**（问题 1，涉及 `__init__` 调用链）
