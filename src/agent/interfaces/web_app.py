from __future__ import annotations

import os
from typing import Any

import requests
from flask import Flask, Response, jsonify, request

from ..application.agent_service import AgentService
from ..config import AgentConfig


def _json_error(message: str, code: str, status: int = 400) -> tuple[Response, int]:
    return jsonify({"error": message, "code": code}), status


_INVALID_CHUNK_PARAMS_MESSAGE = "chunk_size 必须为正整数；chunk_overlap 必须为非负整数，且小于 chunk_size"


def _read_json_object() -> tuple[dict[str, Any] | None, tuple[Response, int] | None]:
    raw = request.get_data(as_text=True)
    if not raw.strip():
        return None, _json_error("请求体不能为空", "empty_body")
    data = request.get_json(force=True, silent=True)
    if data is None or not isinstance(data, dict):
        return None, _json_error("请求体须为合法 JSON 对象", "invalid_json")
    return data, None


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是整数")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        return int(s)
    return int(value)


def _parse_optional_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError("dedup_across_docs 必须是布尔值")


def _resolve_chunk_params(
    payload: dict[str, Any], default_chunk_size: int, default_chunk_overlap: int
) -> tuple[int | None, int | None] | None:
    try:
        chunk_size = _parse_optional_int(payload.get("chunk_size"), "chunk_size")
        chunk_overlap = _parse_optional_int(payload.get("chunk_overlap"), "chunk_overlap")
    except (TypeError, ValueError):
        return None
    if chunk_size is not None and chunk_size <= 0:
        return None
    if chunk_overlap is not None and chunk_overlap < 0:
        return None
    effective_chunk_size = chunk_size if chunk_size is not None else default_chunk_size
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else default_chunk_overlap
    if effective_chunk_overlap >= effective_chunk_size:
        return None
    return chunk_size, chunk_overlap


def _map_rag_upstream_error(exc: Exception) -> tuple[str, str, int]:
    if isinstance(exc, requests.exceptions.HTTPError):
        return "模型服务返回非成功状态码", "rag_upstream_http_error", 502
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return "模型服务不可用，请稍后重试", "rag_model_unavailable", 503
    if isinstance(exc, requests.exceptions.RequestException):
        return "模型服务请求失败", "rag_upstream_request_error", 502
    return "RAG 服务内部错误", "rag_internal_error", 500


def _register_optional_cors(app: Flask) -> None:
    """若设置环境变量 WEB_CORS_ORIGINS，则为响应附加 CORS 头并处理 OPTIONS 预检。"""
    if not os.getenv("WEB_CORS_ORIGINS", "").strip():
        return

    @app.after_request
    def _cors_headers(response: Response) -> Response:
        raw = os.getenv("WEB_CORS_ORIGINS", "").strip()
        if not raw:
            return response
        if raw == "*":
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            allowed = [x.strip() for x in raw.split(",") if x.strip()]
            origin = request.headers.get("Origin")
            if origin and origin in allowed:
                response.headers["Access-Control-Allow-Origin"] = origin
            elif len(allowed) == 1:
                response.headers["Access-Control-Allow-Origin"] = allowed[0]
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.before_request
    def _cors_preflight() -> Response | None:
        if os.getenv("WEB_CORS_ORIGINS", "").strip() and request.method == "OPTIONS":
            if request.path.startswith("/api"):
                return Response(status=204)
        return None


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini Evolving Agent</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
    .left { padding: 16px; border-right: 1px solid #334155; overflow:auto; background:#111827; }
    .right { display:flex; flex-direction:column; }
    .box { background:#1e293b; border:1px solid #334155; border-radius:8px; padding:12px; margin-bottom:12px; }
    label { font-size:12px; color:#94a3b8; display:block; margin-bottom:6px; }
    input, select, textarea, button { width:100%; box-sizing:border-box; margin-bottom:8px; border-radius:6px; border:1px solid #475569; background:#0b1220; color:#e2e8f0; padding:8px; }
    button { background:#1d4ed8; border:none; cursor:pointer; }
    button:hover { background:#1e40af; }
    #chat { flex:1; overflow:auto; padding:16px; }
    .msg { margin-bottom:12px; padding:10px; border-radius:8px; border:1px solid #334155; white-space:pre-wrap; }
    .u { background:#1f2937; }
    .a { background:#13273f; }
    .footer { padding:12px; border-top:1px solid #334155; display:flex; gap:8px; }
    .footer input { flex:1; margin:0; }
    .hint { font-size:12px; color:#94a3b8; }
    .row { display:flex; gap:8px; }
    .row > * { flex:1; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <h3>Agent 设置</h3>
      <div class="box">
        <label>Session ID</label>
        <input id="sessionId" value="default" />
        <label>Provider</label>
        <select id="provider">
          <option value="ollama">ollama</option>
          <option value="openai_compatible">openai_compatible</option>
          <option value="mock">mock</option>
        </select>
        <label>Model Name</label>
        <input id="modelName" value="qwen2.5:7b" />
        <button onclick="saveModel()">保存模型配置</button>
      </div>

      <div class="box">
        <h4>MEMORY.md</h4>
        <textarea id="memoryText" rows="10"></textarea>
        <button onclick="saveMemory()">保存记忆</button>
      </div>

      <div class="box">
        <h4>Skills</h4>
        <div class="row">
          <input id="skillName" placeholder="skill 名称" />
          <button onclick="loadSkill()">读取</button>
        </div>
        <textarea id="skillText" rows="8" placeholder="skill 内容"></textarea>
        <button onclick="saveSkill()">保存 Skill</button>
        <div id="skillsList" class="hint"></div>
      </div>

      <div class="box">
        <h4>RAG 文档摄入</h4>
        <label>Doc ID</label>
        <input id="docId" value="finance-doc-001" />
        <label>Source</label>
        <input id="docSource" value="manual-input" />
        <label>Content</label>
        <textarea id="docContent" rows="8" placeholder="粘贴公告、研报或制度文本"></textarea>
        <button onclick="ingestDoc()">写入检索库</button>
      </div>

      <div class="box">
        <h4>RAG 直连（/api/rag）</h4>
        <label>问题</label>
        <textarea id="ragQuestion" rows="4" placeholder="仅走检索+生成，不经 Agent 工具循环"></textarea>
        <label>top_k（可选，默认用服务端 RETRIEVAL_TOP_K）</label>
        <input id="ragTopK" placeholder="例如 6" />
        <button onclick="ragDirect()">RAG 回答</button>
      </div>
    </div>

    <div class="right">
      <div id="chat"></div>
      <div class="footer">
        <input id="input" placeholder="输入你的问题，支持工具调用、记忆、技能..." />
        <button style="width:180px" onclick="sendMsg()">发送</button>
      </div>
    </div>
  </div>

  <script>
    async function api(path, method="GET", body=null){
      const res = await fetch(path, {
        method,
        headers: {"Content-Type":"application/json"},
        body: body ? JSON.stringify(body) : null
      });
      const text = await res.text();
      if(!res.ok){
        let detail = text;
        try{
          const j = JSON.parse(text);
          if(j && j.error) detail = j.error + (j.code ? " (" + j.code + ")" : "");
        }catch(_){}
        throw new Error(detail || ("HTTP " + res.status));
      }
      return text ? JSON.parse(text) : {};
    }

    function add(role, text){
      const chat = document.getElementById("chat");
      const d = document.createElement("div");
      d.className = "msg " + (role === "user" ? "u":"a");
      d.innerText = (role === "user" ? "你: " : "Agent: ") + text;
      chat.appendChild(d);
      chat.scrollTop = chat.scrollHeight;
    }

    async function init(){
      const s = await api("/api/state");
      document.getElementById("provider").value = s.model_provider;
      document.getElementById("modelName").value = s.model_name;
      document.getElementById("skillsList").innerText = "已安装技能: " + (s.skills.join(", ") || "无");
      const m = await api("/api/memory");
      document.getElementById("memoryText").value = m.content;
      add("assistant", "系统已就绪。你可以直接对话，或先改模型与记忆。");
    }

    async function sendMsg(){
      const msg = document.getElementById("input").value.trim();
      if(!msg) return;
      const session_id = document.getElementById("sessionId").value.trim() || "default";
      add("user", msg);
      document.getElementById("input").value = "";
      try{
        const r = await api("/api/chat", "POST", {session_id, message: msg});
        const tools = r.tool_calls.length ? "\\n[工具] " + r.tool_calls.join(" | ") : "";
        const rag = r.rag && r.rag.citations ? ("\\n[RAG引用]\\n" + r.rag.citations.join("\\n")) : "";
        add("assistant", r.answer + tools + rag);
      }catch(err){
        add("assistant", "请求失败: " + err.message);
      }
    }

    async function saveModel(){
      const provider = document.getElementById("provider").value;
      const model_name = document.getElementById("modelName").value.trim();
      const r = await api("/api/model", "POST", {provider, model_name});
      add("assistant", `模型已更新: ${r.provider}/${r.model_name}`);
    }

    async function saveMemory(){
      const content = document.getElementById("memoryText").value;
      await api("/api/memory", "POST", {content});
      add("assistant", "MEMORY.md 已保存。");
    }

    async function loadSkill(){
      const name = document.getElementById("skillName").value.trim();
      if(!name) return;
      const r = await api("/api/skill/" + encodeURIComponent(name));
      document.getElementById("skillText").value = r.content;
    }

    async function saveSkill(){
      const name = document.getElementById("skillName").value.trim();
      const content = document.getElementById("skillText").value;
      if(!name){ alert("请先输入 skill 名称"); return; }
      const r = await api("/api/skill", "POST", {name, content});
      add("assistant", r.message);
      const s = await api("/api/state");
      document.getElementById("skillsList").innerText = "已安装技能: " + (s.skills.join(", ") || "无");
    }

    async function ingestDoc(){
      const doc_id = document.getElementById("docId").value.trim();
      const source = document.getElementById("docSource").value.trim();
      const content = document.getElementById("docContent").value.trim();
      if(!doc_id || !source || !content){
        alert("doc_id/source/content 不能为空");
        return;
      }
      try{
        const r = await api("/api/ingest", "POST", {doc_id, source, content});
        add("assistant", `文档已入库: ${r.doc_id}, chunks=${r.deduplicated_chunks}, trace=${r.trace_id}`);
      }catch(err){
        add("assistant", "入库失败: " + err.message);
      }
    }

    async function ragDirect(){
      const question = document.getElementById("ragQuestion").value.trim();
      if(!question){ alert("问题不能为空"); return; }
      const tk = document.getElementById("ragTopK").value.trim();
      const body = { question };
      if(tk){ body.top_k = parseInt(tk, 10); if(isNaN(body.top_k)){ alert("top_k 须为整数"); return; } }
      try{
        const r = await api("/api/rag", "POST", body);
        const cites = (r.citations && r.citations.length) ? ("\\n[引用]\\n" + r.citations.join("\\n")) : "";
        add("assistant", r.answer + cites + "\\n(trace=" + r.trace_id + ")");
      }catch(err){
        add("assistant", "RAG 请求失败: " + err.message);
      }
    }

    document.getElementById("input").addEventListener("keydown", (e) => {
      if(e.key === "Enter"){ sendMsg(); }
    });

    init();
  </script>
</body>
</html>
"""


def create_app(config: AgentConfig | None = None) -> Flask:
    cfg = config or AgentConfig.from_env()
    service = AgentService(cfg)
    app = Flask(__name__)
    _register_optional_cors(app)

    @app.get("/")
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/state")
    def state():
        return jsonify(service.get_state())

    @app.post("/api/chat")
    def chat():
        payload, err = _read_json_object()
        if err:
            return err
        message = str(payload.get("message", "")).strip()
        session_id = str(payload.get("session_id", "default")).strip() or "default"
        if not message:
            return _json_error("message 不能为空", "validation_error")
        return jsonify(service.chat(session_id=session_id, user_message=message))

    @app.post("/api/model")
    def update_model():
        payload, err = _read_json_object()
        if err:
            return err
        provider = str(payload.get("provider", "")).strip()
        model_name = str(payload.get("model_name", "")).strip()
        try:
            return jsonify(service.update_model(provider=provider, model_name=model_name))
        except Exception as exc:  # noqa: BLE001
            return _json_error(str(exc), "model_update_failed")

    @app.get("/api/memory")
    def get_memory():
        return jsonify({"content": service.get_memory()})

    @app.post("/api/memory")
    def save_memory():
        payload, err = _read_json_object()
        if err:
            return err
        content = str(payload.get("content", ""))
        service.save_memory(content)
        return jsonify({"message": "ok"})

    @app.get("/api/skill/<name>")
    def get_skill(name: str):
        return jsonify({"name": name, "content": service.get_skill(name)})

    @app.post("/api/skill")
    def save_skill():
        payload, err = _read_json_object()
        if err:
            return err
        name = str(payload.get("name", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not name:
            return _json_error("name 不能为空", "validation_error")
        msg = service.save_skill(name, content)
        return jsonify({"message": msg})

    @app.post("/api/ingest")
    def ingest_document():
        payload, err = _read_json_object()
        if err:
            return err
        doc_id = str(payload.get("doc_id", "")).strip()
        source = str(payload.get("source", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not doc_id or not source or not content:
            return _json_error("doc_id/source/content 不能为空", "validation_error")
        raw_meta = payload.get("metadata")
        doc_metadata: dict[str, str] | None = None
        if raw_meta is not None:
            if not isinstance(raw_meta, dict):
                return _json_error("metadata 须为 JSON 对象", "validation_error")
            doc_metadata = {}
            for k, v in raw_meta.items():
                if v is None:
                    continue
                doc_metadata[str(k)] = v if isinstance(v, str) else str(v)
        parsed_chunk_params = _resolve_chunk_params(
            payload=payload,
            default_chunk_size=service.config.chunk_size,
            default_chunk_overlap=service.config.chunk_overlap,
        )
        if parsed_chunk_params is None:
            return _json_error(_INVALID_CHUNK_PARAMS_MESSAGE, "invalid_chunk_params")
        chunk_size, chunk_overlap = parsed_chunk_params
        try:
            dedup_across_docs = _parse_optional_bool(payload.get("dedup_across_docs"), False)
        except ValueError as exc:
            return _json_error(str(exc), "validation_error")
        return jsonify(
            service.ingest_document(
                doc_id=doc_id,
                source=source,
                content=content,
                doc_metadata=doc_metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                dedup_across_docs=dedup_across_docs,
            )
        )

    @app.post("/api/rag")
    def rag_answer():
        payload, err = _read_json_object()
        if err:
            return err
        question = str(payload.get("question", "")).strip()
        if not question:
            return _json_error("question 不能为空", "validation_error")
        top_k = payload.get("top_k")
        try:
            top_k_value = int(top_k) if top_k is not None else None
        except (TypeError, ValueError):
            return _json_error("top_k 必须是整数", "invalid_top_k")
        try:
            return jsonify(service.rag_answer(question=question, top_k=top_k_value))
        except Exception as exc:  # noqa: BLE001
            message, code, status = _map_rag_upstream_error(exc)
            return _json_error(message, code, status)

    @app.get("/api/metrics")
    def metrics():
        return jsonify(service.get_metrics())

    return app
