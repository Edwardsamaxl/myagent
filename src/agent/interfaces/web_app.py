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
  <title>Mini Agent</title>
  <style>
    :root {
      --sidebar: #171717;
      --sidebar-hover: #2f2f2f;
      --main-bg: #ffffff;
      --text: #0d0d0d;
      --text-dim: #6e6e80;
      --border: #e5e5e5;
      --accent: #10a37f;
      --accent-dim: #1a7f64;
      --user-bubble: #f4f4f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Söhne", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background: var(--main-bg);
      height: 100vh;
      overflow: hidden;
    }
    .app { display: flex; height: 100vh; }
    .sidebar {
      width: 260px;
      min-width: 260px;
      background: var(--sidebar);
      color: #ececec;
      display: flex;
      flex-direction: column;
      border-right: 1px solid #2f2f2f;
      min-height: 0;
    }
    .sidebar-scroll {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      -webkit-overflow-scrolling: touch;
    }
    .sidebar-top { padding: 12px; }
    .btn-new {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid #4d4d4f;
      border-radius: 8px;
      background: transparent;
      color: #ececec;
      font-size: 14px;
      cursor: pointer;
      text-align: left;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .btn-new:hover { background: var(--sidebar-hover); }
    .session-line {
      margin-top: 12px;
      padding: 8px 10px;
      font-size: 11px;
      color: #8e8ea0;
      word-break: break-all;
      border-radius: 6px;
      background: #212121;
    }
    .sidebar-actions { padding: 8px 12px 12px; display: flex; flex-direction: column; gap: 8px; }
    .btn-model {
      width: 100%;
      padding: 10px 14px;
      border: 1px solid #4d4d4f;
      border-radius: 8px;
      background: #2a2a2a;
      color: #ececec;
      font-size: 13px;
      cursor: pointer;
      text-align: left;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn-model:hover { background: var(--sidebar-hover); border-color: #6e6e80; }
    .btn-model .ico { font-size: 16px; opacity: 0.9; }
    .session-list-wrap { padding: 0 8px 8px; }
    .session-list-title { font-size: 11px; color: #8e8ea0; padding: 8px 8px 4px; text-transform: uppercase; letter-spacing: 0.04em; }
    #sessionList { list-style: none; margin: 0; padding: 0; }
    .session-item {
      display: flex;
      align-items: center;
      gap: 4px;
      border-radius: 8px;
      margin-bottom: 2px;
      padding: 2px 4px;
    }
    .session-item:hover { background: #2f2f2f; }
    .session-item.active { background: #3b3b3b; }
    .session-item .sel {
      flex: 1;
      min-width: 0;
      text-align: left;
      padding: 8px 6px;
      border: none;
      background: transparent;
      color: #ececec;
      font-size: 13px;
      cursor: pointer;
      border-radius: 6px;
    }
    .session-item .sel .t { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .session-item .sel .sub { font-size: 11px; color: #8e8ea0; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .session-item .icon-btn {
      flex-shrink: 0;
      width: 28px;
      height: 28px;
      border: none;
      border-radius: 6px;
      background: transparent;
      color: #acacbe;
      cursor: pointer;
      font-size: 14px;
      line-height: 1;
    }
    .session-item .icon-btn:hover { background: #4d4d4f; color: #fff; }
    .sidebar-foot {
      flex-shrink: 0;
      padding: 10px 12px;
      font-size: 11px;
      color: #8e8ea0;
      line-height: 1.5;
      border-top: 1px solid #2f2f2f;
    }
    .main { flex: 1; display: flex; flex-direction: column; min-width: 0; background: var(--main-bg); }
    .topbar {
      height: 48px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 16px;
    }
    .topbar-title { font-size: 15px; font-weight: 600; color: var(--text); }
    .composer-rag {
      max-width: 768px;
      margin: 0 auto 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      font-size: 13px;
      color: var(--text-dim);
    }
    .composer-rag .rag-toggle-wrap { display: flex; align-items: center; gap: 8px; }
    .btn-rag {
      position: relative;
      width: 48px;
      height: 28px;
      border-radius: 999px;
      border: none;
      background: #d9d9e3;
      cursor: pointer;
      transition: background 0.2s;
      padding: 0;
    }
    .btn-rag.on { background: var(--accent); }
    .btn-rag::after {
      content: "";
      position: absolute;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #fff;
      top: 3px;
      left: 3px;
      transition: transform 0.2s;
      box-shadow: 0 1px 2px rgba(0,0,0,.15);
    }
    .btn-rag.on::after { transform: translateX(20px); }
    .chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 24px 16px 120px;
      max-width: 768px;
      width: 100%;
      margin: 0 auto;
    }
    .row-msg { display: flex; margin-bottom: 20px; gap: 12px; }
    .row-msg.user { justify-content: flex-end; }
    .avatar {
      width: 28px;
      height: 28px;
      border-radius: 4px;
      flex-shrink: 0;
      font-size: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
    }
    .row-msg.assistant .avatar { background: var(--accent); color: #fff; }
    .row-msg.user .avatar { background: #5436da; color: #fff; order: 2; }
    .bubble {
      max-width: min(100%, 600px);
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 15px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .row-msg.assistant .bubble {
      background: transparent;
      padding-left: 0;
    }
    .row-msg.user .bubble {
      background: var(--user-bubble);
      border: 1px solid var(--border);
    }
    .msg-meta {
      font-size: 12px;
      color: var(--text-dim);
      margin-top: 6px;
      padding-left: 40px;
    }
    .row-msg.user .msg-meta { text-align: right; padding-left: 0; padding-right: 40px; }
    .system-hint {
      text-align: center;
      font-size: 12px;
      color: var(--text-dim);
      margin: 16px 0;
      padding: 8px;
    }
    .composer-wrap {
      position: sticky;
      bottom: 0;
      background: linear-gradient(to top, #fff 70%, transparent);
      padding: 12px 16px 20px;
    }
    .composer {
      max-width: 768px;
      margin: 0 auto;
      display: flex;
      gap: 10px;
      align-items: flex-end;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 10px 12px;
      background: #fff;
      box-shadow: 0 2px 12px rgba(0,0,0,.06);
    }
    .composer textarea {
      flex: 1;
      border: none;
      resize: none;
      font-family: inherit;
      font-size: 15px;
      line-height: 1.5;
      min-height: 24px;
      max-height: 200px;
      outline: none;
    }
    .btn-send {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      border: none;
      background: var(--text);
      color: #fff;
      cursor: pointer;
      flex-shrink: 0;
      font-size: 18px;
      line-height: 1;
    }
    .btn-send:hover { opacity: 0.9; }
    .btn-send:disabled { opacity: 0.35; cursor: not-allowed; }
    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,.45);
      z-index: 100;
      align-items: center;
      justify-content: center;
    }
    .modal-backdrop.show { display: flex; }
    .modal {
      background: #fff;
      border-radius: 12px;
      padding: 24px;
      width: 90%;
      max-width: 420px;
      box-shadow: 0 8px 32px rgba(0,0,0,.2);
    }
    .modal h2 { margin: 0 0 16px; font-size: 18px; }
    .modal label { display: block; font-size: 12px; color: var(--text-dim); margin-bottom: 6px; }
    .modal input, .modal select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 14px;
      margin-bottom: 14px;
    }
    .modal-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 8px; }
    .modal-actions button {
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
      border: 1px solid var(--border);
      background: #fff;
    }
    .modal-actions .primary {
      background: var(--text);
      color: #fff;
      border-color: var(--text);
    }
    details.adv { margin: 0 8px 10px; border: 1px solid #3e3e3e; border-radius: 8px; overflow: hidden; }
    details.adv summary {
      padding: 10px 12px;
      cursor: pointer;
      font-size: 13px;
      color: #c5c5d2;
      background: #212121;
    }
    details.adv .adv-body { padding: 12px; background: #1e1e1e; font-size: 12px; color: #acacbe; }
    details.adv label { color: #8e8ea0; display: block; margin-bottom: 4px; margin-top: 8px; }
    details.adv input, details.adv textarea, details.adv select {
      width: 100%;
      padding: 8px;
      border-radius: 6px;
      border: 1px solid #4d4d4f;
      background: #2b2b2b;
      color: #ececec;
      margin-bottom: 8px;
    }
    details.adv button {
      margin-top: 8px;
      padding: 8px 12px;
      border-radius: 6px;
      border: none;
      background: #3e3e3e;
      color: #ececec;
      cursor: pointer;
    }
    details.adv button:hover { background: #4d4d4f; }
    @media (max-width: 720px) {
      .sidebar { width: 100%; max-width: 100%; position: absolute; z-index: 50; transform: translateX(-100%); }
      .sidebar.open { transform: translateX(0); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="sidebar-scroll">
        <div class="sidebar-top">
          <button type="button" class="btn-new" id="btnNewChat" title="新建对话">
            <span style="font-size:18px;line-height:1">+</span>
            <span>新建对话</span>
          </button>
        </div>
        <div class="session-list-wrap">
          <div class="session-list-title">历史对话</div>
          <ul id="sessionList"></ul>
        </div>
        <div class="sidebar-actions">
          <button type="button" class="btn-model" id="btnOpenModel" title="选择模型与提供商">
            <span class="ico">⚙</span>
            <span>模型与提供商</span>
          </button>
        </div>
        <details class="adv" id="advKb">
        <summary>高级 · 知识库（入库 / RAG 直连）</summary>
        <div class="adv-body">
          <label>Doc ID</label>
          <input id="docId" value="finance-doc-001" />
          <label>Source</label>
          <input id="docSource" value="manual-input" />
          <label>Content</label>
          <textarea id="docContent" rows="5" placeholder="粘贴文本入库"></textarea>
          <button type="button" onclick="ingestDoc()">写入检索库</button>
          <label style="margin-top:12px">RAG 直连问题</label>
          <textarea id="ragQuestion" rows="3" placeholder="仅检索+生成"></textarea>
          <label>top_k（可选）</label>
          <input id="ragTopK" placeholder="默认服务端配置" />
          <button type="button" onclick="ragDirect()">RAG 回答</button>
        </div>
      </details>
      <details class="adv" id="advSkills">
        <summary>高级 · Skills 说明与编辑</summary>
        <div class="adv-body">
          <p style="margin:0 0 10px;line-height:1.5">
            主流产品里类似能力多做成「自定义指令」或 GPT/Agent 配置页；本仓库用 <code>workspace/skills/*.md</code> 文件承载，模型会在上下文中读取。
          </p>
          <div id="skillsList" style="margin-bottom:10px;color:#c5c5d2"></div>
          <label>编辑（名称 + 内容）</label>
          <input id="skillName" placeholder="skill 名称" />
          <div style="display:flex;gap:8px">
            <button type="button" onclick="loadSkill()">读取</button>
            <button type="button" onclick="saveSkill()">保存</button>
          </div>
          <textarea id="skillText" rows="5" placeholder="skill 内容"></textarea>
        </div>
      </details>
      </div>
      <div class="sidebar-foot">
        记忆（MEMORY.md）由后端注入上下文；对话内自动沉淀记忆需编排侧实现。
      </div>
    </aside>
    <main class="main">
      <header class="topbar">
        <span class="topbar-title">Mini Agent</span>
      </header>
      <div class="chat-scroll" id="chat"></div>
      <div class="composer-wrap">
        <div class="composer-rag">
          <span>检索增强（RAG）</span>
          <div class="rag-toggle-wrap">
            <button type="button" class="btn-rag on" id="btnRag" aria-pressed="true" title="开启后语料类问题会检索知识库"></button>
            <span id="ragStateLabel" style="color:var(--accent);font-weight:600;font-size:12px">已开启</span>
          </div>
        </div>
        <div class="composer">
          <textarea id="input" rows="1" placeholder="发消息…（Enter 发送，Shift+Enter 换行）"></textarea>
          <button type="button" class="btn-send" id="btnSend" title="发送">↑</button>
        </div>
      </div>
    </main>
  </div>

  <div class="modal-backdrop" id="modelBackdrop">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modelTitle">
      <h2 id="modelTitle">模型与提供商</h2>
      <label>Provider</label>
      <select id="provider">
        <option value="ollama">ollama</option>
        <option value="openai_compatible">openai_compatible</option>
        <option value="mock">mock</option>
      </select>
      <label>Model Name</label>
      <input id="modelName" value="qwen2.5:7b" />
      <div class="modal-actions">
        <button type="button" id="modelCancel">取消</button>
        <button type="button" class="primary" id="modelSave">保存</button>
      </div>
    </div>
  </div>

  <input type="hidden" id="sessionId" value="" />

  <script>
    async function api(path, method, body) {
      const res = await fetch(path, {
        method: method || "GET",
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

    function newSessionId(){
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
      return "s-" + Date.now() + "-" + Math.random().toString(36).slice(2, 9);
    }

    var __sessionCache = [];

    function currentSessionId(){
      return document.getElementById("sessionId").value.trim();
    }

    function setSession(id){
      document.getElementById("sessionId").value = id;
      localStorage.setItem("ui_last_session", id);
    }

    function getOrder(){
      try { return JSON.parse(localStorage.getItem("ui_session_order") || "[]"); } catch(e){ return []; }
    }
    function saveOrder(arr){ localStorage.setItem("ui_session_order", JSON.stringify(arr)); }

    function getTitles(){
      try { return JSON.parse(localStorage.getItem("ui_session_titles") || "{}"); } catch(e){ return {}; }
    }
    function getTitle(id){
      const t = getTitles()[id];
      return t && t.title ? t.title : "";
    }
    function setTitle(id, title){
      const o = getTitles();
      o[id] = { title: title, ts: Date.now() };
      localStorage.setItem("ui_session_titles", JSON.stringify(o));
    }

    function syncOrderWithServer(sessions){
      const ids = sessions.map(function(x){ return x.id; });
      var order = getOrder().filter(function(id){ return ids.indexOf(id) >= 0; });
      for (var i = 0; i < ids.length; i++){
        if (order.indexOf(ids[i]) < 0) order.push(ids[i]);
      }
      saveOrder(order);
      return order;
    }

    function isRagOn(){
      return document.getElementById("btnRag").classList.contains("on");
    }

    function setRagUi(on){
      const b = document.getElementById("btnRag");
      const lab = document.getElementById("ragStateLabel");
      if(on){
        b.classList.add("on");
        b.setAttribute("aria-pressed","true");
        lab.textContent = "已开启";
        lab.style.color = "var(--accent)";
      } else {
        b.classList.remove("on");
        b.setAttribute("aria-pressed","false");
        lab.textContent = "已关闭";
        lab.style.color = "var(--text-dim)";
      }
      sessionStorage.setItem("ui_use_rag", on ? "1" : "0");
    }

    function scrollChat(){ const el = document.getElementById("chat"); el.scrollTop = el.scrollHeight; }

    function addUser(text){
      const chat = document.getElementById("chat");
      const row = document.createElement("div");
      row.className = "row-msg user";
      row.innerHTML = '<div class="bubble"></div><div class="avatar">我</div>';
      row.querySelector(".bubble").textContent = text;
      chat.appendChild(row);
      scrollChat();
    }

    function addAssistant(text, meta){
      const chat = document.getElementById("chat");
      const row = document.createElement("div");
      row.className = "row-msg assistant";
      const parts = ['<div class="avatar">AI</div><div style="flex:1;min-width:0">','<div class="bubble"></div>'];
      if(meta && (meta.steps != null || (meta.tools && meta.tools.length))){
        let m = [];
        if(meta.steps != null) m.push("步数 " + meta.steps);
        if(meta.tools && meta.tools.length) m.push("工具 " + meta.tools.slice(0,3).join(" · "));
        parts.push('<div class="msg-meta">' + m.join("  ·  ") + '</div>');
      }
      parts.push("</div>");
      row.innerHTML = parts.join("");
      row.querySelector(".bubble").textContent = text;
      chat.appendChild(row);
      scrollChat();
    }

    function addSystem(text){
      const chat = document.getElementById("chat");
      const d = document.createElement("div");
      d.className = "system-hint";
      d.textContent = text;
      chat.appendChild(d);
      scrollChat();
    }

    function clearChat(){
      document.getElementById("chat").innerHTML = "";
    }

    async function refreshSkillsList(){
      try{
        const s = await api("/api/state", "GET");
        document.getElementById("skillsList").textContent =
          "已安装：" + (s.skills && s.skills.length ? s.skills.join(", ") : "无");
      }catch(_){
        document.getElementById("skillsList").textContent = "（无法拉取列表）";
      }
    }

    document.getElementById("advSkills").addEventListener("toggle", function(){
      if(this.open) refreshSkillsList();
    });

    function renderSessionList(){
      const ul = document.getElementById("sessionList");
      ul.innerHTML = "";
      const cur = currentSessionId();
      const order = getOrder();
      const byId = {};
      for (var i = 0; i < __sessionCache.length; i++) byId[__sessionCache[i].id] = __sessionCache[i];
      for (var j = 0; j < order.length; j++){
        const id = order[j];
        const s = byId[id];
        if (!s) continue;
        const li = document.createElement("li");
        li.className = "session-item" + (id === cur ? " active" : "");
        const title = getTitle(id) || (s.preview && s.preview.trim()) || "新对话";
        const sub = (s.message_count || 0) + " 条 · " + id.slice(0, 8) + "…";
        const btnSel = document.createElement("button");
        btnSel.type = "button";
        btnSel.className = "sel";
        btnSel.innerHTML = '<span class="t"></span><span class="sub"></span>';
        btnSel.querySelector(".t").textContent = title;
        btnSel.querySelector(".sub").textContent = sub;
        btnSel.addEventListener("click", function(){ selectSession(id); });
        const btnRen = document.createElement("button");
        btnRen.type = "button";
        btnRen.className = "icon-btn";
        btnRen.title = "重命名";
        btnRen.textContent = "✎";
        btnRen.addEventListener("click", function(e){ e.stopPropagation(); renameSession(id); });
        const btnDel = document.createElement("button");
        btnDel.type = "button";
        btnDel.className = "icon-btn";
        btnDel.title = "删除";
        btnDel.textContent = "×";
        btnDel.addEventListener("click", function(e){ e.stopPropagation(); deleteSessionById(id); });
        li.appendChild(btnSel);
        li.appendChild(btnRen);
        li.appendChild(btnDel);
        ul.appendChild(li);
      }
    }

    async function refreshSessionList(){
      const data = await api("/api/sessions", "GET");
      __sessionCache = data.sessions || [];
      syncOrderWithServer(__sessionCache);
      renderSessionList();
    }

    async function selectSession(id){
      setSession(id);
      const data = await api("/api/session/" + encodeURIComponent(id) + "/history", "GET");
      clearChat();
      const msgs = data.messages || [];
      for (var i = 0; i < msgs.length; i++){
        const m = msgs[i];
        if (m.role === "user") addUser(m.content || "");
        else if (m.role === "assistant") addAssistant(m.content || "", null);
      }
      if (msgs.length === 0){
        addAssistant("在此对话中输入消息即可开始。每条对话互不共享上下文。", null);
      }
      renderSessionList();
    }

    async function createNewChat(){
      const id = newSessionId();
      await api("/api/session", "POST", { session_id: id });
      await refreshSessionList();
      var order = getOrder().filter(function(x){ return x !== id; });
      order.unshift(id);
      saveOrder(order);
      if (!getTitle(id)) setTitle(id, "新对话");
      renderSessionList();
      await selectSession(id);
    }

    async function deleteSessionById(id){
      if (!confirm("删除此对话？服务端历史将一并删除。")) return;
      try{
        await api("/api/session/" + encodeURIComponent(id), "DELETE");
      }catch(e){
        if (e.message.indexOf("404") < 0 && e.message.indexOf("不存在") < 0) { alert(e.message); return; }
      }
      var order = getOrder().filter(function(x){ return x !== id; });
      saveOrder(order);
      var titles = getTitles();
      delete titles[id];
      localStorage.setItem("ui_session_titles", JSON.stringify(titles));
      if (currentSessionId() === id){
        if (order.length > 0) await selectSession(order[0]);
        else await createNewChat();
      } else {
        await refreshSessionList();
      }
    }

    function renameSession(id){
      const cur = getTitle(id) || "";
      const t = prompt("对话名称（仅保存在本机浏览器）", cur);
      if (t === null) return;
      setTitle(id, t.trim() || "对话");
      renderSessionList();
    }

    document.getElementById("btnNewChat").addEventListener("click", function(){
      createNewChat().catch(function(e){ alert(e.message); });
    });

    document.getElementById("btnRag").addEventListener("click", function(){
      setRagUi(!isRagOn());
    });

    document.getElementById("btnOpenModel").addEventListener("click", function(){
      document.getElementById("modelBackdrop").classList.add("show");
    });
    document.getElementById("modelCancel").addEventListener("click", function(){
      document.getElementById("modelBackdrop").classList.remove("show");
    });
    document.getElementById("modelBackdrop").addEventListener("click", function(e){
      if(e.target === this) this.classList.remove("show");
    });
    document.getElementById("modelSave").addEventListener("click", async function(){
      const provider = document.getElementById("provider").value;
      const model_name = document.getElementById("modelName").value.trim();
      try{
        await api("/api/model", "POST", {provider, model_name});
        document.getElementById("modelBackdrop").classList.remove("show");
        addSystem("模型已更新为 " + provider + " / " + model_name);
      }catch(err){
        alert(err.message);
      }
    });

    async function sendMsg(){
      const ta = document.getElementById("input");
      const msg = ta.value.trim();
      if(!msg) return;
      var session_id = currentSessionId();
      if (!session_id){
        await createNewChat();
        session_id = currentSessionId();
      }
      ta.value = "";
      addUser(msg);
      try{
        const body = { session_id: session_id, message: msg, use_rag: isRagOn() };
        const r = await api("/api/chat", "POST", body);
        let extra = "";
        if(r.tool_calls && r.tool_calls.length) extra = "\\n\\n[工具] " + r.tool_calls.join(" | ");
        addAssistant((r.answer || "") + extra, { steps: r.steps_used, tools: r.tool_calls || [] });
        var tit = getTitle(session_id);
        if (!tit || tit === "新对话"){
          var snip = msg.length > 28 ? msg.slice(0, 28) + "…" : msg;
          setTitle(session_id, snip);
        }
        refreshSessionList().catch(function(){});
      }catch(err){
        addAssistant("请求失败：" + err.message, null);
      }
    }

    document.getElementById("btnSend").addEventListener("click", sendMsg);
    document.getElementById("input").addEventListener("keydown", function(e){
      if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendMsg(); }
    });

    async function loadSkill(){
      const name = document.getElementById("skillName").value.trim();
      if(!name) return;
      const r = await api("/api/skill/" + encodeURIComponent(name), "GET");
      document.getElementById("skillText").value = r.content;
    }

    async function saveSkill(){
      const name = document.getElementById("skillName").value.trim();
      const content = document.getElementById("skillText").value;
      if(!name){ alert("请填写 skill 名称"); return; }
      await api("/api/skill", "POST", {name, content});
      await refreshSkillsList();
      addSystem("Skill 已保存：" + name);
    }

    async function ingestDoc(){
      const doc_id = document.getElementById("docId").value.trim();
      const source = document.getElementById("docSource").value.trim();
      const content = document.getElementById("docContent").value.trim();
      if(!doc_id || !source || !content){ alert("doc_id / source / content 不能为空"); return; }
      try{
        const r = await api("/api/ingest", "POST", {doc_id, source, content});
        addSystem("已入库 " + r.doc_id + "，chunks=" + r.deduplicated_chunks);
      }catch(err){ alert(err.message); }
    }

    async function ragDirect(){
      const question = document.getElementById("ragQuestion").value.trim();
      if(!question){ alert("问题不能为空"); return; }
      const tk = document.getElementById("ragTopK").value.trim();
      const body = { question };
      if(tk){ body.top_k = parseInt(tk, 10); if(isNaN(body.top_k)){ alert("top_k 须为整数"); return; } }
      try{
        const r = await api("/api/rag", "POST", body);
        let cites = "";
        if(r.citations && r.citations.length) cites = "\\n\\n[引用]\\n" + r.citations.join("\\n");
        addAssistant("[RAG 直连] " + r.answer + cites + "\\n\\n(trace=" + r.trace_id + ")", null);
      }catch(err){ addAssistant("RAG 失败：" + err.message, null); }
    }

    async function init(){
      const s = await api("/api/state", "GET");
      document.getElementById("provider").value = s.model_provider;
      document.getElementById("modelName").value = s.model_name;
      const defRag = s.framework && s.framework.rag_enabled !== false;
      const stored = sessionStorage.getItem("ui_use_rag");
      const on = stored === null ? defRag : stored === "1";
      setRagUi(on);
      const data = await api("/api/sessions", "GET");
      __sessionCache = data.sessions || [];
      syncOrderWithServer(__sessionCache);
      const last = localStorage.getItem("ui_last_session");
      const ids = __sessionCache.map(function(x){ return x.id; });
      if (last && ids.indexOf(last) >= 0){
        await selectSession(last);
      } else if (ids.length > 0){
        var ord = getOrder();
        var pick = ord.length ? ord[0] : ids[0];
        if (ids.indexOf(pick) < 0) pick = ids[0];
        await selectSession(pick);
      } else {
        await createNewChat();
      }
    }

    init().catch(function(e){ console.error(e); alert("初始化失败：" + e.message); });
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

    @app.get("/api/sessions")
    def list_sessions_ui():
        return jsonify({"sessions": service.list_chat_sessions()})

    @app.post("/api/session")
    def ensure_session_route():
        payload, err = _read_json_object()
        if err:
            return err
        sid = str(payload.get("session_id", "")).strip()
        if not sid:
            return _json_error("session_id 不能为空", "validation_error")
        service.ensure_chat_session(sid)
        return jsonify({"session_id": sid})

    @app.get("/api/session/<session_id>/history")
    def session_history(session_id: str):
        return jsonify(
            {"session_id": session_id, "messages": service.get_chat_history(session_id)}
        )

    @app.delete("/api/session/<session_id>")
    def delete_session_route(session_id: str):
        if not service.delete_chat_session(session_id):
            return _json_error("会话不存在", "not_found", 404)
        return jsonify({"ok": True})

    @app.post("/api/chat")
    def chat():
        payload, err = _read_json_object()
        if err:
            return err
        message = str(payload.get("message", "")).strip()
        session_id = str(payload.get("session_id", "default")).strip() or "default"
        if not message:
            return _json_error("message 不能为空", "validation_error")
        use_rag: bool | None = None
        if "use_rag" in payload:
            v = payload.get("use_rag")
            if isinstance(v, bool):
                use_rag = v
            elif isinstance(v, str):
                s = v.strip().lower()
                if s in {"1", "true", "yes", "y", "on"}:
                    use_rag = True
                elif s in {"0", "false", "no", "n", "off", ""}:
                    use_rag = False
                else:
                    return _json_error("use_rag 必须是布尔值", "validation_error")
            elif isinstance(v, int) and not isinstance(v, bool):
                use_rag = bool(v)
            elif v is None:
                use_rag = None
            else:
                return _json_error("use_rag 必须是布尔值", "validation_error")
        return jsonify(service.chat(session_id=session_id, user_message=message, use_rag=use_rag))

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
