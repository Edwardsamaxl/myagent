from __future__ import annotations

from flask import Flask, jsonify, request

from ..application.agent_service import AgentService
from ..config import AgentConfig


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
      if(!res.ok){ throw new Error(await res.text()); }
      return res.json();
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
        add("assistant", r.answer + tools);
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

    @app.get("/")
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/state")
    def state():
        return jsonify(service.get_state())

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(force=True)
        message = str(payload.get("message", "")).strip()
        session_id = str(payload.get("session_id", "default")).strip() or "default"
        if not message:
            return jsonify({"error": "message 不能为空"}), 400
        return jsonify(service.chat(session_id=session_id, user_message=message))

    @app.post("/api/model")
    def update_model():
        payload = request.get_json(force=True)
        provider = str(payload.get("provider", "")).strip()
        model_name = str(payload.get("model_name", "")).strip()
        try:
            return jsonify(service.update_model(provider=provider, model_name=model_name))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/memory")
    def get_memory():
        return jsonify({"content": service.get_memory()})

    @app.post("/api/memory")
    def save_memory():
        payload = request.get_json(force=True)
        content = str(payload.get("content", ""))
        service.save_memory(content)
        return jsonify({"message": "ok"})

    @app.get("/api/skill/<name>")
    def get_skill(name: str):
        return jsonify({"name": name, "content": service.get_skill(name)})

    @app.post("/api/skill")
    def save_skill():
        payload = request.get_json(force=True)
        name = str(payload.get("name", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not name:
            return jsonify({"error": "name 不能为空"}), 400
        msg = service.save_skill(name, content)
        return jsonify({"message": msg})

    return app
