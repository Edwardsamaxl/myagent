from __future__ import annotations

import os
from typing import Any

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

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


def create_app(config: AgentConfig | None = None) -> Flask:
    cfg = config or AgentConfig.from_env()
    service = AgentService(cfg)
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
    _register_optional_cors(app)

    @app.get("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, filename)

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
        try:
            return jsonify(
                service.chat(session_id=session_id, user_message=message, use_rag=use_rag)
            )
        except ValueError as exc:
            return _json_error(str(exc), "chat_failed", 400)
        except requests.RequestException as exc:
            return _json_error(str(exc), "chat_upstream", 502)

    @app.post("/api/chat/stream")
    def chat_stream():
        """SSE 流式聊天接口。"""
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

        def generate():
            try:
                result = service.chat(session_id=session_id, user_message=message, use_rag=use_rag)
                answer = result.get("answer", "")
                if not answer:
                    yield "data: {\"text\":\"\"}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                # Stream word by word for visual feedback
                words = answer.split("")
                for i, ch in enumerate(words):
                    chunk = answer[:i+1]
                    import json as _json
                    yield f"data: {_json.dumps({'text': ch, 'done': False})}\n\n"
                yield "data: [DONE]\n\n"
            except ValueError as exc:
                import json as _json
                yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
            except requests.RequestException as exc:
                import json as _json
                yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
            except Exception as exc:
                import json as _json
                yield f"data: {_json.dumps({'error': '服务内部错误'})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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
                return _json_error("metadata 须为 JSON 对象", "invalid_json")
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
