import os
import json
import traceback
import datetime
import uuid


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _log_dir() -> str:
    base = os.getenv("ERROR_LOG_DIR")
    if not base:
        base = os.path.join(os.path.dirname(__file__), "tmp", "error_logs")
    os.makedirs(base, exist_ok=True)
    return base


def _safe_write_json(data: dict) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"error_{ts}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(_log_dir(), fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _serialize_langchain_messages(messages) -> list:
    serialized = []
    for m in messages or []:
        # m may be a HumanMessage/AIMessage/SystemMessage
        typ = getattr(m, "type", getattr(m, "_type", None))
        content = getattr(m, "content", None)
        if isinstance(content, list):
            parts = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                q = dict(p)
                if q.get("type") == "image_url":
                    url = (q.get("image_url") or {}).get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        q["image_url"] = {"url": "data:(omitted)"}
                if q.get("type") == "file" and "data" in q:
                    q["data"] = "(omitted)"
                parts.append(q)
            content_out = parts
        else:
            content_out = content
        serialized.append({"type": typ, "content": content_out})
    return serialized


def _serialize_discord_messages(messages) -> list:
    out = []
    for m in messages or []:
        try:
            content = getattr(m, "content", None)
            author_bot = bool(getattr(getattr(m, "author", None), "bot", False))
            atts = []
            for a in getattr(m, "attachments", []) or []:
                name = getattr(a, "filename", None)
                url = getattr(a, "url", None)
                atts.append({"filename": name, "url": url})
            out.append({"author_bot": author_bot, "content": content, "attachments": atts})
        except Exception:
            out.append({"error": "failed to serialize message"})
    return out


def _list_dir_files(dir_path: str) -> list:
    files = []
    if not dir_path or not os.path.isdir(dir_path):
        return files
    try:
        for name in os.listdir(dir_path):
            p = os.path.join(dir_path, name)
            if os.path.isfile(p):
                try:
                    size = os.path.getsize(p)
                except Exception:
                    size = None
                files.append({"name": name, "size": size})
    except Exception:
        pass
    return files


def log_agent_error(messages, input_dir_path: str, exc: Exception) -> None:
    if not _env_bool("ERROR_LOG_ENABLED", False):
        return
    data = {
        "scope": "agent.invoke",
        "inputs": {
            "messages": _serialize_langchain_messages(messages),
            "input_dir_files": _list_dir_files(input_dir_path),
        },
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }
    _safe_write_json(data)


def log_app_error(discord_messages, agent_messages, input_dir_path: str, exc: Exception) -> None:
    if not _env_bool("ERROR_LOG_ENABLED", False):
        return
    data = {
        "scope": "app.on_message",
        "inputs": {
            "discord_messages": _serialize_discord_messages(discord_messages),
            "agent_messages": _serialize_langchain_messages(agent_messages),
            "input_dir_files": _list_dir_files(input_dir_path),
        },
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }
    _safe_write_json(data)
