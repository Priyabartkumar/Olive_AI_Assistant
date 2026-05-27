import json
import os
import time
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "requests.jsonl")


def _ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_request(
    model_name: str,
    user_message: str,
    response: str,
    latency_ms: float,
    tool_used: str | None = None,
    guardrail_triggered: bool = False,
    error: str | None = None,
):
    _ensure_log_dir()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "user_message": user_message[:500],
        "response_preview": response[:500],
        "response_length": len(response),
        "latency_ms": round(latency_ms, 2),
        "tool_used": tool_used,
        "guardrail_triggered": guardrail_triggered,
        "error": error,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_logs() -> list[dict]:
    if not os.path.exists(LOG_FILE):
        return []
    logs = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs


def get_summary(logs: list[dict]) -> dict:
    if not logs:
        return {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "models_used": {},
            "tools_used": {},
            "guardrails_triggered": 0,
            "errors": 0,
        }

    models = {}
    tools = {}
    total_latency = 0
    guardrails = 0
    errors = 0

    for log in logs:
        model = log.get("model", "unknown")
        models[model] = models.get(model, 0) + 1

        tool = log.get("tool_used")
        if tool:
            tools[tool] = tools.get(tool, 0) + 1

        total_latency += log.get("latency_ms", 0)

        if log.get("guardrail_triggered"):
            guardrails += 1

        if log.get("error"):
            errors += 1

    return {
        "total_requests": len(logs),
        "avg_latency_ms": round(total_latency / len(logs), 2),
        "models_used": models,
        "tools_used": tools,
        "guardrails_triggered": guardrails,
        "errors": errors,
    }
