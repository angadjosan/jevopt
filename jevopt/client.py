"""Minimal client for OpenRouter's decisions endpoint (the Jev API)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API_ROOT = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
MODEL = "typesafe/jev-1.13"


class JevError(RuntimeError):
    pass


def api_key() -> str:
    for var in ("OPENROUTER", "OPENROUTER_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key
    raise JevError("No API key found. Set OPENROUTER (or OPENROUTER_API_KEY).")


def ask(state, questions: dict, model: str = MODEL, retries: int = 4) -> dict:
    """POST one state plus a map of questions; return the parsed response.

    Retries 429/529 with exponential backoff, as the API docs ask.
    """
    payload = {"model": model, "state": state, "questions": questions}
    request = urllib.request.Request(
        API_ROOT + "/alpha/decisions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/angadjosan/Jevplayground",
            "X-Title": "Jev playground",
        },
    )
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            if exc.code in (429, 529) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise JevError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise JevError(f"Could not reach {API_ROOT}: {exc.reason}") from exc
    raise JevError("unreachable")
