#!/usr/bin/env python3
"""Minimal OpenRouter client for typesafe/jev-1.13.

No third-party dependencies -- stdlib only.

    export OPENROUTER=sk-or-...
    ./jev.py "Write a haiku about type inference"
    ./jev.py --stream --system "You are terse." "Explain HM type inference"
    ./jev.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_ROOT = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = "typesafe/jev-1.13"


def api_key() -> str:
    for var in ("OPENROUTER", "OPENROUTER_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key
    sys.exit("No API key found. Set OPENROUTER (or OPENROUTER_API_KEY).")


def request(path: str, payload: dict | None = None, stream: bool = False):
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
        # Optional OpenRouter attribution headers.
        "HTTP-Referer": "https://github.com/angadjosan/Jevplayground",
        "X-Title": "Jev playground",
    }
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API_ROOT + path, data=data, headers=headers)
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            body = json.dumps(json.loads(body), indent=2)
        except ValueError:
            pass
        sys.exit(f"HTTP {exc.code} from {path}:\n{body}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach {API_ROOT}: {exc.reason}")
    return resp if stream else json.load(resp)


def check(model: str) -> None:
    """Confirm the model is live and print what OpenRouter advertises for it."""
    models = request("/models")["data"]
    match = next((m for m in models if m["id"] == model), None)
    if match is None:
        near = [m["id"] for m in models if model.split("/")[0] in m["id"]]
        print(f"{model} not found in /models.")
        if near:
            print("Same provider:", ", ".join(sorted(near)))
        sys.exit(1)
    fields = ("id", "name", "context_length", "pricing", "architecture",
              "supported_parameters")
    print(json.dumps({k: match.get(k) for k in fields}, indent=2))


def messages(args: argparse.Namespace) -> list[dict]:
    msgs = []
    if args.system:
        msgs.append({"role": "system", "content": args.system})
    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        sys.exit("Empty prompt.")
    msgs.append({"role": "user", "content": prompt})
    return msgs


def complete(args: argparse.Namespace) -> None:
    payload = {
        "model": args.model,
        "messages": messages(args),
        "stream": args.stream,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens

    if not args.stream:
        body = request("/chat/completions", payload)
        choice = body["choices"][0]
        print(choice["message"]["content"])
        report(body.get("usage"), choice.get("finish_reason"), args.verbose)
        return

    usage = finish = None
    with request("/chat/completions", payload, stream=True) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            # OpenRouter sends ": OPENROUTER PROCESSING" comments as keepalives.
            if not line or line.startswith(":") or not line.startswith("data: "):
                continue
            chunk = line[len("data: "):]
            if chunk == "[DONE]":
                break
            event = json.loads(chunk)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                text = choice.get("delta", {}).get("content")
                if text:
                    print(text, end="", flush=True)
    print()
    report(usage, finish, args.verbose)


def report(usage: dict | None, finish: str | None, verbose: bool) -> None:
    if not verbose:
        return
    parts = []
    if usage:
        parts.append(
            f"tokens: {usage.get('prompt_tokens')} in / "
            f"{usage.get('completion_tokens')} out"
        )
    if finish:
        parts.append(f"finish: {finish}")
    if parts:
        print("\n[" + ", ".join(parts) + "]", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", nargs="?", help="prompt text (default: stdin)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", help="system prompt")
    parser.add_argument("--stream", action="store_true", help="stream the reply")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print token usage and finish reason to stderr")
    parser.add_argument("--check", action="store_true",
                        help="look the model up in /models and exit")
    args = parser.parse_args()

    if args.check:
        check(args.model)
    else:
        complete(args)


if __name__ == "__main__":
    main()
