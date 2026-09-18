#!/usr/bin/env python3
"""Ask typesafe/jev-1.13 structured questions through OpenRouter.

Jev is a System One decision model: it answers typed questions about a piece
of state instead of writing prose, so it speaks OpenRouter's /api/alpha/decisions
endpoint rather than /chat/completions.

No third-party dependencies -- stdlib only.

    export OPENROUTER=sk-or-...
    ./jev.py "Payouts failing for 3 days!" --noul 'urgent: Is this urgent?'
    ./jev.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_ROOT = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
DEFAULT_MODEL = "typesafe/jev-1.13"
BAR_WIDTH = 24


# --------------------------------------------------------------------------- io

def api_key() -> str:
    for var in ("OPENROUTER", "OPENROUTER_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key
    sys.exit("No API key found. Set OPENROUTER (or OPENROUTER_API_KEY).")


def decide(payload: dict) -> dict:
    req = urllib.request.Request(
        API_ROOT + "/alpha/decisions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            # Optional OpenRouter attribution headers.
            "HTTP-Referer": "https://github.com/angadjosan/Jevplayground",
            "X-Title": "Jev playground",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code}: {http_error(exc.read().decode(errors='replace'))}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach {API_ROOT}: {exc.reason}")


def http_error(body: str) -> str:
    """Unwrap OpenRouter's {"error": {"message": ...}}.

    Validation failures arrive with a JSON array *as a string* inside message,
    so try to unwrap that too rather than printing it escaped.
    """
    try:
        message = json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return body
    try:
        return json.dumps(json.loads(message), indent=2)
    except ValueError:
        return message


# ---------------------------------------------------------------- question specs

def split_once(text: str, sep: str, what: str) -> tuple[str, str]:
    head, found, tail = text.partition(sep)
    if not found:
        sys.exit(f"Malformed {what}: expected {sep!r} in {text!r}")
    return head.strip(), tail.strip()


def parse_noul(spec: str) -> tuple[str, dict]:
    """'name: instructions'"""
    name, instructions = split_once(spec, ":", "--noul")
    return name, {"type": "noul", "instructions": instructions}


def parse_choice(spec: str) -> tuple[str, dict]:
    """'name: instructions = option | option: rubric | ...'"""
    head, options = split_once(spec, "=", "--choice")
    name, instructions = split_once(head, ":", "--choice")
    criteria: dict[str, str | None] = {}
    for option in options.split("|"):
        label, _, rubric = option.partition(":")
        label = label.strip()
        if not label:
            sys.exit(f"Malformed --choice: empty option in {spec!r}")
        criteria[label] = rubric.strip() or None
    if len(criteria) < 2:
        sys.exit(f"Malformed --choice: needs at least two options in {spec!r}")
    return name, {"type": "choice", "instructions": instructions, "criteria": criteria}


def parse_score(spec: str) -> tuple[str, dict]:
    """'name: instructions = lowest level | ... | highest level'"""
    head, levels = split_once(spec, "=", "--score")
    name, instructions = split_once(head, ":", "--score")
    criteria = [level.strip() for level in levels.split("|") if level.strip()]
    if len(criteria) < 2:
        sys.exit(f"Malformed --score: needs at least two levels in {spec!r}")
    return name, {"type": "score", "instructions": instructions, "criteria": criteria}


def build_questions(args: argparse.Namespace) -> dict:
    questions: dict[str, dict] = {}
    for spec in args.noul:
        name, question = parse_noul(spec)
        questions[name] = question
    for spec in args.choice:
        name, question = parse_choice(spec)
        questions[name] = question
    for spec in args.score:
        name, question = parse_score(spec)
        questions[name] = question
    if args.questions:
        extra = read_json(args.questions, "--questions")
        if not isinstance(extra, dict):
            sys.exit("--questions must contain a JSON object of question id -> question.")
        questions.update(extra)
    if not questions:
        sys.exit("No questions asked. Use --noul/--choice/--score or --questions.")
    return questions


def read_json(path: str, flag: str):
    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    try:
        return json.loads(text)
    except ValueError as exc:
        sys.exit(f"{flag}: not valid JSON ({exc}).")


def build_state(args: argparse.Namespace):
    """State is a string, or JSON (object/array) when --json-state is set."""
    if args.state_file:
        raw = sys.stdin.read() if args.state_file == "-" else \
            open(args.state_file, encoding="utf-8").read()
    elif args.state is not None:
        raw = args.state
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        sys.exit("Empty state. Pass it as an argument, with --state-file, or on stdin.")
    if args.json_state:
        try:
            return json.loads(raw)
        except ValueError as exc:
            sys.exit(f"--json-state: state is not valid JSON ({exc}).")
    return raw


# ------------------------------------------------------------------------ output

def bar(value: float) -> str:
    filled = round(max(0.0, min(1.0, value)) * BAR_WIDTH)
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def show_distribution(probabilities: dict, labels: dict | None = None) -> None:
    ordered = sorted(probabilities.items(), key=lambda kv: -kv[1])
    width = max(len(str(k)) for k, _ in ordered)
    for key, probability in ordered:
        label = f"{key:<{width}}"
        if labels:
            label += f"  {labels.get(str(key), '')}"
        print(f"    {bar(probability)}  {probability:>5.0%}  {label}".rstrip())


def show(name: str, answer: dict) -> None:
    kind = answer.get("type")
    confidence = answer.get("confidence")
    suffix = f"   [confidence {confidence:.2f}]" if confidence is not None else ""

    if kind == "noul":
        value = answer["noul"]
        print(f"{name}  ({kind}){suffix}")
        print(f"    {bar(value)}  {value:>5.0%}  yes")
    elif kind == "choice":
        print(f"{name}  ({kind}) -> {answer['choice']}{suffix}")
        show_distribution(answer.get("probabilities", {}))
    elif kind == "score":
        legend = answer.get("legend", {})
        top = max(legend, key=int) if legend else "?"
        nearest = legend.get(str(round(answer["score"])), "")
        print(f"{name}  ({kind}) -> {answer['score']:.2f} / {top}"
              f"  {nearest}{suffix}")
        show_distribution(answer.get("probabilities", {}), legend)
    else:  # a question type this CLI does not know about yet
        print(f"{name}  ({kind})")
        print("    " + json.dumps(answer, indent=2).replace("\n", "\n    "))
    print()


def report(body: dict, verbose: bool) -> None:
    if not verbose:
        return
    usage = body.get("usage", {})
    parts = [f"model: {body.get('model')}"]
    if body.get("provider"):
        parts.append(f"provider: {body['provider']}")
    if usage:
        parts.append(f"tokens: {usage.get('input_tokens')} in / "
                     f"{usage.get('output_tokens')} out")
    if usage.get("cost") is not None:
        parts.append(f"cost: ${usage['cost']:.6f}")
    print("[" + ", ".join(parts) + "]", file=sys.stderr)


# -------------------------------------------------------------------------- main

def check(model: str) -> None:
    """Send the cheapest possible decision to prove the model answers.

    Decision models are not listed in OpenRouter's /models catalogue, so a live
    round trip is the only honest availability check.
    """
    body = decide({
        "model": model,
        "state": "ping",
        "questions": {"ok": {"type": "noul", "instructions": "Is this a test?"}},
    })
    usage = body.get("usage", {})
    print(f"{model} is answering.")
    print(f"  resolved model: {body.get('model')}")
    print(f"  provider:       {body.get('provider')}")
    print(f"  probe answer:   noul {body['answers']['ok']['noul']}")
    print(f"  probe cost:     ${usage.get('cost', 0):.6f} "
          f"({usage.get('input_tokens')} in / {usage.get('output_tokens')} out)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""question syntax:
  --noul   'name: instructions'
  --choice 'name: instructions = option | option: rubric | ...'
  --score  'name: instructions = lowest level | ... | highest level'

examples:
  ./jev.py "Payouts have failed for 3 days" \\
      --noul 'urgent: Does this convey urgency?' \\
      --choice 'team: Who handles this? = billing | technical | sales' \\
      --score 'mood: How frustrated? = Calm | Frustrated | Very angry'

  ./jev.py --state-file ticket.json --json-state --questions questions.json
""")
    parser.add_argument("state", nargs="?",
                        help="the text to evaluate (default: stdin)")
    parser.add_argument("--state-file", metavar="FILE",
                        help="read state from FILE ('-' for stdin)")
    parser.add_argument("--json-state", action="store_true",
                        help="parse the state as JSON and send it structured")
    parser.add_argument("--noul", action="append", default=[], metavar="SPEC",
                        help="yes/no question (repeatable)")
    parser.add_argument("--choice", action="append", default=[], metavar="SPEC",
                        help="pick-one question (repeatable)")
    parser.add_argument("--score", action="append", default=[], metavar="SPEC",
                        help="rate-against-levels question (repeatable)")
    parser.add_argument("--questions", metavar="FILE",
                        help="JSON file of raw question objects ('-' for stdin)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true",
                        help="print the raw API response")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print model, token usage and cost to stderr")
    parser.add_argument("--check", action="store_true",
                        help="probe the model with one cheap question and exit")
    args = parser.parse_args()

    if args.check:
        check(args.model)
        return

    questions = build_questions(args)
    body = decide({
        "model": args.model,
        "state": build_state(args),
        "questions": questions,
    })

    if args.json:
        print(json.dumps(body, indent=2))
        return

    answers = body.get("answers", {})
    # Report in the order asked, not the order returned.
    for name in list(questions) + [k for k in answers if k not in questions]:
        if name in answers:
            show(name, answers[name])
    report(body, args.verbose)


if __name__ == "__main__":
    main()
