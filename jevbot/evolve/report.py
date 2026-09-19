"""Turn one or more results.json files into a markdown comparison.

Offline only: this reads what a finished `jevbot.evolve.run` wrote and reformats
it. No Jev calls, no dataset, no physics -- safe to run while an optimisation is
still in flight, and cheap to re-run while wording the README.

    python3 -m jevbot.evolve.report "v1=jevbot/evolve/results.json" [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# Importing the package reaches pybullet, which greets C-level stdout on import;
# the markdown may be going to that same stdout, so park fd 1 on stderr for it.
_stdout = os.dup(1)
os.dup2(2, 1)
try:
    from ..sim import ACTIONS
    from . import prompt
finally:
    os.dup2(_stdout, 1)
    os.close(_stdout)

TOP_PAIRS = 8
TOP_WRONG = 3


def load(spec: str):
    """'name=path' -> (name, data), or None (with a complaint) if unreadable."""
    name, sep, path = spec.partition("=")
    if not sep:
        name, path = os.path.splitext(os.path.basename(spec))[0], spec
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"skipping {spec}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"skipping {spec}: expected a JSON object at the top level", file=sys.stderr)
        return None
    return name, data


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _split(report: dict, arm: str) -> tuple[dict, dict]:
    entry = _dict(report.get(arm))
    return _dict(entry.get("val")), _dict(entry.get("test"))


def _action_of(component: str) -> str:
    return component[len(prompt.PREFIX):] if component.startswith(prompt.PREFIX) else component


def arm_order(runs) -> list[str]:
    """Arms in first-seen order, so runs naming different arms still line up."""
    order: dict[str, None] = {}
    for _name, data in runs:
        for arm in _dict(data.get("report")):
            order.setdefault(arm, None)
    return list(order)


def table(runs, arms) -> list[str]:
    out = ["## Results", "",
           "| run | arm | val acc | test acc | test margin | val - test |",
           "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, data in runs:
        report = _dict(data.get("report"))
        for arm in (a for a in arms if a in report):
            val, test = _split(report, arm)
            va, ta, margin = _num(val.get("accuracy")), _num(test.get("accuracy")), _num(test.get("mean_margin"))
            # Positive = worse on held-out data than on the set GEPA selected on.
            drop = f"{(va - ta) * 100:+.1f} pts" if va is not None and ta is not None else "n/a"
            out.append(f"| {name} | {arm} "
                       f"| {f'{va:.1%}' if va is not None else 'n/a'} "
                       f"| {f'{ta:.1%}' if ta is not None else 'n/a'} "
                       f"| {f'{margin:+.3f}' if margin is not None else 'n/a'} "
                       f"| **{drop}** |")
    return out + ["", "_val - test is the overfitting indicator: positive means the arm "
                      "lost accuracy off the set it was selected on._", ""]


def cost(runs) -> list[str]:
    out = ["## Cost", ""]
    for name, data in runs:
        calls, spend = _num(data.get("jev_calls")), _num(data.get("spend_usd"))
        muts = data.get("mutations")
        out.append(f"- **{name}**: {calls if calls is not None else 'n/a'} Jev calls, "
                   f"{f'${spend:.5f}' if spend is not None else 'spend n/a'}, "
                   f"{len(muts) if isinstance(muts, list) else 0} mutations applied")
    return out + [""]


def learned(runs) -> list[str]:
    """Only the clauses the search bolted onto the seed text are interesting."""
    out = ["## What the search learned", ""]
    for name, data in runs:
        evolved = _dict(data.get("evolved"))
        lines = []
        for action in ACTIONS:
            text = evolved.get(prompt.PREFIX + action)
            _base, clauses = prompt.split_clauses(text) if isinstance(text, str) else ("", [])
            if clauses:
                lines.append(f"- **{action}** — " + " ".join(clauses))
        out += [f"### {name}", ""] + (lines or ["_nothing added to the seed criteria._"]) + [""]
    return out


def confusions(runs) -> list[str]:
    out = ["## Confusion pairs attacked", ""]
    for name, data in runs:
        muts = data.get("mutations")
        pairs = Counter(
            (_action_of(str(m.get("component", "?"))), str(m.get("confused_with", "?")))
            for m in (muts if isinstance(muts, list) else []) if isinstance(m, dict))
        lines = [f"- {action} vs {other}: {count}"
                 for (action, other), count in pairs.most_common(TOP_PAIRS)]
        out += [f"### {name}", ""] + (lines or ["_no mutations recorded._"]) + [""]
    return out


def remaining(runs, arms) -> list[str]:
    out = ["## Remaining test errors", ""]
    for name, data in runs:
        report = _dict(data.get("report"))
        lines = []
        for arm in (a for a in arms if a in report):
            _val, test = _split(report, arm)
            wrong = test.get("wrong")
            wrong = [w for w in wrong if isinstance(w, dict)] if isinstance(wrong, list) else []
            chosen = Counter(str(w.get("chose", "?")) for w in wrong).most_common(TOP_WRONG)
            breakdown = ", ".join(f"{action} x{count}" for action, count in chosen)
            lines.append(f"- **{arm}**: {len(wrong)} wrong"
                         + (f" — chose {breakdown}" if breakdown else ""))
        out += [f"### {name}", ""] + (lines or ["_no arms in this file._"]) + [""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", metavar="NAME=PATH")
    parser.add_argument("--out", help="write the markdown here instead of stdout")
    args = parser.parse_args()

    runs = [loaded for loaded in (load(spec) for spec in args.runs) if loaded]
    if not runs:
        print("no readable results files", file=sys.stderr)
        raise SystemExit(1)

    arms = arm_order(runs)
    lines = (["# Evolve results", ""] + table(runs, arms) + cost(runs)
             + learned(runs) + confusions(runs) + remaining(runs, arms))
    text = "\n".join(lines).rstrip() + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"markdown -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
