"""Turn jevopt run results -- and, optionally, paired comparisons -- into markdown.

Offline only: it reads what a finished `jevopt.optimize` (and `jevopt.compare`)
wrote and reformats it -- no Jev calls, no evaluation.

The clause breakdown is the one section that cannot be read off the JSON alone:
telling which sentences the search *added* needs the task's seed option text and
its clause grammar, so the task is named on the command line and imported exactly
as the optimiser takes it. Nothing here knows about any one domain.

    python3 -m jevopt.report --task jevopt.tasks.triage \\
        "triage=runs/triage.results.json" --compare "triage=runs/triage.compare.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from . import grammar
from .optimize import load_task
from .task import Task

TOP_PAIRS = 8
TOP_WRONG = 3
ALPHA = 0.05          # the threshold compare.py wrote its "significant" flag with
NOISE_HINT = "noise"  # substring marking a noise-floor arm, e.g. "noise_twin"


def load(spec: str):
    """'name=path' -> (name, data), or None (with a complaint) if unreadable."""
    name, sep, path = spec.partition("=")
    if not sep:
        name, path = os.path.splitext(os.path.basename(spec))[0], spec
    try:
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object at the top level")
    except (OSError, ValueError) as exc:
        print(f"skipping {spec}: {exc}", file=sys.stderr)
        return None
    return name, data


# Every value read below comes out of a JSON file an older run may have written
# differently, so nothing is assumed present, numeric, or the right shape.
def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}

def _list(value) -> list:
    return value if isinstance(value, list) else []

def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value

def _fmt(value, kind: str = "pct") -> str:
    """A cell: percentage, signed percentage points, or a signed margin."""
    if _num(value) is None:
        return "n/a"
    return {"pct": f"{value:.1%}", "pt": f"{100 * value:+.1f} pt",
            "f3": f"{value:+.3f}"}[kind]

def _side(report: dict, arm: str, side: str) -> dict:
    return _dict(_dict(report.get(arm)).get(side))

def _option_of(component: str) -> str:
    if component.startswith(grammar.PREFIX):
        return component[len(grammar.PREFIX):]
    return component


def mismatch(name: str, data: dict, task: Task) -> str | None:
    """Options this task does not know make every task-aware section come out
    empty -- that is the wrong --task, not a finding, so say it out loud."""
    evolved = _dict(data.get("evolved"))
    unknown = sorted({_option_of(key) for key in evolved} - set(task.options))
    if not evolved:
        return f"{name}: no 'evolved' candidate in this file, so no clauses to report"
    return (f"{name}: recorded task {str(data.get('task'))!r}, whose options "
            f"{unknown} are "
            f"unknown to task {task.name!r} -- pass the --task this run used"
            ) if unknown else None


def table(runs, arms) -> list[str]:
    out = ["## Results", "",
           "| run | arm | val acc | test acc | test margin | val - test |",
           "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, data in runs:
        report = _dict(data.get("report"))
        for arm in (a for a in arms if a in report):
            val, test = _side(report, arm, "val"), _side(report, arm, "test")
            va, ta = _num(val.get("accuracy")), _num(test.get("accuracy"))
            # Positive = worse on held-out data than on the set GEPA selected on.
            drop = ("n/a" if va is None or ta is None
                    else f"{(va - ta) * 100:+.1f} pts")
            out.append(f"| {name} | {arm} | {_fmt(va)} | {_fmt(ta)} "
                       f"| {_fmt(_num(test.get('mean_margin')), 'f3')} | **{drop}** |")
    return out + ["", "_val - test is the overfitting indicator: positive means the arm "
                      "lost accuracy off the set it was selected on._", ""]


def cost(runs) -> list[str]:
    out = ["## Cost", ""]
    for name, data in runs:
        calls, spend = _num(data.get("jev_calls")), _num(data.get("spend_usd"))
        out.append(f"- **{name}**: {calls if calls is not None else 'n/a'} Jev calls, "
                   f"{f'${spend:.5f}' if spend is not None else 'spend n/a'}, "
                   f"{len(_list(data.get('mutations')))} mutations applied")
    return out + [""]


def learned(runs, task: Task) -> list[str]:
    """Only the clauses the search bolted onto the seed text are interesting.

    options_of() walks the *task's* options, falling back to their seed text, and
    split_clauses() recognises a clause by asking the task's grammar what it can
    produce -- so this is domain-agnostic; no option name is hardcoded."""
    out = ["## What the search learned", ""]
    for name, data in runs:
        split = ((option, grammar.split_clauses(text, task)[1]) for option, text
                 in grammar.options_of(task, _dict(data.get("evolved"))).items())
        lines = [f"- **{o}** — " + " ".join(clauses) for o, clauses in split if clauses]
        out += ([f"### {name}", ""]
                + (lines or ["_nothing added to the seed criteria._"]) + [""])
    return out


def confusions(runs) -> list[str]:
    out = ["## Confusion pairs attacked", ""]
    for name, data in runs:
        counts = Counter(
            (_option_of(str(m.get("component", "?"))), str(m.get("confused_with", "?")))
            for m in _list(data.get("mutations")) if isinstance(m, dict))
        out += [f"### {name}", ""] + ([f"- {a} vs {b}: {n}" for (a, b), n
                in counts.most_common(TOP_PAIRS)] or ["_no mutations recorded._"]) + [""]
    return out


def remaining(runs, arms) -> list[str]:
    out = ["## Remaining test errors", ""]
    for name, data in runs:
        report, lines = _dict(data.get("report")), []
        for arm in (a for a in arms if a in report):
            wrong = [w for w in _list(_side(report, arm, "test").get("wrong"))
                     if isinstance(w, dict)]
            chosen = Counter(
                str(w.get("chose", "?")) for w in wrong).most_common(TOP_WRONG)
            breakdown = ", ".join(f"{option} x{n}" for option, n in chosen)
            lines.append(f"- **{arm}**: {len(wrong)} wrong"
                         + (f" — chose {breakdown}" if breakdown else ""))
        out += [f"### {name}", ""] + (lines or ["_no arms in this file._"]) + [""]
    return out


def comparison(name: str, data: dict) -> list[str]:
    """compare.py's evidence: the marginal accuracies, then the pairwise tests
    that are the actual comparison, then what the noise floor makes of them."""
    arms = _dict(data.get("arms"))
    pairs = [p for p in _list(data.get("pairs")) if isinstance(p, dict)]
    out = [f"## Comparison: {name}", "",
           f"_{data.get('n_instances', '?')} shared test instances, split seed "
           f"{data.get('split_seed', '?')}, "
           f"{_dict(data.get('bootstrap')).get('resamples', 0)}"
           f" paired bootstrap resamples; {len(arms)} arms, {len(pairs)} pairs._", "",
           "| arm | test acc | 95% Wilson (unpaired) | mean margin |",
           "| --- | ---: | :---: | ---: |"]
    for arm, row in ((a, _dict(r)) for a, r in arms.items()):
        lo, hi = (_list(row.get("wilson95")) + [None, None])[:2]
        out.append(f"| {arm} | {_fmt(_num(row.get('accuracy')))} | {_fmt(_num(lo))} – "
                   f"{_fmt(_num(hi))} | {_fmt(_num(row.get('mean_margin')), 'f3')} |")
    out += ["", "_Those intervals are the unpaired view and overlap heavily; they "
                "are NOT "
                "the test. The pairwise rows are, because every arm answered these same "
                "instances._", "",
            "### Pairwise (McNemar exact, two-sided)", "",
            "| pair | acc d | b | c | McNemar p | paired 95% CI on acc d | verdict |",
            "| --- | ---: | ---: | ---: | ---: | :---: | --- |"]
    undecided = []
    for e in pairs:
        p, ci = _num(e.get("mcnemar_p")), _list(e.get("accuracy_diff_ci95"))
        sig = bool(e.get("significant", p is not None and p < ALPHA))
        undecided += [] if sig else [e]
        span = ", ".join(_fmt(x, "pt") for x in ci[:2]) if len(ci) == 2 else ""
        out.append(f"| {e.get('a')} vs {e.get('b')} "
                   f"| {_fmt(_num(e.get('accuracy_diff')), 'pt')} "
                   f"| {e.get('b_count', '?')} | {e.get('c_count', '?')} "
                   f"| {f'{p:.4f}' if p is not None else 'n/a'} "
                   f"| {f'[{span}]' if span else 'n/a'} "
                   f"| {'separates' if sig else '**NOT distinguishable**'} |")
    return out + ["", f"**{len(pairs) - len(undecided)} of {len(pairs)} pairs "
                      f"separate at "
                      f"p < {ALPHA}**; this data cannot order the other "
                      f"{len(undecided)}, where "
                      f"the sign of the difference is not evidence. Expect ~"
                      f"{ALPHA * len(pairs):.1f} false positives among "
                      f"{len(pairs)} pairs tested "
                      f"at once.", ""] + noise_floor(arms, pairs, undecided)


def noise_floor(arms: dict, pairs: list, undecided: list) -> list[str]:
    """A noise twin is a control that should carry no signal at all, so an arm
    that cannot be told apart from it has not been shown to work."""
    floor = next((a for a in arms if NOISE_HINT in a.lower()), None)
    if floor is None:
        return []
    tied = sorted({e["b"] if e.get("a") == floor else e["a"]
                   for e in undecided if floor in (e.get("a"), e.get("b"))})
    tested = sum(1 for e in pairs if floor in (e.get("a"), e.get("b")))
    floor_acc = _fmt(_num(_dict(arms.get(floor)).get("accuracy")))
    return [f"**Noise floor: `{floor}` at {floor_acc}.**"
            f" That arm is the control: a pair that does not separate from it is "
            f"uninterpretable, whichever way it points.", "",
            f"- not distinguishable from the floor ({len(tied)} of the {tested} "
            f"arms tested "
            f"against it): " + ", ".join(f"`{a}`" for a in tied) if tied else
            f"- all {tested} other arms separate from the floor.", ""]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add = parser.add_argument
    add("runs", nargs="*", metavar="NAME=PATH", help="results JSON from jevopt.optimize")
    add("--task", default="jevopt.tasks.robot", help="dotted path to a module exposing "
        "build() -> Task; needed to tell added clauses from the seed option text")
    add("--compare", action="append", default=[], metavar="NAME=PATH",
        help="comparison JSON from jevopt.compare; repeatable")
    add("--out", help="write the markdown here instead of stdout")
    args = parser.parse_args(argv)

    try:
        task = load_task(args.task)
    except Exception as exc:              # a bad --task is a typo, not a crash
        print(f"cannot load task {args.task!r}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    runs = [loaded for loaded in (load(spec) for spec in args.runs) if loaded]
    compares = [loaded for loaded in (load(spec) for spec in args.compare) if loaded]
    if not runs and not compares:
        print("no readable results or comparison files", file=sys.stderr)
        raise SystemExit(1)
    warnings = [w for w in (mismatch(n, d, task) for n, d in runs) if w]
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    lines = [f"# jevopt report — task `{task.name}`", ""]
    lines += [f"> **warning:** {w}" for w in warnings] + ([""] if warnings else [])
    if runs:
        # Arms in first-seen order, so runs naming different arms still line up.
        arms = list({arm: None for _n, d in runs for arm in _dict(d.get("report"))})
        lines += (table(runs, arms) + cost(runs) + learned(runs, task)
                  + confusions(runs) + remaining(runs, arms))
    for name, data in compares:
        lines += comparison(name, data)

    text = "\n".join(lines).rstrip() + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"markdown -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
