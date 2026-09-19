"""Turn jevopt run results -- and, optionally, paired comparisons -- into markdown.

Offline only: it reads what a finished `jevopt.optimize` (and `jevopt.compare`)
wrote and reformats it -- no Jev calls, no evaluation.

The clause breakdown is the one section that cannot be read off the JSON alone:
telling which sentences the search *added* needs the task's seed option text and
its clause grammar, so the task is named on the command line and imported the way
the optimiser takes it. Nothing here knows about any one domain.

    python3 -m jevopt.report --task jevopt.tasks.triage \\
        "triage=runs/triage.results.json" --compare "t=runs/triage.compare.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from . import grammar
from .optimize import add_task_argument, check_writable, load_task
from .task import Task

TOP_PAIRS = 8
TOP_WRONG = 3
SEED_ARM = "seed"            # the arm optimize.py scores twice
REPEAT_ARM = "seed (repeat)"  # ... and the second score: the measured noise floor
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
        return f"{name}: no 'evolved' candidate here, so no clauses to report"
    if unknown:
        return (f"{name}: recorded task {str(data.get('task'))!r}, whose options "
                f"{unknown} are unknown to task {task.name!r} -- pass that --task")
    return None


def _floor_note(name: str, report: dict) -> list[str]:
    """What the same prompt, scored twice, moved by -- in the table, not a footnote.

    Jev samples, so a run has reported a 4.8-point "regression" between a prompt
    and itself. Any gap in the rows above smaller than this one is noise.
    """
    seed, repeat = _side(report, SEED_ARM, "test"), _side(report, REPEAT_ARM, "test")
    first, second = _num(seed.get("accuracy")), _num(repeat.get("accuracy"))
    if first is None or second is None:
        return [f"> **{name}: no noise floor measured** — this run did not score "
                f"the seed twice, so no gap in it can be called a difference.", ""]
    floor = abs(first - second)
    inside = sorted(arm for arm, row in report.items()
                    if arm not in (SEED_ARM, REPEAT_ARM)
                    and _num(_dict(_dict(row).get("test")).get("accuracy")) is not None
                    and abs(_num(_dict(row)["test"]["accuracy"]) - first) <= floor)
    note = (f"> **{name}: noise floor {_fmt(floor, 'pt')} on test** — `{SEED_ARM}` "
            f"and `{REPEAT_ARM}` are the same prompt scored twice.")
    if inside:
        note += (" Not shown to differ from the seed: "
                 + ", ".join(f"`{a}`" for a in inside) + ".")
    return [note, ""]


def table(runs, arms) -> list[str]:
    out = ["## Results", "",
           "| run | arm | n val | val acc | n test | test acc | test margin "
           "| val - test |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in runs:
        report = _dict(data.get("report"))
        for arm in (a for a in arms if a in report):
            val, test = _side(report, arm, "val"), _side(report, arm, "test")
            va, ta = _num(val.get("accuracy")), _num(test.get("accuracy"))
            # Positive = worse on held-out data than on the set GEPA selected on.
            gap = None if va is None or ta is None else va - ta
            out.append(f"| {name} | {arm} | {val.get('n', 'n/a')} | {_fmt(va)} "
                       f"| {test.get('n', 'n/a')} | {_fmt(ta)} "
                       f"| {_fmt(_num(test.get('mean_margin')), 'f3')} "
                       f"| **{_fmt(gap, 'pt')}** |")
    out += [""]
    for name, data in runs:
        out += _floor_note(name, _dict(data.get("report")))
        if data.get("evolved_identical_to_seed"):
            out += [f"> **{name}: the evolved candidate is byte-identical to the "
                    f"seed prompt.** The search changed nothing that survived "
                    f"selection; there is no evolved arm to compare.", ""]
    return out + ["_val - test is the overfitting indicator: positive means "
                  "the arm lost accuracy off the set it was selected on._", ""]


def cost(runs) -> list[str]:
    """One call figure, the same one the run printed: every Jev call it made.

    Older files carry only the evaluation count, so the breakdown is shown when
    the run recorded one and left out when it did not, rather than reconstructed.
    """
    out = ["## Cost", ""]
    for name, data in runs:
        calls, spend = _num(data.get("jev_calls")), _num(data.get("spend_usd"))
        failures = _num(data.get("jev_call_failures"))
        parts = _dict(data.get("jev_calls_breakdown"))
        detail = ", ".join(f"{key.replace('_', ' ')} {value}"
                           for key, value in parts.items()
                           if key != "total" and _num(value) is not None)
        out.append(f"- **{name}**: {calls if calls is not None else 'n/a'} Jev "
                   f"calls{f' ({detail})' if detail else ''}, "
                   f"{f'${spend:.5f}' if spend is not None else 'spend n/a'}"
                   f", {len(_list(data.get('mutations')))} mutations applied"
                   + (f", {failures} failed calls" if failures else ""))
    return out + [""]


def learned(runs, task: Task) -> list[str]:
    """Only the clauses the search bolted onto the seed text are interesting.

    options_of() walks the *task's* options, falling back to their seed text, and
    split_clauses() recognises a clause by asking the task's grammar what it can
    produce -- so this is domain-agnostic; no option name is hardcoded."""
    out = ["## What the search learned", ""]
    for name, data in runs:
        options = grammar.options_of(task, _dict(data.get("evolved")))
        split = ((option, grammar.split_clauses(text, task)[1])
                 for option, text in options.items())
        lines = [f"- **{option}** — " + " ".join(cl) for option, cl in split if cl]
        out += ([f"### {name}", ""]
                + (lines or ["_nothing added to the seed criteria._"]) + [""])
    return out


def confusions(runs) -> list[str]:
    out = ["## Confusion pairs attacked", ""]
    for name, data in runs:
        counts = Counter(
            (_option_of(str(m.get("component", "?"))), str(m.get("confused_with")))
            for m in _list(data.get("mutations")) if isinstance(m, dict))
        lines = [f"- {a} vs {b}: {n}" for (a, b), n in counts.most_common(TOP_PAIRS)]
        out += [f"### {name}", ""] + (lines or ["_no mutations recorded._"]) + [""]
    return out


def remaining(runs, arms) -> list[str]:
    out = ["## Remaining test errors", ""]
    for name, data in runs:
        report, lines = _dict(data.get("report")), []
        for arm in (a for a in arms if a in report):
            wrong = [w for w in _list(_side(report, arm, "test").get("wrong"))
                     if isinstance(w, dict)]
            chose = Counter(str(w.get("chose", "?")) for w in wrong)
            top = ", ".join(f"{o} x{n}" for o, n in chose.most_common(TOP_WRONG))
            lines.append(f"- **{arm}**: {len(wrong)} wrong"
                         + (f" — chose {top}" if top else ""))
        out += [f"### {name}", ""] + (lines or ["_no arms in this file._"]) + [""]
    return out


def comparison(name: str, data: dict) -> list[str]:
    """compare.py's evidence: the marginal accuracies, then the pairwise tests
    that are the actual comparison, then what the noise floor makes of them."""
    arms = _dict(data.get("arms"))
    pairs = [p for p in _list(data.get("pairs")) if isinstance(p, dict)]
    resamples = _dict(data.get("bootstrap")).get("resamples", 0)
    out = [f"## Comparison: {name}", "",
           f"_{data.get('n_instances', '?')} shared test instances, split seed "
           f"{data.get('split_seed', '?')}, {resamples} paired bootstrap "
           f"resamples; {len(arms)} arms, {len(pairs)} pairs._", "",
           "| arm | test acc | 95% Wilson (unpaired) | mean margin |",
           "| --- | ---: | :---: | ---: |"]
    for arm, row in ((a, _dict(r)) for a, r in arms.items()):
        lo, hi = (_list(row.get("wilson95")) + [None, None])[:2]
        out.append(f"| {arm} | {_fmt(_num(row.get('accuracy')))} "
                   f"| {_fmt(_num(lo))} – {_fmt(_num(hi))} "
                   f"| {_fmt(_num(row.get('mean_margin')), 'f3')} |")
    out += ["", "_Those intervals are the unpaired view and overlap heavily; they "
                "are NOT the test. The pairwise rows are, because every arm "
                "answered these same instances._", "",
            "### Pairwise (McNemar exact, two-sided)", "",
            "| pair | acc d | b | c | McNemar p | paired 95% CI | verdict |",
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
    out += ["", f"**{len(pairs) - len(undecided)} of {len(pairs)} pairs separate "
                f"at p < {ALPHA}**; this data cannot order the other "
                f"{len(undecided)}, where the sign of the difference is not "
                f"evidence. Expect ~{ALPHA * len(pairs):.1f} false positives "
                f"among {len(pairs)} pairs tested at once.", ""]
    return out + noise_floor(arms, pairs, undecided)


def noise_floor(arms: dict, pairs: list, undecided: list) -> list[str]:
    """A noise twin is a control that should carry no signal at all, so an arm
    that cannot be told apart from it has not been shown to work."""
    floor = next((a for a in arms if NOISE_HINT in a.lower()), None)
    if floor is None:
        return []
    tied = sorted({e["b"] if e.get("a") == floor else e["a"] for e in undecided
                   if floor in (e.get("a"), e.get("b"))})
    tested = sum(1 for e in pairs if floor in (e.get("a"), e.get("b")))
    verdict = (f"- not distinguishable from the floor ({len(tied)} of the "
               f"{tested} arms tested against it): "
               + ", ".join(f"`{a}`" for a in tied)) if tied else (
               f"- all {tested} other arms separate from the floor.")
    accuracy = _fmt(_num(_dict(arms.get(floor)).get("accuracy")))
    return [f"**Noise floor: `{floor}` at {accuracy}.** That arm is the control: "
            f"a pair that does not separate from it is uninterpretable, whichever "
            f"way it points.", "", verdict, ""]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add = parser.add_argument
    add("runs", nargs="*", metavar="NAME=PATH",
        help="results JSON from jevopt.optimize; repeatable")
    add_task_argument(parser)
    add("--compare", action="append", default=[], metavar="NAME=PATH",
        help="comparison JSON from jevopt.compare; repeatable")
    add("--out", help="write the markdown here instead of stdout")
    add("--force", action="store_true", help="overwrite an existing --out file")
    args = parser.parse_args(argv)

    task = load_task(args.task)           # a bad --task is a typo, not a traceback
    check_writable((args.out,), args.force)

    runs = [got for got in (load(spec) for spec in args.runs) if got]
    compares = [got for got in (load(spec) for spec in args.compare) if got]
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
        arms = list({a: None for _n, d in runs for a in _dict(d.get("report"))})
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
