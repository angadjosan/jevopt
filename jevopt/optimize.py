"""Evolve a task's option descriptions with GEPA, using a proposer that never
generates.

The Pareto front, minibatch acceptance test and candidate pool are the authors'
GEPA engine; only the mutation operator is replaced. Nothing here knows what the
decision is about: the task is named on the command line, imported, and asked
for its options, instances and conditions.

    python3 -m jevopt.optimize --task jevopt.tasks.triage --budget 1500

This module also holds the checks every spending command shares -- loading a
task, refusing an unusable split, warning about contradictory labels, counting
every Jev call, and refusing to overwrite a previous run's files -- because
baselines.py, compare.py and report.py already import from here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
from collections import Counter

import gepa

from . import client, conditions, grammar, proposer
from .adapter import JevAdapter
from .task import Task

# A failed Jev call scores -1.0, which is exactly what a confidently wrong
# answer scores, so a run against a dead key prints a full table of 0.0% and
# exits 0. Past this share of failed calls the numbers are not a result and are
# not written anywhere. Two percent (with at least one failure always forgiven)
# is the widest band that still separates the two regimes seen in practice: a
# 429 that outlived its retries is one call in hundreds, while a bad key, a
# wrong base URL or an unreachable API fails every single call.
MAX_FAIL_RATE = 0.02

TASK_HELP = ("dotted path to a module exposing build() -> Task; the module must "
             "be importable from here (cwd or PYTHONPATH), e.g. jevopt.tasks.triage")


# ------------------------------------------------------------ loading a task

def load_task(dotted: str) -> Task:
    """Import `dotted` and call its build(). A task module is data plus a
    builder, so it can pull its instances from wherever it likes.

    A mistyped module used to surface as a dozen frames of importlib internals.
    The failure is a typo or a missing PYTHONPATH, so it is reported as one line
    that says which of the three things went wrong.
    """
    try:
        module = importlib.import_module(dotted)
    except Exception as exc:
        raise SystemExit(
            f"cannot load task {dotted!r}: {type(exc).__name__}: {exc}\n"
            f"  --task takes a {TASK_HELP}") from exc
    build = getattr(module, "build", None)
    if not callable(build):
        raise SystemExit(
            f"cannot load task {dotted!r}: the module imported, but has no "
            f"build() function.\n  A task module exposes build() -> Task; "
            f"{dotted!r} may be the package rather than the task module.")
    try:
        return build()
    except Exception as exc:
        raise SystemExit(f"cannot load task {dotted!r}: build() raised "
                         f"{type(exc).__name__}: {exc}") from exc


def add_task_argument(parser: argparse.ArgumentParser) -> None:
    """--task is required, never defaulted.

    It used to default to the repo's demo task, so a mistyped or forgotten
    --task produced a confident report about somebody else's data -- or, in the
    spending commands, bought one. There is no task a general tool can guess, and
    the cost of requiring it is one line of usage text.
    """
    parser.add_argument("--task", required=True, help=TASK_HELP)


# ------------------------------------------------------------------ preflight

def check_writable(paths, force: bool) -> None:
    """Refuse to overwrite existing output. Called before anything is spent.

    The default output paths are runs/<task>.*, which is where this repo's
    published evidence lives, so the quickstart as written destroys the
    artifacts docs/findings.md cites.
    """
    existing = sorted({p for p in paths if p and os.path.exists(p)})
    if existing and not force:
        listing = "\n".join(f"    {p}" for p in existing)
        raise SystemExit(
            f"refusing to overwrite {len(existing)} existing file(s):\n{listing}\n"
            "  Those may be a previous run's published results. Write elsewhere "
            "(--out/--results), or pass --force to replace them.")


def _split_fault(task: Task, name: str, parts: dict[str, list]) -> str:
    n_signatures = len({task.signature(i["state"]) for i in task.instances})
    sizes = ", ".join(f"{k} {len(v)}" for k, v in parts.items())
    lines = [
        f"the {name} split is empty, so this run cannot measure anything.",
        f"  {len(task.instances)} instances collapse to {n_signatures} distinct "
        f"condition signature(s); the split works on signatures, not instances, "
        f"so {n_signatures} of them cannot fill three parts ({sizes}).",
        f"  Likely cause: too few conditions ({len(task.conditions)}) for the "
        f"states. conditions.diagnose(task.instances) says:",
    ]
    diagnose = getattr(conditions, "diagnose", None)
    if diagnose is None:                      # tolerate an older conditions.py
        lines.append("    (conditions.diagnose unavailable in this build)")
        return "\n".join(lines)
    report = diagnose(task.instances)
    kept = report.get("kept", {})
    lines.append("    kept: " + (", ".join(f"{f} ({n} values)"
                                           for f, n in kept.items()) or "nothing"))
    for field, (n_values, reason) in report.get("dropped", {}).items():
        lines.append(f"    dropped: {field} ({n_values} values) -- {reason}")
    lines.append("  Add conditions the states do not suggest, or widen "
                 "conditions.MAX_VALUES, so the fields the decision turns on "
                 "become part of the grammar.")
    return "\n".join(lines)


def check_split(task: Task, parts: dict[str, list]) -> None:
    """Refuse to start on an empty split.

    An empty valset is not a bad run, it is a non-terminating one: GEPA raises
    AssertionError every iteration, raise_on_exception=False swallows it, and a
    failed iteration spends no budget, so the loop never ends -- three million
    log lines in forty seconds, exit code never returned.
    """
    for name, part in parts.items():
        if not part:
            raise SystemExit("refusing to start: " + _split_fault(task, name, parts))


def warn_conflicts(task: Task) -> None:
    """Warn about instances the grammar cannot tell apart but that disagree.

    Contradictory labels cap every arm below 100% and cost search budget on a
    confusion no wording can fix, so it is said before anything is spent.
    """
    groups = getattr(task, "conflicting_groups", None)
    groups = groups() if callable(groups) else []
    if not groups:
        return
    instances = sum(g["size"] for g in groups)
    worst = groups[0]
    labels = " vs ".join("/".join(label) for label in worst["labels"])
    print(
        f"warning: contradictory training data -- {len(groups)} signature "
        f"group(s) covering {instances} of {len(task.instances)} instances hold "
        f"states this task's conditions cannot tell apart, yet whose labels "
        f"disagree.\n"
        f"  largest group: {worst['size']} instances labelled {labels}\n"
        f"    example state: {json.dumps(worst['example'], sort_keys=True)}\n"
        f"  Likely cause: the field the decision turns on is not in the grammar "
        f"(dropped as an identifier, or never in the state). Those instances are "
        f"unlearnable by construction and cap every arm below 100%.",
        file=sys.stderr)


def check_failures(adapter: JevAdapter, stage: str, announce: bool = True) -> None:
    """Refuse to report a run whose calls mostly failed. Always says the count.

    A failure is scored, not raised: it lands in the table as a wrong answer
    with margin -1.0. So the count has to be printed even when it is zero, and
    past MAX_FAIL_RATE nothing is written at all.
    """
    calls = max(adapter.calls, 1)
    failures = getattr(adapter, "failures", 0)
    if announce or failures:
        print(f"{failures} of {adapter.calls} Jev calls failed "
              f"({failures / calls:.1%})", file=sys.stderr)
    if failures <= max(1, int(MAX_FAIL_RATE * calls)):
        return
    first = getattr(adapter, "first_error", None) or "(no error text recorded)"
    raise SystemExit(
        f"refusing to report a run that mostly failed: {failures} of "
        f"{adapter.calls} Jev calls raised during {stage}.\n"
        f"  first error: {first}\n"
        "  A failed call scores -1.0, the same as a confidently wrong answer, so "
        "these numbers would read as a result. Nothing has been written.")


class CallMeter:
    """Counts every Jev call, wherever it is made.

    adapter.calls counts evaluation calls only. The proposer asks Jev to pick
    each repair through client.ask directly, and those calls reached no counter
    at all, so the reported figure understated the spend by however hard the
    search worked. client.ask is the one place they all pass through.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()
        self._real = None

    def __enter__(self) -> CallMeter:
        self._real = client.ask
        real = self._real

        def counted(*args, **kwargs):
            with self._lock:
                self.calls += 1
            return real(*args, **kwargs)

        client.ask = counted
        return self

    def __exit__(self, *exc) -> None:
        client.ask = self._real


class QuietLogger:
    """GEPA's per-iteration bookkeeping, dropped.

    It is 70-120 lines a run and buries the summary. Anything that reads like a
    problem still gets through: --quiet hides noise, not failures.
    """

    KEEP = ("exception", "error", "warning", "traceback")

    def log(self, message: str) -> None:
        if any(word in message.lower() for word in self.KEEP):
            print(message)


# ---------------------------------------------------------------- measurement

def measure(adapter: JevAdapter, candidate: dict, instances: list) -> dict:
    batch = adapter.evaluate(instances, candidate, capture_traces=True)
    traces = batch.trajectories
    correct = sum(t["correct"] for t in traces)
    return {
        "accuracy": correct / len(traces),
        "mean_margin": sum(t["margin"] for t in traces) / len(traces),
        "mean_confidence": sum(t["confidence"] for t in traces) / len(traces),
        "n": len(traces),
        "wrong": [{"chose": t["chosen"], "should_be": t["acceptable"]}
                  for t in traces if not t["correct"]],
    }


def majority_option(instances: list) -> str:
    """The option acceptable in the most training instances, ties by name."""
    counts = Counter(option for i in instances for option in i["acceptable"])
    return min(counts, key=lambda option: (-counts[option], option))


def measure_majority(option: str, instances: list) -> dict:
    """Score "always answer the commonest option" -- no prompt, no Jev calls.

    It is the number an arm has to beat before it has demonstrated anything: a
    prompt below this row lost to a constant.
    """
    hits = [option in i["acceptable"] for i in instances]
    return {
        "accuracy": sum(hits) / len(instances),
        "mean_margin": None, "mean_confidence": None,
        "n": len(instances), "always_answers": option,
        "wrong": [{"chose": option, "should_be": sorted(i["acceptable"])}
                  for i, hit in zip(instances, hits, strict=True) if not hit],
    }


def _cell(value, spec: str, width: int) -> str:
    return f"{'--':>{width}}" if value is None else f"{value:{spec}}"


def print_results_table(report: dict) -> None:
    """One row per arm, with n, so a difference can be read against its sample."""
    width = max(len(name) for name in report)
    print(f"{'arm':{width}s} {'n val':>6s} {'val acc':>9s} {'val margin':>12s}"
          f" {'n test':>7s} {'test acc':>10s} {'test margin':>13s}")
    for name, row in report.items():
        val, test = row["val"], row["test"]
        print(f"{name:{width}s} {val['n']:6d} {_cell(val['accuracy'], '9.1%', 9)}"
              f" {_cell(val['mean_margin'], '+12.3f', 12)}"
              f" {test['n']:7d} {_cell(test['accuracy'], '10.1%', 10)}"
              f" {_cell(test['mean_margin'], '+13.3f', 13)}")


def noise_floor(report: dict, seed: str, repeat: str) -> dict:
    """How far apart the same prompt lands from itself, scored twice.

    Jev samples, so an arm gap smaller than this is not a finding whichever way
    it points -- and this table has reported 4.8-point "regressions" between one
    prompt and itself.
    """
    floor = {split: abs(report[seed][split]["accuracy"]
                        - report[repeat][split]["accuracy"])
             for split in ("val", "test")}
    print(f"\nnoise floor: the seed prompt, scored twice, moved "
          f"{100 * floor['val']:.1f} pt on val and {100 * floor['test']:.1f} pt "
          f"on test (n={report[seed]['val']['n']}/{report[seed]['test']['n']}). "
          f"That is measurement noise, not a result.")
    inside = [name for name, row in report.items()
              if name not in (seed, repeat)
              and abs(row["test"]["accuracy"]
                      - report[seed]["test"]["accuracy"]) <= floor["test"]]
    if inside:
        print("  within that floor on test, i.e. not shown to differ from the "
              "seed: " + ", ".join(inside))
    return floor


# ----------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_task_argument(parser)
    parser.add_argument("--budget", type=int, default=1500,
                        help="max metric calls the SEARCH may spend (measurement "
                             "is extra and is stated up front)")
    parser.add_argument("--minibatch", type=int, default=10)
    parser.add_argument("--no-jev-choice", action="store_true",
                        help="ablation: take the statistically strongest repair "
                             "instead of letting Jev pick among the shortlist")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--out",
                        help="evolved prompt text (default: runs/<task>.prompt.txt)")
    parser.add_argument("--results",
                        help="results JSON (default: runs/<task>.results.json)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing output files")
    parser.add_argument("--quiet", action="store_true",
                        help="drop GEPA's per-iteration log lines")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds both the train/val/test split and the search")
    args = parser.parse_args(argv)

    task = load_task(args.task)
    out = args.out or f"runs/{task.name}.prompt.txt"
    results_path = args.results or f"runs/{task.name}.results.json"
    check_writable((out, results_path), args.force)

    train, val, test = task.split(seed=args.seed)
    check_split(task, {"train": train, "val": val, "test": test})
    warn_conflicts(task)

    adapter = JevAdapter(task)
    mutator = proposer.JevProposer(task, train, use_jev_choice=not args.no_jev_choice)

    # Every arm scored with Jev: the seed, the seed again (the noise floor), the
    # evolved candidate, and the reference if the task has one. The majority arm
    # is free. Stated before anything is spent, so the run can be cost-planned.
    scored_arms = 3 + (1 if task.reference is not None else 0)
    per_arm = len(val) + len(test)
    planned = args.budget + scored_arms * per_arm

    print(f"task {task.name}: {len(task.options)} options, "
          f"{len(task.conditions)} conditions\n"
          f"train {len(train)} / val {len(val)} / test {len(test)}; "
          f"repair choice: {'Jev' if not args.no_jev_choice else 'statistical argmax'}\n"
          f"budget plan: {args.budget} search evaluations + {scored_arms} scored "
          f"arms x (val {len(val)} + test {len(test)} = {per_arm}) = {planned} "
          f"Jev evaluations.\n"
          f"  The majority arm costs nothing, and an evolved candidate identical "
          f"to the seed drops an arm ({per_arm} fewer). The search budget is "
          f"checked between iterations, so it can overrun by up to one iteration "
          f"(minibatch {args.minibatch} + val {len(val)}); the proposer's repair "
          f"calls are extra and are counted in the total reported below.\n")

    with CallMeter() as meter:
        result = gepa.optimize(
            seed_candidate=grammar.components(task.options),
            trainset=train,
            valset=val,
            adapter=adapter,
            custom_candidate_proposer=mutator,
            candidate_selection_strategy="pareto",
            module_selector="round_robin",
            use_merge=not args.no_merge,
            reflection_minibatch_size=args.minibatch,
            max_metric_calls=args.budget,
            display_progress_bar=False,
            logger=QuietLogger() if args.quiet else None,
            # A GEPA failure must stop the run. Swallowing it made a failing
            # iteration free -- it consumed no budget -- so the search spun an
            # unbounded loop instead of ending.
            raise_on_exception=True,
            seed=args.seed,
        )
        search_calls, search_evals = meter.calls, adapter.calls

        evolved = dict(result.best_candidate)
        print(f"\nGEPA search finished: {search_evals} Jev evaluations "
              f"+ {search_calls - search_evals} proposer repair calls, "
              f"${adapter.spend:.4f} spent, {len(mutator.log)} mutations applied")
        # Checked here rather than only at the end: if the calls are failing,
        # the measurement passes would spend the same way for the same nothing.
        check_failures(adapter, "the search", announce=False)

        seed_candidate = grammar.components(task.options)
        identical = evolved == seed_candidate
        if identical:
            print("\n!! the evolved candidate is BYTE-IDENTICAL to the seed "
                  "prompt: the search changed nothing that survived selection. "
                  "It is not shown as a separate arm below -- two scores for one "
                  "prompt are two samples of the same thing, and their "
                  "difference is noise.")

        arms = {"seed": seed_candidate, "seed (repeat)": seed_candidate}
        if not identical:
            arms["GEPA-evolved"] = evolved
        if task.reference is not None:      # only tasks that have one get the arm
            arms["reference"] = grammar.components(task.reference)

        print()
        report = {name: {"val": measure(adapter, candidate, val),
                         "test": measure(adapter, candidate, test)}
                  for name, candidate in arms.items()}
        majority = majority_option(train)
        report["majority"] = {"val": measure_majority(majority, val),
                              "test": measure_majority(majority, test)}

    print_results_table(report)
    print(f"majority = always answer {majority!r}, the commonest acceptable "
          f"option in train; it asks Jev nothing.")
    floor = noise_floor(report, "seed", "seed (repeat)")

    measurement_evals = adapter.calls - search_evals
    print(f"\nJev calls: {meter.calls} total = {search_evals} search evaluations "
          f"+ {search_calls - search_evals} proposer repair calls "
          f"+ {measurement_evals} measurement evaluations "
          f"(planned {planned}); ${adapter.spend:.4f} spent.")
    check_failures(adapter, "the run")

    for path in (out, results_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(grammar.render_prompt(task, evolved) + "\n")
    with open(results_path, "w") as fh:
        json.dump({"task": task.name, "report": report, "mutations": mutator.log,
                   "evolved": evolved,
                   "evolved_identical_to_seed": identical,
                   "noise_floor": floor,
                   "majority_option": majority,
                   "jev_calls": meter.calls,
                   "jev_calls_breakdown": {
                       "search_evaluations": search_evals,
                       "proposer_repair_calls": search_calls - search_evals,
                       "measurement_evaluations": measurement_evals,
                       "planned": planned, "total": meter.calls},
                   "jev_call_failures": getattr(adapter, "failures", 0),
                   "split_seed": args.seed,
                   "spend_usd": round(adapter.spend, 5)},
                  fh, indent=2)
    print(f"\nevolved prompt -> {out}\nresults -> {results_path}")


if __name__ == "__main__":
    main()
