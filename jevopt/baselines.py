"""Controls that skip the search: is GEPA earning its keep?

The evolved prompt is built from two ingredients -- a clause grammar and an
evidence gate -- wired together by GEPA's Pareto search. Only the third part is
expensive, so the honest question is whether it contributes anything the first
two do not already give for free.

Two arms answer that, both built in a single pass with no acceptance tests, no
minibatches and no Jev calls during construction:

  greedy   attach each option's top-K sound clauses, ranked by the same
           Evidence.score the proposer shortlists with
  random   attach K clauses drawn uniformly from that option's sound pool,
           averaged over several seeds

If greedy matches the evolved prompt, the search is decoration. If random also
matches, even the ranking is decoration and the grammar is doing all the work.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from . import grammar
from .adapter import JevAdapter
from .evidence import Evidence
from .optimize import (
    CallMeter,
    add_task_argument,
    check_failures,
    check_split,
    check_writable,
    load_task,
    measure,
    warn_conflicts,
)
from .task import Task

TEMPLATES = ("only", "never", "prefer")


def sound_clauses(task: Task, evidence: Evidence, option: str) -> list[dict]:
    """Every clause the evidence gate admits for this option, best first."""
    out, seen = [], set()
    for condition in task.conditions:
        for template in TEMPLATES:
            partners = ([o for o in task.options if o != option]
                        if template == "prefer" else [None])
            for partner in partners:
                strength = evidence.score(option, template, condition.id, partner)
                if strength is None:
                    continue
                clause = grammar.render_clause(task, template, condition.id, partner)
                if clause in seen:
                    continue
                seen.add(clause)
                out.append({"clause": clause, "strength": round(strength, 3),
                            "template": template, "condition": condition.id,
                            "partner": partner})
    # Ties are common (strength is a coverage rate), so break them on the clause
    # text to keep the greedy arm deterministic across runs.
    out.sort(key=lambda row: (-row["strength"], row["clause"]))
    return out


def _build(task: Task, picks: dict[str, list[dict]]) -> dict[str, str]:
    options = dict(task.options)
    for option, rows in picks.items():
        for row in rows:
            options[option] = grammar.apply_clause(options[option], row["clause"])
    return grammar.components(options)


def greedy_arm(task: Task, evidence: Evidence, k: int) -> tuple[dict, dict]:
    picks = {o: sound_clauses(task, evidence, o)[:k] for o in task.options}
    return _build(task, picks), picks


def random_arm(task: Task, evidence: Evidence, k: int, seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    picks = {}
    for option in task.options:
        pool = sound_clauses(task, evidence, option)
        picks[option] = rng.sample(pool, min(k, len(pool)))
    return _build(task, picks), picks


def _summary(picks: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {option: [row["clause"] for row in rows] for option, rows in picks.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_task_argument(parser)
    parser.add_argument("--k", type=int, default=2, help="clauses attached per option")
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--out",
                        help="results JSON (default: runs/<task>.baselines.json)")
    parser.add_argument("--limit", type=int, default=0,
                        help="truncate val/test to this many instances (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing results file")
    parser.add_argument("--seed", type=int, default=0,
                        help="split seed; must match the run being compared against")
    args = parser.parse_args(argv)

    task = load_task(args.task)
    out = args.out or f"runs/{task.name}.baselines.json"
    check_writable((out,), args.force)
    train, val, test = task.split(seed=args.seed)
    check_split(task, {"train": train, "val": val, "test": test})
    warn_conflicts(task)
    if args.limit > 0:
        val, test = val[:args.limit], test[:args.limit]

    evidence = Evidence(task, train)
    adapter = JevAdapter(task)

    arms = [("greedy", *greedy_arm(task, evidence, args.k))]
    for seed in range(args.random_seeds):
        arms.append((f"random.s{seed}", *random_arm(task, evidence, args.k, seed)))

    print(f"task {task.name}; train {len(train)} / val {len(val)} / test {len(test)}; "
          f"k={args.k}; {args.random_seeds} random seed(s)\n"
          f"budget plan: {len(arms)} arms x (val {len(val)} + test {len(test)}"
          f" = {len(val) + len(test)}) = {len(arms) * (len(val) + len(test))} "
          f"Jev evaluations; the construction itself costs nothing.\n")

    report = {}
    with CallMeter() as meter:
        for name, candidate, picks in arms:
            report[name] = {
                "val": measure(adapter, candidate, val),
                "test": measure(adapter, candidate, test),
                "clauses": _summary(picks),
                "prompt": grammar.render_prompt(task, candidate),
            }

    randoms = [r for n, r in report.items() if n.startswith("random.")]
    if randoms:
        report["random.mean"] = {
            split: {key: sum(r[split][key] for r in randoms) / len(randoms)
                    for key in ("accuracy", "mean_margin", "mean_confidence")}
            for split in ("val", "test")
        }

    # Refuse BEFORE the table, not after it. A reader going top-down would
    # otherwise meet a full page of 0.0% / -1.000 rows -- the exact confident
    # fake result this check exists to suppress -- and only then the refusal.
    check_failures(adapter, "these baselines")

    # n is printed beside the accuracies: a few points between arms on this many
    # instances is not a difference, and the reader should not have to go looking.
    print(f"{'arm':14s} {'n val':>6s} {'val acc':>8s} {'n test':>7s} "
          f"{'test acc':>9s} {'test margin':>12s}")
    for name in list(report):
        row = report[name]
        print(f"{name:14s} {row['val'].get('n', len(val)):6d} "
              f"{row['val']['accuracy']:8.1%} {row['test'].get('n', len(test)):7d} "
              f"{row['test']['accuracy']:9.1%} {row['test']['mean_margin']:+12.3f}")

    print(f"\nJev calls: {meter.calls} total, all of them evaluations "
          f"({adapter.calls} measured); ${adapter.spend:.4f} spent.")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": task.name, "k": args.k, "limit": args.limit,
                   "report": report, "jev_calls": meter.calls,
                   "jev_calls_breakdown": {"measurement_evaluations": adapter.calls,
                                           "total": meter.calls},
                   "jev_call_failures": getattr(adapter, "failures", 0),
                   "spend_usd": round(adapter.spend, 5)}, fh, indent=2)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
