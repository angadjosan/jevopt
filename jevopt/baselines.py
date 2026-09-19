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
from .optimize import load_task, measure
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
    parser.add_argument("--task", default="jevopt.tasks.triage",
                        help="dotted path to a module exposing build() -> Task")
    parser.add_argument("--k", type=int, default=2, help="clauses attached per option")
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--out",
                        help="results JSON (default: runs/<task>.baselines.json)")
    parser.add_argument("--limit", type=int, default=0,
                        help="truncate val/test to this many instances (0 = all)")
    parser.add_argument("--seed", type=int, default=0,
                        help="split seed; must match the run being compared against")
    args = parser.parse_args(argv)

    task = load_task(args.task)
    out = args.out or f"runs/{task.name}.baselines.json"
    train, val, test = task.split(seed=args.seed)
    if args.limit > 0:
        val, test = val[:args.limit], test[:args.limit]

    evidence = Evidence(task, train)
    adapter = JevAdapter(task)

    arms = [("greedy", *greedy_arm(task, evidence, args.k))]
    for seed in range(args.random_seeds):
        arms.append((f"random.s{seed}", *random_arm(task, evidence, args.k, seed)))

    print(f"task {task.name}; train {len(train)} / val {len(val)} / test {len(test)}; "
          f"k={args.k}; {args.random_seeds} random seed(s)\n")

    report = {}
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

    print(f"{'arm':14s} {'val acc':>8s} {'test acc':>9s} {'test margin':>12s}")
    for name in list(report):
        row = report[name]
        print(f"{name:14s} {row['val']['accuracy']:8.1%} {row['test']['accuracy']:9.1%} "
              f"{row['test']['mean_margin']:+12.3f}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"task": task.name, "k": args.k, "limit": args.limit,
                   "report": report, "jev_calls": adapter.calls,
                   "spend_usd": round(adapter.spend, 5)}, fh, indent=2)
    print(f"\n{adapter.calls} Jev evaluations, ${adapter.spend:.4f} spent"
          f"\nresults -> {out}")


if __name__ == "__main__":
    main()
