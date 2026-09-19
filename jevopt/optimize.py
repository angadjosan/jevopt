"""Evolve a task's option descriptions with GEPA, using a proposer that never
generates.

The Pareto front, minibatch acceptance test and candidate pool are the authors'
GEPA engine; only the mutation operator is replaced. Nothing here knows what the
decision is about: the task is named on the command line, imported, and asked
for its options, instances and conditions.

    python3 -m jevopt.optimize --task jevopt.tasks.robot --budget 1500
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

import gepa

from . import grammar, proposer
from .adapter import JevAdapter
from .task import Task


def load_task(dotted: str) -> Task:
    """Import `dotted` and call its build(). A task module is data plus a
    builder, so it can pull its instances from wherever it likes."""
    return importlib.import_module(dotted).build()


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="jevopt.tasks.robot",
                        help="dotted path to a module exposing build() -> Task")
    parser.add_argument("--budget", type=int, default=1500,
                        help="max metric calls (Jev evaluations)")
    parser.add_argument("--minibatch", type=int, default=10)
    parser.add_argument("--no-jev-choice", action="store_true",
                        help="ablation: take the statistically strongest repair "
                             "instead of letting Jev pick among the shortlist")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--out",
                        help="evolved prompt text (default: runs/<task>.prompt.txt)")
    parser.add_argument("--results",
                        help="results JSON (default: runs/<task>.results.json)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds both the train/val/test split and the search")
    args = parser.parse_args(argv)

    task = load_task(args.task)
    out = args.out or f"runs/{task.name}.prompt.txt"
    results_path = args.results or f"runs/{task.name}.results.json"

    train, val, test = task.split(seed=args.seed)
    adapter = JevAdapter(task)
    mutator = proposer.JevProposer(task, train, use_jev_choice=not args.no_jev_choice)

    print(f"task {task.name}: {len(task.options)} options, "
          f"{len(task.conditions)} conditions\n"
          f"train {len(train)} / val {len(val)} / test {len(test)}; "
          f"budget {args.budget} metric calls; "
          f"repair choice: {'Jev' if not args.no_jev_choice else 'statistical argmax'}\n")

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
        raise_on_exception=False,
        seed=args.seed,
    )

    evolved = dict(result.best_candidate)
    print(f"\nGEPA finished: {adapter.calls} Jev evaluations, "
          f"${adapter.spend:.4f} spent, {len(mutator.log)} mutations applied\n")

    arms = {"seed": grammar.components(task.options), "GEPA-evolved": evolved}
    if task.reference is not None:          # only tasks that have one get the arm
        arms["reference"] = grammar.components(task.reference)

    report = {}
    width = max(len(name) for name in arms)
    print(f"{'arm':{width}s} {'val acc':>9s} {'val margin':>12s}"
          f" {'test acc':>10s} {'test margin':>13s}")
    for name, candidate in arms.items():
        report[name] = {"val": measure(adapter, candidate, val),
                        "test": measure(adapter, candidate, test)}
        val_s, test_s = report[name]["val"], report[name]["test"]
        print(f"{name:{width}s} {val_s['accuracy']:9.1%} {val_s['mean_margin']:+12.3f}"
              f" {test_s['accuracy']:10.1%} {test_s['mean_margin']:+13.3f}")

    for path in (out, results_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(grammar.render_prompt(task, evolved) + "\n")
    with open(results_path, "w") as fh:
        json.dump({"task": task.name, "report": report, "mutations": mutator.log,
                   "evolved": evolved,
                   "jev_calls": adapter.calls, "split_seed": args.seed,
                   "spend_usd": round(adapter.spend, 5)},
                  fh, indent=2)
    print(f"\nevolved prompt -> {out}\nresults -> {results_path}")


if __name__ == "__main__":
    main()
