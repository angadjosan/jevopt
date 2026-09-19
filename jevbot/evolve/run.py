"""Evolve the action criteria with GEPA, using a proposer that never generates.

The Pareto front, minibatch acceptance test and candidate pool are the authors'
GEPA engine; only the mutation operator is replaced.
"""

from __future__ import annotations

import argparse
import json
import os

import gepa

from . import prompt, proposer
from .adapter import JevAdapter


def load(path: str):
    with open(path) as fh:
        data = json.load(fh)
    return data["train"], data["val"], data["test"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="jevbot/evolve/dataset.json")
    parser.add_argument("--budget", type=int, default=1500,
                        help="max metric calls (Jev evaluations)")
    parser.add_argument("--minibatch", type=int, default=10)
    parser.add_argument("--no-jev-choice", action="store_true",
                        help="ablation: take the statistically strongest repair "
                             "instead of letting Jev pick among the shortlist")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--out", default="jevbot/evolve/evolved_prompt.txt")
    parser.add_argument("--results", default="jevbot/evolve/results.json")
    args = parser.parse_args()

    train, val, test = load(args.dataset)
    adapter = JevAdapter()
    mutator = proposer.JevProposer(train, use_jev_choice=not args.no_jev_choice)

    print(f"train {len(train)} / val {len(val)} / test {len(test)}; "
          f"budget {args.budget} metric calls; "
          f"repair choice: {'Jev' if not args.no_jev_choice else 'statistical argmax'}\n")

    result = gepa.optimize(
        seed_candidate=proposer.seed_candidate(),
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
        seed=0,
    )

    evolved = dict(result.best_candidate)
    print(f"\nGEPA finished: {adapter.calls} Jev evaluations, "
          f"${adapter.spend:.4f} spent, {len(mutator.log)} mutations applied\n")

    arms = {
        "seed (naive)": proposer.seed_candidate(),
        "hand-tuned": proposer.handtuned_candidate(),
        "GEPA-evolved": evolved,
    }
    report = {}
    for name, candidate in arms.items():
        report[name] = {"val": measure(adapter, candidate, val),
                        "test": measure(adapter, candidate, test)}
        val_s, test_s = report[name]["val"], report[name]["test"]
        print(f"{name:14s}  val acc {val_s['accuracy']:.1%}  margin {val_s['mean_margin']:+.3f}"
              f"   |  test acc {test_s['accuracy']:.1%}  margin {test_s['mean_margin']:+.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(prompt.render_prompt(evolved) + "\n")
    with open(args.results, "w") as fh:
        json.dump({"report": report, "mutations": mutator.log,
                   "evolved": evolved,
                   "jev_calls": adapter.calls, "spend_usd": round(adapter.spend, 5)},
                  fh, indent=2)
    print(f"\nevolved prompt -> {args.out}\nresults -> {args.results}")


if __name__ == "__main__":
    main()
