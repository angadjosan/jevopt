"""Controls that skip the search: is GEPA earning its keep?

The evolved prompt is built from two ingredients -- a clause grammar and an
evidence gate -- wired together by GEPA's Pareto search. Only the third part is
expensive, so the honest question is whether it contributes anything the first
two do not already give for free.

Two arms answer that, both built in a single pass with no acceptance tests, no
minibatches and no Jev calls during construction:

  greedy   attach each action's top-K sound clauses, ranked by the same
           Evidence.score the proposer shortlists with
  random   attach K clauses drawn uniformly from that action's sound pool,
           averaged over several seeds

If greedy matches the evolved prompt, the search is decoration. If random also
matches, even the ranking is decoration and the grammar is doing all the work.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from ..sim import ACTIONS
from . import prompt, proposer
from .adapter import JevAdapter
from .run import load, measure

TEMPLATES = ("only", "never", "prefer")


def sound_clauses(evidence: proposer.Evidence, action: str) -> list[dict]:
    """Every clause the evidence gate admits for this action, best first."""
    out, seen = [], set()
    for cid, _phrase, _test in prompt.CONDITIONS:
        for template in TEMPLATES:
            partners = [a for a in ACTIONS if a != action] if template == "prefer" else [None]
            for partner in partners:
                strength = evidence.score(action, template, cid, partner)
                if strength is None:
                    continue
                clause = prompt.render_clause(template, cid, partner)
                if clause in seen:
                    continue
                seen.add(clause)
                out.append({"clause": clause, "strength": round(strength, 3),
                            "template": template, "condition": cid, "partner": partner})
    # Ties are common (strength is a coverage rate), so break them on the clause
    # text to keep the greedy arm deterministic across runs.
    out.sort(key=lambda row: (-row["strength"], row["clause"]))
    return out


def _build(picks: dict[str, list[dict]]) -> dict[str, str]:
    criteria = dict(prompt.SEED_CRITERIA)
    for action, rows in picks.items():
        for row in rows:
            criteria[action] = prompt.apply_clause(criteria[action], row["clause"])
    return prompt.components(criteria)


def greedy_arm(evidence: proposer.Evidence, k: int) -> tuple[dict, dict]:
    picks = {a: sound_clauses(evidence, a)[:k] for a in ACTIONS}
    return _build(picks), picks


def random_arm(evidence: proposer.Evidence, k: int, seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    picks = {}
    for action in ACTIONS:
        pool = sound_clauses(evidence, action)
        picks[action] = rng.sample(pool, min(k, len(pool)))
    return _build(picks), picks


def _summary(picks: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {action: [row["clause"] for row in rows] for action, rows in picks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="jevbot/evolve/dataset.json")
    parser.add_argument("--k", type=int, default=2, help="clauses attached per action")
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--out", default="jevbot/evolve/baselines.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="truncate val/test to this many instances (0 = all)")
    args = parser.parse_args()

    train, val, test = load(args.dataset)
    if args.limit > 0:
        val, test = val[:args.limit], test[:args.limit]

    evidence = proposer.Evidence(train)
    adapter = JevAdapter()

    arms = [("greedy", *greedy_arm(evidence, args.k))]
    for seed in range(args.random_seeds):
        arms.append((f"random.s{seed}", *random_arm(evidence, args.k, seed)))

    print(f"train {len(train)} / val {len(val)} / test {len(test)}; "
          f"k={args.k}; {args.random_seeds} random seed(s)\n")

    report = {}
    for name, candidate, picks in arms:
        report[name] = {
            "val": measure(adapter, candidate, val),
            "test": measure(adapter, candidate, test),
            "clauses": _summary(picks),
            "prompt": prompt.render_prompt(candidate),
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"k": args.k, "limit": args.limit, "report": report,
                   "jev_calls": adapter.calls, "spend_usd": round(adapter.spend, 5)},
                  fh, indent=2)
    print(f"\n{adapter.calls} Jev evaluations, ${adapter.spend:.4f} spent"
          f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
