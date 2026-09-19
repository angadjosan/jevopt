"""Paired comparison of prompt arms, on identical test instances.

WHY PAIRED. Arms have been compared by unpaired accuracy on ~89 test instances.
At that size a 95% interval on a single accuracy is about +/-10 points -- wider
than nearly every difference being claimed. A random-clause control appeared to
beat the full optimiser, yet three random seeds spanned 8.8 points, so on
marginal accuracies alone nothing here is distinguishable from anything else.

Every arm, however, answers the SAME instances, so the comparison is paired,
and a paired test looks only where two arms DISAGREE: the shared difficulty of
the test set cancels instead of being charged to both arms as noise. Two arms
differing on 10 instances all one way separate at p < 0.01 even when their
marginal intervals overlap almost entirely.

So: per-arm accuracy with a Wilson interval (the honest marginal uncertainty,
which stays wide), then McNemar's exact test and a paired bootstrap per pair --
the actual evidence about differences. Stdlib only, no normal approximations.

    python3 -m jevopt.compare --candidate gepa=runs/robot.results.json --greedy 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

from . import grammar
from .adapter import JevAdapter
from .baselines import greedy_arm, random_arm
from .evidence import Evidence
from .optimize import load_task
from .task import Task

Z95 = 1.959963984540054
ALPHA = 0.05

# ----------------------------------------------------------------- statistics

def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval: unlike the normal approximation it stays in
    [0, 1] and keeps width at p = 0 or 1."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At p = 0 or 1 the arithmetic leaves a rounding crumb (~1e-18) on the wrong
    # side of the estimate, so clamp against p as well as against [0, 1].
    return max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half))


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact McNemar on the discordant pairs only: b = A right & B wrong,
    c = A wrong & B right. Instances both arms get right (or both wrong) say
    nothing about which is better and are excluded -- that exclusion is the
    whole source of the extra power. Under the null the b+c discordances are
    fair coin flips, so the two-sided exact p is twice the tail at min(b, c)."""
    nb = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    nc = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = nb + nc
    if n == 0:
        return nb, nc, 1.0                      # no disagreement, no evidence
    tail = sum(math.comb(n, i) for i in range(min(nb, nc) + 1)) * (0.5 ** n)
    return nb, nc, min(1.0, 2.0 * tail)


def draw_resamples(n: int, reps: int, seed: int) -> list[list[int]]:
    """Resample INSTANCE INDICES once and reuse the identical draws for every
    arm -- that sharing is what makes the bootstrap paired."""
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(reps)]


def boot_means(values: list[float], draws: list[list[int]]) -> list[float]:
    return [sum(values[i] for i in draw) / len(draw) for draw in draws]


def percentile_ci(diffs: list[float]) -> tuple[float, float]:
    xs = sorted(diffs)
    n = len(xs)
    lo = xs[min(n - 1, int(math.floor(0.025 * n)))]
    hi = xs[max(0, int(math.ceil(0.975 * n)) - 1)]
    return lo, hi


# ---------------------------------------------------------------------- arms

def load_candidate(path: str) -> dict[str, str]:
    """A results JSON's `evolved` key: criteria.<option> -> text."""
    with open(path) as fh:
        data = json.load(fh)
    evolved = data.get("evolved")
    if not isinstance(evolved, dict) or not evolved:
        raise SystemExit(f"{path}: no non-empty 'evolved' candidate in this JSON")
    stray = [k for k in evolved if not k.startswith(grammar.PREFIX)]
    if stray:
        raise SystemExit(f"{path}: keys missing the {grammar.PREFIX!r} prefix: {stray}")
    return dict(evolved)


def build_arms(task: Task, train: list[dict], args) -> list[tuple[str, dict]]:
    arms: list[tuple[str, dict]] = [("seed", grammar.components(task.options))]
    if task.reference is not None:
        arms.append(("reference", grammar.components(task.reference)))
    for spec in args.candidate:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--candidate wants NAME=PATH, got {spec!r}")
        arms.append((name, load_candidate(path)))
    if args.greedy or args.random_seeds:
        evidence = Evidence(task, train)        # gate is train-only, costs no calls
        if args.greedy:
            arms.append((f"greedy.k{args.greedy}",
                         greedy_arm(task, evidence, args.greedy)[0]))
        k = args.greedy or args.random_k
        for seed in range(args.random_seeds):
            arms.append((f"random.s{seed}", random_arm(task, evidence, k, seed)[0]))
    if len({n for n, _ in arms}) != len(arms):
        raise SystemExit(f"duplicate arm names: {[n for n, _ in arms]}")
    return arms


def evaluate(adapter: JevAdapter, candidate: dict, instances: list[dict],
             cache: dict) -> tuple[list[bool], list[float]]:
    """Per-instance correctness and margin, memoised on the candidate text: two
    arms that are the same prompt cost one pass, not two."""
    key = json.dumps(candidate, sort_keys=True)
    if key not in cache:
        traces = adapter.evaluate(instances, candidate, capture_traces=True).trajectories
        cache[key] = ([bool(t["correct"]) for t in traces],
                      [float(t["margin"]) for t in traces])
    return cache[key]


# -------------------------------------------------------------------- report

def pct(x: float) -> str:
    return f"{100 * x:+6.1f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add = parser.add_argument
    add("--task", default="jevopt.tasks.robot", help="module exposing build() -> Task")
    add("--split-seed", type=int, default=0, help="must match the seed the runs used")
    add("--candidate", action="append", default=[], metavar="NAME=PATH",
        help="evolved arm from a results JSON's 'evolved' key; repeatable")
    add("--greedy", type=int, default=0,
        help="clauses/option for a greedy arm (0 = skip)")
    add("--random-seeds", type=int, default=0, help="how many random-clause control arms")
    add("--random-k", type=int, default=2,
        help="clauses/option for random arms if no --greedy")
    add("--bootstrap", type=int, default=10000, help="paired resamples (0 = skip)")
    add("--bootstrap-seed", type=int, default=0)
    add("--limit", type=int, default=0,
        help="truncate the test set, for cheap smoke runs")
    add("--out", help="JSON with the raw per-instance vectors, for re-analysis")
    args = parser.parse_args(argv)

    task = load_task(args.task)
    train, _val, test = task.split(seed=args.split_seed)
    if args.limit > 0:
        test = test[:args.limit]
    arms = build_arms(task, train, args)
    n = len(test)

    print(f"task {task.name}: {len(arms)} arms x {n} shared test instances "
          f"(split seed {args.split_seed}{', TRUNCATED' if args.limit else ''})\n")

    adapter, cache, results = JevAdapter(task), {}, {}
    for name, candidate in arms:
        correct, margins = evaluate(adapter, candidate, test, cache)
        lo, hi = wilson(sum(correct), n)
        results[name] = {"accuracy": sum(correct) / n, "wilson95": [lo, hi],
                         "mean_margin": sum(margins) / n, "correct": correct,
                         "margins": margins, "candidate": candidate}

    width = max(len(name) for name in results)
    print(f"{'arm':{width}s} {'acc':>7s}  {'95% Wilson (unpaired)':>23s} {'margin':>9s}")
    for name, row in results.items():
        lo, hi = row["wilson95"]
        print(f"{name:{width}s} {row['accuracy']:7.1%}  [{lo:6.1%},{hi:7.1%}] "
              f"({100 * (hi - lo) / 2:4.1f}pt) {row['mean_margin']:+9.3f}")
    print("\nThose intervals are the unpaired view and will overlap heavily. They are "
          "NOT the\ntest: the pairwise rows are, because every arm saw these same "
          "instances.\n")

    draws = (draw_resamples(n, args.bootstrap, args.bootstrap_seed)
             if args.bootstrap else [])
    boots = {name: (boot_means([float(c) for c in row["correct"]], draws),
                    boot_means(row["margins"], draws))
             for name, row in results.items()} if draws else {}

    pairs, names = [], list(results)
    for i, a in enumerate(names):
        for b in names[i + 1:]:            # every unordered pair, A-vs-B once
            nb, nc, p = mcnemar_exact(results[a]["correct"], results[b]["correct"])
            entry = {"a": a, "b": b, "b_count": nb, "c_count": nc,
                     "accuracy_diff": results[a]["accuracy"] - results[b]["accuracy"],
                     "mcnemar_p": p, "significant": p < ALPHA,
                     "margin_diff": results[a]["mean_margin"] - results[b]["mean_margin"]}
            if draws:
                for key, col in (("accuracy_diff_ci95", 0), ("margin_diff_ci95", 1)):
                    left, right = boots[a][col], boots[b][col]
                    entry[key] = list(percentile_ci(
                        [x - y for x, y in zip(left, right, strict=True)]))
            pairs.append(entry)

    print(f"{'pair':{2 * width + 4}s} {'acc d':>6s} {'b':>4s} {'c':>4s} "
          f"{'McNemar p':>10s} {'paired 95% CI on acc d':>24s} {'margin d':>9s} "
          f"{'verdict':>9s}")
    for e in pairs:
        ci = e.get("accuracy_diff_ci95")
        ci_s = f"[{pct(ci[0])},{pct(ci[1])}] pt" if ci else f"{'--':>24s}"
        print(f"{e['a'] + ' vs ' + e['b']:{2 * width + 4}s} {pct(e['accuracy_diff'])} "
              f"{e['b_count']:4d} {e['c_count']:4d} {e['mcnemar_p']:10.4f} {ci_s} "
              f"{e['margin_diff']:+9.3f} "
              f"{'SIGNIF' if e['significant'] else 'not dist.':>9s}")

    sig = sum(e["significant"] for e in pairs)
    print(f"\n{sig} of {len(pairs)} pairs separate at p < {ALPHA} "
          f"(McNemar exact, two-sided, no multiplicity correction).")
    for e in (x for x in pairs if not x["significant"]):
        gap = 100 * e["accuracy_diff"]
        lead = ("they tie outright" if gap == 0 else
                f"the {abs(gap):.1f}pt lead for {e['a'] if gap > 0 else e['b']}")
        print(f"  NOT DISTINGUISHABLE: {e['a']} vs {e['b']} -- {lead}, on "
              f"{e['b_count']}/{e['c_count']} discordant instances "
              f"(p={e['mcnemar_p']:.3f})."
              f" This data cannot order them; the sign is not evidence.")
    print(f"\nWith {len(pairs)} pairs tested at once, expect ~{ALPHA * len(pairs):.1f} "
          f"false positives by chance; treat a lone p just under 0.05 as weak."
          f"\n\n{adapter.calls} Jev evaluations, ${adapter.spend:.4f} spent.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"task": task.name, "split_seed": args.split_seed,
                       "limit": args.limit, "n_instances": n,
                       "acceptable": [i["acceptable"] for i in test],
                       "bootstrap": {"resamples": args.bootstrap,
                                     "seed": args.bootstrap_seed},
                       "arms": results, "pairs": pairs, "jev_calls": adapter.calls,
                       "spend_usd": round(adapter.spend, 5)}, fh, indent=2)
        print(f"comparison -> {args.out}")


if __name__ == "__main__":
    main()
