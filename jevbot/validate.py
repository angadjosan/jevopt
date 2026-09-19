"""Does an optimised prompt actually fly the arm?

Dataset accuracy is a proxy: `jevopt` scores a candidate on frozen states, one
step at a time, against a label written in code. This runs the real closed-loop
task instead -- physics, camera, compounding errors, the lot -- once per apple
placement for each prompt arm, and reports how many apples ended up in the air.

    python3 -m jevbot.validate --arms seed,reference,evolved

Three arms are available: `seed` is the task's naive option text, `reference`
its hand-written one, and `evolved` the candidate in a results JSON written by
`python3 -m jevopt.optimize --task jevopt.tasks.robot`.
"""

from __future__ import annotations

import argparse
import json

from jevopt import grammar
from jevopt.tasks import robot

from .eval import SCENES
from .policy import NOULS
from .run import episode


def load_candidate(path: str) -> dict[str, str]:
    """A results JSON's `evolved` key: criteria.<option> -> text.

    Same shape `jevopt.compare` reads, re-read here rather than imported: that
    module pulls in the optimiser (and gepa), which the demo does not need.
    """
    with open(path) as fh:
        data = json.load(fh)
    evolved = data.get("evolved")
    if not isinstance(evolved, dict) or not evolved:
        raise SystemExit(f"{path}: no non-empty 'evolved' candidate in this JSON")
    stray = [key for key in evolved if not key.startswith(grammar.PREFIX)]
    if stray:
        raise SystemExit(f"{path}: keys missing the {grammar.PREFIX!r} prefix: {stray}")
    return dict(evolved)


def build_questions(task, candidate: dict[str, str]) -> dict:
    """The candidate's choice question, plus the two nouls the loop logs.

    `grammar.questions` names the choice question after the optimiser; the
    control loop reads it as `action`. The nouls are not optimised -- they are
    the same in every arm, so the comparison still isolates the option text.
    """
    choice = grammar.questions(task, candidate)[grammar.QUESTION]
    return {"action": choice, **NOULS}


def run_arm(name: str, questions: dict, max_steps: int) -> dict:
    results = []
    for xy in SCENES:
        result, _ = episode(xy, max_steps=max_steps, quiet=True, questions=questions)
        results.append(result)
        print(f"   {name:14s} apple@{xy}  "
              f"{'PICKED UP' if result['success'] else 'failed   '}  "
              f"steps={result['steps']:3d}  ({result['stopped_because']})", flush=True)
    wins = sum(r["success"] for r in results)
    steps = [r["steps"] for r in results if r["success"]]
    mean_steps = round(sum(steps) / len(steps), 1) if steps else None
    return {"success": wins, "total": len(results),
            "mean_steps_when_successful": mean_steps,
            "cost_usd": round(sum(r["cost_usd"] for r in results), 5),
            "scenes": results}


def candidates(task, names: list[str], results_path: str) -> dict[str, dict]:
    """Resolve arm names to candidates, before any Jev call is spent."""
    out = {}
    for name in names:
        if name == "seed":
            out[name] = grammar.components(task.options)
        elif name == "reference":
            if task.reference is None:
                raise SystemExit(f"task {task.name} has no reference prompt")
            out[name] = grammar.components(task.reference)
        elif name == "evolved":
            out[name] = load_candidate(results_path)
        else:
            raise SystemExit(f"unknown arm {name!r} (seed, reference, evolved)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="seed,reference,evolved",
                        help="comma-separated: seed, reference, evolved")
    parser.add_argument("--results", default="runs/robot.results.json",
                        help="results JSON the 'evolved' arm is read from")
    parser.add_argument("--out", default="runs/robot.validation.json")
    parser.add_argument("--max-steps", type=int, default=45)
    args = parser.parse_args()

    task = robot.build()
    arms = candidates(task, [n for n in args.arms.split(",") if n], args.results)

    report = {}
    for name, candidate in arms.items():
        print(f"\n{name}:")
        report[name] = run_arm(name, build_questions(task, candidate), args.max_steps)

    print(f"\nclosed-loop result ({len(SCENES)} placements per arm)")
    for name, res in report.items():
        print(f"  {name:10s} {res['success']}/{res['total']} picked up   "
              f"mean steps {res['mean_steps_when_successful']}   "
              f"${res['cost_usd']:.5f}")
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nvalidation -> {args.out}")


if __name__ == "__main__":
    main()
