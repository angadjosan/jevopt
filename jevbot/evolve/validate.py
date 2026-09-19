"""Does the evolved prompt actually fly the arm?

Dataset accuracy is a proxy. This runs the real closed-loop task -- physics,
camera, the lot -- once per apple placement for each prompt arm.
"""

from __future__ import annotations

import argparse
import json

from ..eval import SCENES
from ..run import episode
from . import prompt, proposer


def run_arm(name: str, candidate: dict, max_steps: int) -> dict:
    questions = prompt.questions(candidate, with_nouls=True)
    results = []
    for xy in SCENES:
        result, _ = episode(xy, max_steps=max_steps, quiet=True, questions=questions)
        results.append(result)
        print(f"   {name:14s} apple@{xy}  "
              f"{'PICKED UP' if result['success'] else 'failed   '}  "
              f"steps={result['steps']:3d}  ({result['stopped_because']})", flush=True)
    wins = sum(r["success"] for r in results)
    steps = [r["steps"] for r in results if r["success"]]
    return {"success": wins, "total": len(results),
            "mean_steps_when_successful": round(sum(steps) / len(steps), 1) if steps else None,
            "cost_usd": round(sum(r["cost_usd"] for r in results), 5),
            "scenes": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="jevbot/evolve/results.json")
    parser.add_argument("--out", default="jevbot/evolve/validation.json")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--arms", default="seed,handtuned,evolved")
    args = parser.parse_args()

    with open(args.results) as fh:
        evolved = json.load(fh)["evolved"]

    available = {
        "seed": proposer.seed_candidate(),
        "handtuned": proposer.handtuned_candidate(),
        "evolved": evolved,
    }
    report = {}
    for name in args.arms.split(","):
        print(f"\n{name}:")
        report[name] = run_arm(name, available[name], args.max_steps)

    print("\nclosed-loop result")
    for name, res in report.items():
        print(f"  {name:10s} {res['success']}/{res['total']} picked up   "
              f"mean steps {res['mean_steps_when_successful']}")
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
