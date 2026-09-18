"""Run Jev over several apple placements and report a success rate."""

from __future__ import annotations

import argparse
import json

from .run import episode

SCENES = [(0.40, 0.10), (0.30, -0.15), (0.35, 0.22), (0.25, 0.00),
          (0.20, -0.25), (0.42, -0.05)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args()

    results, total = [], 0.0
    for xy in SCENES:
        result, _ = episode(xy, max_steps=args.max_steps, quiet=True)
        total += result["cost_usd"]
        results.append(result)
        print(f"apple@{xy}  {'PICKED UP' if result['success'] else 'failed   '}  "
              f"steps={result['steps']:3d}  ({result['stopped_because']})  "
              f"${result['cost_usd']:.5f}", flush=True)

    wins = sum(r["success"] for r in results)
    print(f"\n{wins}/{len(results)} apples picked up. Total Jev spend ${total:.5f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"scenes": results, "success": wins,
                       "total": len(results), "cost_usd": round(total, 6)}, fh, indent=2)


if __name__ == "__main__":
    main()
