"""Run one episode: Jev picks every action until the apple is up, or time runs out."""

from __future__ import annotations

import argparse
import json
import os

from .perception import describe, observe
from .policy import decide
from .sim import TABLE_TOP, ArmWorld


def snapshot(world: ArmWorld, step: int, last: tuple) -> dict:
    rgb, depth, view, proj, _ = world.camera()
    state = describe(observe(rgb, depth, view, proj), world.ee_pos(),
                     world.finger_gap(), world.fingers_commanded(),
                     world.object_between_fingers(), step, last[0], last[1],
                     TABLE_TOP)
    return state, rgb


def episode(apple_xy, max_steps=45, model=None, frame_dir=None, quiet=False) -> dict:
    world = ArmWorld(apple_xy=apple_xy)
    frames, log = [], []
    last = (None, None)
    spend = 0.0
    stopped = "ran out of steps"

    for step in range(max_steps):
        state, rgb = snapshot(world, step, last)
        frames.append(rgb)
        choice = decide(state, model=model) if model else decide(state)
        spend += choice["usage"].get("cost", 0.0) or 0.0
        action = choice["action"]

        if not quiet:
            where = state.get("where_the_apple_is_relative_to_the_gripper", {})
            print(f"{step:3d}  {action:<14s} conf {choice['confidence']:.2f}  "
                  f"held {choice['holding']:.2f}  | "
                  f"{where.get('along_forward_back_axis','?')}, "
                  f"{where.get('along_left_right_axis','?')}, "
                  f"{where.get('height','?')}")

        log.append({"step": step, "state": state, "action": action,
                    "confidence": choice["confidence"],
                    "probabilities": choice["probabilities"],
                    "centred": choice["centred"], "holding": choice["holding"]})

        if action == "done":
            stopped = "Jev called done"
            break
        last = (action, world.apply(action))
    else:
        step = max_steps - 1

    state, rgb = snapshot(world, step + 1, last)
    frames.append(rgb)
    success = world.apple_lifted()
    result = {
        "apple_xy": list(apple_xy), "steps": step + 1, "success": bool(success),
        "stopped_because": stopped, "cost_usd": round(spend, 6),
        "apple_height_m": round(world.apple_pos()[2], 4),
        "table_top_m": TABLE_TOP, "log": log,
    }

    if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
        from PIL import Image
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(frame_dir, f"f{i:03d}.png"))
    world.close()
    return result, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apple", default="0.40,0.10",
                        help="apple x,y on the table (default 0.40,0.10)")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--model", default=None)
    parser.add_argument("--frames", metavar="DIR", help="save every frame as PNG")
    parser.add_argument("--json", metavar="FILE", help="write the full log as JSON")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    x, y = (float(v) for v in args.apple.split(","))
    result, _ = episode((x, y), args.max_steps, args.model, args.frames, args.quiet)

    print(f"\n{'PICKED UP THE APPLE' if result['success'] else 'FAILED'} "
          f"after {result['steps']} steps ({result['stopped_because']}); "
          f"apple at z={result['apple_height_m']} (table {TABLE_TOP}); "
          f"Jev cost ${result['cost_usd']:.5f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"log written to {args.json}")


if __name__ == "__main__":
    main()
