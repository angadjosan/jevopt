"""A frozen dataset of control states, labelled with the actions that are OK.

Running physics inside the optimizer loop would make the search unbearably slow,
so the states are harvested once from oracle rollouts (plus edge cases the oracle
rarely reaches), deduplicated, and reused. Every candidate prompt is then scored
by single-step agreement -- pure Jev calls, parallelisable, one score per
instance, which is exactly the shape GEPA's Pareto front wants.
"""

from __future__ import annotations

import json
import random

from .perception import describe, observe
from .sim import ACTIONS, TABLE_TOP, ArmWorld

HARVEST_POSITIONS = [
    (0.40, 0.10), (0.30, -0.15), (0.35, 0.22), (0.25, 0.00), (0.20, -0.25),
    (0.42, -0.05), (0.33, 0.05), (0.28, 0.18), (0.45, 0.15), (0.22, -0.10),
    (0.38, -0.22), (0.30, 0.00),
]


def acceptable(state: dict) -> list[str]:
    """The actions that make non-wasteful progress from this state.

    Written from the state alone, in code -- this is the label, not a policy the
    model sees. Several actions can be right at once (two axes both misaligned),
    so the label is a set.
    """
    if state.get("fingers_are_gripping_an_object") == "yes":
        return ["done"] if state.get("gripper_is_held_high_above_the_table") == "yes" \
            else ["ascend"]
    if state.get("fingers") == "closed":
        return ["open_gripper"]
    if state.get("apple_seen_by_camera") != "yes":
        return ["ascend"]

    where = state["where_the_apple_is_relative_to_the_gripper"]
    if state.get("gripper_is_horizontally_over_the_apple") == "yes":
        if state.get("gripper_is_at_the_right_height_to_grip") == "yes":
            return ["close_gripper"]
        return ["ascend"] if "above" in where["height"] else ["descend"]

    moves = []
    fb, lr = where["along_forward_back_axis"], where["along_left_right_axis"]
    if "forward of" in fb:
        moves.append("move_forward")
    elif "behind" in fb:
        moves.append("move_back")
    if "to the left" in lr:
        moves.append("move_left")
    elif "to the right" in lr:
        moves.append("move_right")
    return moves or ["descend"]


def _greedy(state: dict) -> str:
    """Oracle used only to walk the arm around while harvesting states."""
    return acceptable(state)[0]


def _key(state: dict) -> tuple:
    """Bucket identical situations so the set does not fill with duplicates."""
    where = state.get("where_the_apple_is_relative_to_the_gripper", {})
    return (
        state.get("fingers"), state.get("fingers_are_gripping_an_object"),
        state.get("apple_seen_by_camera"),
        state.get("gripper_is_horizontally_over_the_apple"),
        state.get("gripper_is_at_the_right_height_to_grip"),
        state.get("gripper_is_held_high_above_the_table"),
        where.get("along_forward_back_axis"), where.get("along_left_right_axis"),
        where.get("height"),
    )


def edge_cases() -> list[dict]:
    """Situations oracle rollouts almost never produce, written by hand."""
    def base(**over):
        state = {
            "step": 7, "fingers": "open", "fingers_are_gripping_an_object": "no",
            "last_action": "descend", "last_action_result": "moved",
            "apple_seen_by_camera": "yes",
            "where_the_apple_is_relative_to_the_gripper": {
                "along_forward_back_axis": "lined up",
                "along_left_right_axis": "lined up",
                "height": "below the gripper"},
            "gripper_is_horizontally_over_the_apple": "yes",
            "gripper_is_at_the_right_height_to_grip": "no",
            "gripper_is_held_high_above_the_table": "no",
        }
        where = over.pop("where", None)
        if where:
            state["where_the_apple_is_relative_to_the_gripper"].update(where)
        state.update(over)
        return state

    return [
        base(fingers="closed", last_action="close_gripper",
             last_action_result="fingers closed"),
        base(fingers="closed", fingers_are_gripping_an_object="yes",
             gripper_is_at_the_right_height_to_grip="yes",
             where={"height": "level with the gripper"},
             last_action="close_gripper", last_action_result="fingers closed"),
        base(fingers="closed", fingers_are_gripping_an_object="yes",
             gripper_is_held_high_above_the_table="yes",
             where={"height": "level with the gripper"},
             last_action="ascend", last_action_result="moved"),
        base(apple_seen_by_camera="no",
             where={"along_forward_back_axis": "lined up",
                    "along_left_right_axis": "lined up", "height": "below the gripper"}),
        base(where={"height": "above the gripper"},
             last_action="descend", last_action_result="moved"),
        base(where={"along_forward_back_axis": "slightly behind the gripper",
                    "along_left_right_axis": "slightly to the right of the gripper"},
             gripper_is_horizontally_over_the_apple="no"),
        base(where={"along_forward_back_axis": "far behind the gripper",
                    "along_left_right_axis": "far to the left of the gripper"},
             gripper_is_horizontally_over_the_apple="no",
             last_action="move_forward",
             last_action_result="the arm could not move there -- it is at its reach limit"),
        base(where={"along_forward_back_axis": "slightly forward of the gripper",
                    "along_left_right_axis": "lined up",
                    "height": "level with the gripper"},
             gripper_is_horizontally_over_the_apple="no",
             gripper_is_at_the_right_height_to_grip="yes"),
    ]


FB = ["lined up", "slightly forward of the gripper", "far forward of the gripper",
      "slightly behind the gripper", "far behind the gripper"]
LR = ["lined up", "slightly to the left of the gripper", "far to the left of the gripper",
      "slightly to the right of the gripper", "far to the right of the gripper"]
HEIGHT = ["below the gripper", "level with the gripper", "above the gripper"]
GRIP = [("open", "no"), ("closed", "no"), ("closed", "yes")]


def grid(per_label: int = 26, seed: int = 0) -> list[dict]:
    """Every distinguishable situation, sampled evenly across the labels.

    Rollouts alone over-represent whatever the start pose happens to produce
    (21 of 52 harvested states wanted move_forward). A balanced grid keeps the
    metric from being dominated by one action.
    """
    rng = random.Random(seed)
    pool: dict[str, list[dict]] = {}
    for fb in FB:
        for lr in LR:
            for height in HEIGHT:
                for fingers, gripping in GRIP:
                    for high in ("yes", "no"):
                        over = "yes" if (fb == "lined up" and lr == "lined up") else "no"
                        state = {
                            "step": rng.randrange(2, 30),
                            "fingers": fingers,
                            "fingers_are_gripping_an_object": gripping,
                            "last_action": rng.choice(list(ACTIONS)),
                            "last_action_result": "moved",
                            "apple_seen_by_camera": "yes",
                            "where_the_apple_is_relative_to_the_gripper": {
                                "along_forward_back_axis": fb,
                                "along_left_right_axis": lr,
                                "height": height,
                            },
                            "gripper_is_horizontally_over_the_apple": over,
                            "gripper_is_at_the_right_height_to_grip":
                                "yes" if height == "level with the gripper" else "no",
                            "gripper_is_held_high_above_the_table": high,
                        }
                        good = acceptable(state)
                        pool.setdefault("+".join(good), []).append(
                            {"state": state, "acceptable": good, "source": "grid"})

    out = []
    for label, items in sorted(pool.items()):
        rng.shuffle(items)
        out.extend(items[:per_label])
    return out


def harvest(per_bucket: int = 2, seed: int = 0) -> list[dict]:
    """Walk the oracle through every placement, keeping a few of each situation."""
    seen: dict[tuple, int] = {}
    instances: list[dict] = []

    for xy in HARVEST_POSITIONS:
        world = ArmWorld(apple_xy=xy)
        last = (None, None)
        for step in range(60):
            rgb, depth, view, proj, _ = world.camera()
            state = describe(observe(rgb, depth, view, proj), world.ee_pos(),
                             world.finger_gap(), world.fingers_commanded(),
                             world.object_between_fingers(), step, last[0], last[1],
                             TABLE_TOP)
            good = acceptable(state)
            key = _key(state)
            if seen.get(key, 0) < per_bucket:
                seen[key] = seen.get(key, 0) + 1
                instances.append({"state": state, "acceptable": good,
                                  "source": f"rollout{xy}"})
            action = _greedy(state)
            if action == "done":
                break
            last = (action, world.apply(action))
        world.close()

    for state in edge_cases():
        instances.append({"state": state, "acceptable": acceptable(state),
                          "source": "edge_case"})
    instances.extend(grid(seed=seed))

    for inst in instances:
        assert all(a in ACTIONS for a in inst["acceptable"]), inst["acceptable"]
    random.Random(seed).shuffle(instances)
    return instances


def split(instances: list[dict], fractions=(0.4, 0.3, 0.3), seed: int = 0):
    """Split by situation, not by instance.

    The state space is finite and the grid enumerates it, so several instances
    can describe the same situation. Every grammar condition and the label
    function read only the fields in `_key`, which makes two such instances the
    *same* training example -- splitting them apart leaks train into test.
    Splitting whole `_key` groups keeps the test set genuinely unseen.
    """
    groups: dict[tuple, list[dict]] = {}
    for instance in instances:
        groups.setdefault(_key(instance["state"]), []).append(instance)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    a = int(len(keys) * fractions[0])
    b = a + int(len(keys) * fractions[1])
    return tuple([i for key in part for i in groups[key]]
                 for part in (keys[:a], keys[a:b], keys[b:]))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="jevopt/data/robot_states.json")
    parser.add_argument("--per-bucket", type=int, default=3)
    args = parser.parse_args()

    instances = harvest(args.per_bucket)
    train, val, test = split(instances)
    with open(args.out, "w") as fh:
        json.dump({"train": train, "val": val, "test": test}, fh, indent=1)

    from collections import Counter
    counts = Counter("+".join(i["acceptable"]) for i in instances)
    print(f"{len(instances)} instances -> train {len(train)} / val {len(val)} "
          f"/ test {len(test)}   written to {args.out}")
    print("label mix:")
    for label, n in counts.most_common():
        print(f"   {n:3d}  {label}")


if __name__ == "__main__":
    main()
