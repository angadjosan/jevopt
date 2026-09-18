"""Why the state is worded instead of numeric.

The first version of this project handed Jev the raw geometry -- dx/dy/dz in
metres -- and asked it to pick an action. Jev's own docs warn against exactly
that: it is "not a calculator", "struggles with tasks that require numeric
precision", and does better on "semantic representations than numeric"
(https://docs.typesafe.ai/model-jaggedness/jev-1.13).

This runs the same ten situations through both framings so the difference is
measurable rather than asserted.
"""

from __future__ import annotations

from .policy import QUESTIONS, decide
from .sim import ACTIONS

# (name, numeric state, worded state, expected action)
CASES = [
    ("apple far forward", (0.25, 0.0, -0.15), ("far forward of the gripper", "lined up", "below the gripper", "no", "no"), "move_forward"),
    ("apple far left", (0.0, 0.20, -0.15), ("lined up", "far to the left of the gripper", "below the gripper", "no", "no"), "move_left"),
    ("apple far right", (0.0, -0.20, -0.15), ("lined up", "far to the right of the gripper", "below the gripper", "no", "no"), "move_right"),
    ("apple far behind", (-0.18, 0.0, -0.15), ("far behind the gripper", "lined up", "below the gripper", "no", "no"), "move_back"),
    ("apple slightly left", (0.0, 0.04, -0.15), ("lined up", "slightly to the left of the gripper", "below the gripper", "no", "no"), "move_left"),
    ("over it, too high", (0.0, 0.0, -0.15), ("lined up", "lined up", "below the gripper", "yes", "no"), "descend"),
    ("over it, at height", (0.0, 0.0, -0.005), ("lined up", "lined up", "level with the gripper", "yes", "yes"), "close_gripper"),
]


def numeric_state(dx, dy, dz, fingers="open", held=False, ee_z=0.806):
    return {
        "step": 5, "gripper_position": [0.15, 0.0, ee_z], "gripper_fingers": fingers,
        "gripper_finger_gap_m": 0.08 if fingers == "open" else 0.049,
        "object_between_fingers": held,
        "apple_relative_to_gripper": {
            "dx_forward_m": dx, "dy_left_m": dy, "dz_up_m": dz,
            "horizontal_distance_m": round((dx * dx + dy * dy) ** 0.5, 3),
        },
    }


def worded_state(fb, lr, height, over, height_ok, fingers="open", held="no"):
    return {
        "step": 5, "fingers": fingers, "fingers_are_gripping_an_object": held,
        "apple_seen_by_camera": "yes",
        "where_the_apple_is_relative_to_the_gripper": {
            "along_forward_back_axis": fb, "along_left_right_axis": lr,
            "height": height,
        },
        "gripper_is_horizontally_over_the_apple": over,
        "gripper_is_at_the_right_height_to_grip": height_ok,
        "gripper_is_held_high_above_the_table": "no",
    }


NUMERIC_QUESTIONS = {
    **QUESTIONS,
    "action": {
        "type": "choice",
        "criteria": dict(ACTIONS),
        "instructions": (
            "A robot arm with a two-finger gripper must pick up an apple. "
            "'dx_forward_m' is how far forward the apple is from the gripper, "
            "'dy_left_m' how far to the left, 'dz_up_m' how far up (negative "
            "means below). All in metres. Which single action should it take now?"
        ),
    },
}


def main() -> None:
    from . import client
    rows = []
    for name, nums, words, want in CASES:
        body = client.ask(numeric_state(*nums), NUMERIC_QUESTIONS)
        num = body["answers"]["action"]
        wrd = decide(worded_state(*words))
        rows.append((name, want, num["choice"], num["confidence"],
                     wrd["action"], wrd["confidence"]))

    print(f"{'situation':22s} {'expected':14s} | {'numeric':14s} conf  | "
          f"{'worded':14s} conf")
    print("-" * 82)
    n_ok = w_ok = 0
    for name, want, na, nc, wa, wc in rows:
        n_ok += na == want
        w_ok += wa == want
        print(f"{name:22s} {want:14s} | {na:14s} {nc:.2f} {'ok' if na==want else '  '} | "
              f"{wa:14s} {wc:.2f} {'ok' if wa==want else ''}")
    total = len(rows)
    print("-" * 82)
    print(f"numeric state: {n_ok}/{total} correct, mean confidence "
          f"{sum(r[3] for r in rows)/total:.2f}")
    print(f"worded state:  {w_ok}/{total} correct, mean confidence "
          f"{sum(r[5] for r in rows)/total:.2f}")


if __name__ == "__main__":
    main()
