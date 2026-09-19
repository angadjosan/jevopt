"""The robot pick-up decision, re-exposed as a plain `Task`.

The arm's action choice is the optimiser's first real benchmark, but nothing
about it needs a physics engine: the labelled states are already frozen on disk
and the option text is just text. So this module rebuilds the same decision out
of a JSON file and a handful of string literals -- importing it pulls in no
pybullet, no numpy, and none of `jevbot`.

The text constants below are copied verbatim from the robot package (each with
its source noted) rather than imported, precisely to keep that true.
"""

from __future__ import annotations

import json
import os
from importlib import resources

from ..conditions import derive
from ..task import Condition, Task

# --------------------------------------------------------------------- text
# Copied verbatim from jevbot/policy.py (GOAL), plus the question sentence
# (ACTION_INSTRUCTION, which is GOAL plus the question sentence).
GOAL = (
    "A robot arm with a two-finger gripper is working over a table. The task is "
    "to pick up an apple: move the open gripper until it is horizontally over "
    "the apple, lower it to the apple's height, close the fingers on the apple, "
    "then raise it so the apple is held high above the table. The state "
    "describes where the apple is relative to the gripper, as seen by a camera."
)

ACTION_INSTRUCTION = (
    GOAL + " Which single action should the arm take right now to make progress "
    "toward holding the apple in the air?"
)

# The naive first
# draft, describing each option mechanically. This is what the search starts from.
SEED_CRITERIA: dict[str, str] = {
    "move_forward": "Move the gripper 3cm in +x (away from the robot base).",
    "move_back": "Move the gripper 3cm in -x (back toward the robot base).",
    "move_left": "Move the gripper 3cm in +y (to the left from the base's view).",
    "move_right": "Move the gripper 3cm in -y (to the right from the base's view).",
    "descend": "Lower the gripper 2.5cm in -z, toward the table.",
    "ascend": "Raise the gripper 2.5cm in +z, away from the table.",
    "open_gripper": "Open the fingers fully. Releases anything held.",
    "close_gripper": "Close the fingers to grip whatever is between them.",
    "done": "Declare the apple picked up and stop.",
}

# Copied verbatim from jevbot/sim.py ACTIONS -- what hand-tuning produced, and
# the arm the optimiser has to beat.
HANDTUNED_CRITERIA: dict[str, str] = {
    "move_forward": "Move the gripper 3cm forward. Pick this when the apple is "
                    "forward of the gripper and they are not lined up yet.",
    "move_back": "Move the gripper 3cm back. Pick this when the apple is behind "
                 "the gripper and they are not lined up yet.",
    "move_left": "Move the gripper 3cm to the left. Pick this when the apple is "
                 "to the left of the gripper and they are not lined up yet.",
    "move_right": "Move the gripper 3cm to the right. Pick this when the apple "
                  "is to the right of the gripper and they are not lined up yet.",
    "descend": "Lower the gripper 2.5cm toward the table. Pick this when the "
               "fingers are open and the gripper is already horizontally over "
               "the apple but is still too high to grip it.",
    "ascend": "Raise the gripper 2.5cm away from the table. Pick this to lift "
              "the apple once the fingers are gripping it, or to back off if "
              "the gripper is too low.",
    "open_gripper": "Open the fingers. Pick this when the fingers are closed "
                    "but are not gripping anything, so they are ready to try "
                    "again, or to deliberately release an object.",
    "close_gripper": "Close the fingers to grip the apple. Pick this when the "
                     "gripper is horizontally over the apple AND at the right "
                     "height to grip it AND the fingers are still open.",
    "done": "Stop: the task is finished. Pick this only when the fingers are "
            "already gripping the apple AND the gripper is held high above the "
            "table.",
}

DATASET = "jevopt/data/robot_states.json"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------- conditions


def _axis(field: str):
    """The three relative-position fields live one level down in the state."""
    def get(state: dict) -> str:
        value = state.get("where_the_apple_is_relative_to_the_gripper", {})
        return value.get(field, "") if isinstance(value, dict) else ""
    return get


# Of the 21 conditions the robot prototype hand-wrote, 17 are plain
# field=value tests that derive() reproduces exactly. Only these four group two
# values into one fact ("slightly" and "far" are the same direction), which
# equality cannot express -- so only these are passed as extras.
EXTRA_CONDITIONS: list[Condition] = [
    Condition("fb.forward", "the apple is forward of the gripper",
              lambda s: "forward of" in _axis("along_forward_back_axis")(s)),
    Condition("fb.behind", "the apple is behind the gripper",
              lambda s: "behind" in _axis("along_forward_back_axis")(s)),
    Condition("lr.left", "the apple is to the left of the gripper",
              lambda s: "to the left" in _axis("along_left_right_axis")(s)),
    Condition("lr.right", "the apple is to the right of the gripper",
              lambda s: "to the right" in _axis("along_left_right_axis")(s)),
]

# ------------------------------------------------------------------- build


def _read(path: str) -> str:
    """The dataset, whether this is a source checkout or an installed package.

    A relative path is tried as given and against the repo root, so running
    from anywhere in a checkout works; failing both, the file is read out of
    the installed `jevopt` package's data directory.
    """
    for candidate in (path, os.path.join(REPO_ROOT, path)):
        if os.path.exists(candidate):
            with open(candidate) as fh:
                return fh.read()
    return (resources.files("jevopt")
            .joinpath("data").joinpath(os.path.basename(path)).read_text())


def load_instances(path: str = DATASET) -> list[dict]:
    """The three frozen splits, concatenated back into one pool.

    Task.split() re-splits by condition-signature, so keeping the old train/
    val/test boundaries would only constrain it for no benefit -- and the old
    boundaries were drawn per instance, which leaks across identical states.
    """
    data = json.loads(_read(path))
    return [{"state": row["state"], "acceptable": list(row["acceptable"]),
             "source": row.get("source", "")}
            for split in ("train", "val", "test") for row in data[split]]


def build(path: str = DATASET) -> Task:
    instances = load_instances(path)
    return Task(
        name="robot",
        instructions=ACTION_INSTRUCTION,
        options=dict(SEED_CRITERIA),
        instances=instances,
        conditions=derive(instances, extra=EXTRA_CONDITIONS),
        reference=dict(HANDTUNED_CRITERIA),
    )


def main() -> None:
    task = build()
    extra = len(EXTRA_CONDITIONS)
    print(f"instances:  {len(task.instances)}")
    print(f"options:    {len(task.options)}")
    print(f"conditions: {len(task.conditions)} "
          f"({len(task.conditions) - extra} derived + {extra} extra)")

    parts = task.split()
    print("split:      " + " / ".join(str(len(part)) for part in parts)
          + "  (train / val / test)")

    # A signature in two splits is a leak: the same distinguishable state would
    # be both trained on and tested on.
    seen: dict[tuple, int] = {}
    for index, part in enumerate(parts):
        for instance in part:
            signature = task.signature(instance["state"])
            assert seen.setdefault(signature, index) == index, "signature spans splits"
    print(f"signatures: {len(seen)} distinct, none shared between splits")

    for instance in task.instances:
        assert instance["acceptable"], "instance with no acceptable option"
        assert set(instance["acceptable"]) <= set(task.options), "unknown option"
    print("labels:     all non-empty and within the option set")


if __name__ == "__main__":
    main()
