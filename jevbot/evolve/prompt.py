"""The thing being optimised: the text of each action's criterion.

A GEPA candidate is `{"criteria.move_left": "<text>", ...}` -- one component per
action. The shared goal and the question wording are held fixed across every arm
so the experiment isolates one variable: how each option is described.

Because Jev cannot write text, new wording has to come from somewhere other than
a generative model. It comes from a *grammar*: conditions are read mechanically
off the state schema, and clause templates turn them into English. Nothing in
here encodes which action is right -- only the vocabulary for saying when an
action might apply. Finding the right binding is the search problem.
"""

from __future__ import annotations

from ..policy import GOAL
from ..sim import ACTIONS

ACTION_INSTRUCTION = (
    GOAL + " Which single action should the arm take right now to make progress "
    "toward holding the apple in the air?"
)

# The naive first draft: each option describes what it does mechanically.
SEED_CRITERIA = {
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

# What hand-tuning produced, for comparison.
HANDTUNED_CRITERIA = dict(ACTIONS)

PREFIX = "criteria."


def components(criteria: dict[str, str]) -> dict[str, str]:
    return {PREFIX + name: text for name, text in criteria.items()}


def criteria_of(candidate: dict[str, str]) -> dict[str, str]:
    """Candidate -> the criteria map Jev expects, in the canonical action order."""
    return {name: candidate.get(PREFIX + name, SEED_CRITERIA[name]) for name in ACTIONS}


def questions(candidate: dict[str, str], with_nouls: bool = False) -> dict:
    """The action question. Optimisation needs only this one; driving the arm
    also wants the two nouls the control loop logs."""
    built = {
        "action": {
            "type": "choice",
            "instructions": ACTION_INSTRUCTION,
            "criteria": criteria_of(candidate),
        }
    }
    if with_nouls:
        from ..policy import QUESTIONS
        for name in ("centred", "holding"):
            built[name] = QUESTIONS[name]
    return built


# --------------------------------------------------------------- clause grammar

def _axis(field: str):
    def get(state: dict) -> str:
        return state.get("where_the_apple_is_relative_to_the_gripper", {}).get(field, "")
    return get


def _top(field: str):
    def get(state: dict) -> str:
        return state.get(field, "")
    return get


# (id, english phrase, predicate over a state)
CONDITIONS: list[tuple[str, str, object]] = [
    ("fb.forward", "the apple is forward of the gripper",
     lambda s: "forward of" in _axis("along_forward_back_axis")(s)),
    ("fb.behind", "the apple is behind the gripper",
     lambda s: "behind" in _axis("along_forward_back_axis")(s)),
    ("fb.lined", "the apple is lined up on the forward-back axis",
     lambda s: "lined up" == _axis("along_forward_back_axis")(s)),
    ("lr.left", "the apple is to the left of the gripper",
     lambda s: "to the left" in _axis("along_left_right_axis")(s)),
    ("lr.right", "the apple is to the right of the gripper",
     lambda s: "to the right" in _axis("along_left_right_axis")(s)),
    ("lr.lined", "the apple is lined up on the left-right axis",
     lambda s: "lined up" == _axis("along_left_right_axis")(s)),
    ("h.below", "the apple is below the gripper",
     lambda s: "below" in _axis("height")(s)),
    ("h.above", "the apple is above the gripper",
     lambda s: "above" in _axis("height")(s)),
    ("h.level", "the apple is level with the gripper",
     lambda s: "level" in _axis("height")(s)),
    ("over.yes", "the gripper is horizontally over the apple",
     lambda s: _top("gripper_is_horizontally_over_the_apple")(s) == "yes"),
    ("over.no", "the gripper is not yet horizontally over the apple",
     lambda s: _top("gripper_is_horizontally_over_the_apple")(s) == "no"),
    ("height.yes", "the gripper is at the right height to grip",
     lambda s: _top("gripper_is_at_the_right_height_to_grip")(s) == "yes"),
    ("height.no", "the gripper is not at the right height to grip",
     lambda s: _top("gripper_is_at_the_right_height_to_grip")(s) == "no"),
    ("high.yes", "the gripper is held high above the table",
     lambda s: _top("gripper_is_held_high_above_the_table")(s) == "yes"),
    ("high.no", "the gripper is not high above the table",
     lambda s: _top("gripper_is_held_high_above_the_table")(s) == "no"),
    ("fingers.open", "the fingers are open",
     lambda s: _top("fingers")(s) == "open"),
    ("fingers.closed", "the fingers are closed",
     lambda s: _top("fingers")(s) == "closed"),
    ("hold.yes", "the fingers are gripping an object",
     lambda s: _top("fingers_are_gripping_an_object")(s) == "yes"),
    ("hold.no", "the fingers are not gripping anything",
     lambda s: _top("fingers_are_gripping_an_object")(s) == "no"),
    ("seen.yes", "the camera can see the apple",
     lambda s: _top("apple_seen_by_camera")(s) == "yes"),
    ("seen.no", "the camera cannot see the apple",
     lambda s: _top("apple_seen_by_camera")(s) == "no"),
]

CONDITION_BY_ID = {c[0]: c for c in CONDITIONS}

TEMPLATES = {
    "only": "Only choose this when {phrase}.",
    "never": "Never choose this when {phrase}.",
    "prefer": "When {phrase}, choose {other} instead of this.",
}


def render_clause(template: str, condition_id: str, other: str | None = None) -> str:
    phrase = CONDITION_BY_ID[condition_id][1]
    return TEMPLATES[template].format(phrase=phrase, other=other or "")


def holds(condition_id: str, state: dict) -> bool:
    return bool(CONDITION_BY_ID[condition_id][2](state))


def apply_clause(text: str, clause: str) -> str:
    """Append a clause, keeping the component readable and duplicate-free."""
    text = text.rstrip()
    if clause in text:
        return text
    return f"{text} {clause}"


def render_prompt(candidate: dict[str, str]) -> str:
    """The evolved prompt as readable text, for diffing and for the repo."""
    lines = [ACTION_INSTRUCTION, ""]
    for name, text in criteria_of(candidate).items():
        lines.append(f"{name}: {text}")
    return "\n".join(lines)


def all_clauses() -> set[str]:
    """Every string the grammar can produce -- lets us find added clauses in a
    component's text, so they can be removed again as well as appended."""
    out = set()
    for cid, _phrase, _test in CONDITIONS:
        out.add(render_clause("only", cid))
        out.add(render_clause("never", cid))
        for other in ACTIONS:
            out.add(render_clause("prefer", cid, other))
    return out


_ALL = None


def split_clauses(text: str) -> tuple[str, list[str]]:
    """Separate the seed description from clauses the search has appended."""
    global _ALL
    if _ALL is None:
        _ALL = all_clauses()
    parts, base, clauses = [p.strip() for p in text.split(". ")], [], []
    for i, part in enumerate(parts):
        sentence = part if part.endswith(".") else part + "."
        (clauses if sentence in _ALL else base).append(sentence)
    return " ".join(base), clauses


def compose(base: str, clauses: list[str]) -> str:
    return " ".join([base] + clauses).strip()
