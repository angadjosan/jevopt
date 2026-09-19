"""Jev as the robot's policy.

Every tool call the arm exposes is one option in a Jev `choice` question, so
picking an action *is* a single structured decision. Two `noul` questions ride
along in the same request -- batching is both cheaper and faster than asking
separately (https://docs.typesafe.ai/patterns/fan-out) -- and give the loop a
second axis to log and gate on.
"""

from __future__ import annotations

from jevopt import client

from .sim import ACTIONS

GOAL = (
    "A robot arm with a two-finger gripper is working over a table. The task is "
    "to pick up an apple: move the open gripper until it is horizontally over "
    "the apple, lower it to the apple's height, close the fingers on the apple, "
    "then raise it so the apple is held high above the table. The state "
    "describes where the apple is relative to the gripper, as seen by a camera."
)

ACTION_QUESTION = {
    "type": "choice",
    "instructions": (
        GOAL + " Which single action should the arm take right now to make "
        "progress toward holding the apple in the air?"
    ),
    "criteria": dict(ACTIONS),
}

# The side questions. Kept separate so an alternative action question -- the
# optimised one in validate.py, the numeric one in ablation.py -- can be dropped
# in front of the same two nouls.
NOULS = {
    "centred": {
        "type": "noul",
        "instructions": (
            GOAL + " Is the gripper lined up over the apple, so that lowering it "
            "straight down would bring the fingers around the apple?"
        ),
        "criteria": {
            "true": "The state says the gripper is horizontally over the apple",
            "false": "The gripper still has to move sideways first",
        },
    },
    "holding": {
        "type": "noul",
        "instructions": (
            GOAL + " Is the gripper currently holding the apple between its "
            "fingers?"
        ),
        "criteria": {
            "true": "The fingers are closed on the apple and it would lift with them",
            "false": "The fingers are empty",
        },
    },
}

QUESTIONS = {"action": ACTION_QUESTION, **NOULS}


def decide(state: dict, model: str | None = None, questions: dict | None = None) -> dict:
    """One Jev call -> the chosen action plus the two side judgements.

    `questions` overrides the request wholesale, for the arms that swap the
    action question out; it must still answer under "action", "centred" and
    "holding".
    """
    body = client.ask(state, questions or QUESTIONS, model=model or client.MODEL)
    answers = body["answers"]
    action = answers["action"]
    return {
        "action": action["choice"],
        "confidence": action.get("confidence"),
        "probabilities": action.get("probabilities", {}),
        "centred": answers["centred"]["noul"],
        "holding": answers["holding"]["noul"],
        "usage": body.get("usage", {}),
        "model": body.get("model"),
    }
