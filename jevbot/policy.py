"""Jev as the robot's policy.

Every tool call the arm exposes is one option in a Jev `choice` question, so
picking an action *is* a single structured decision. Two `noul` questions ride
along in the same request -- batching is both cheaper and faster than asking
separately (https://docs.typesafe.ai/patterns/fan-out) -- and give the loop a
second axis to log and gate on.
"""

from __future__ import annotations

from . import client
from .sim import ACTIONS

GOAL = (
    "A robot arm with a two-finger gripper is working over a table. The task is "
    "to pick up an apple: move the open gripper until it is horizontally over "
    "the apple, lower it to the apple's height, close the fingers on the apple, "
    "then raise it so the apple is held high above the table. The state "
    "describes where the apple is relative to the gripper, as seen by a camera."
)

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": (
            GOAL + " Which single action should the arm take right now to make "
            "progress toward holding the apple in the air?"
        ),
        "criteria": dict(ACTIONS),
    },
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


def decide(state: dict, model: str = client.MODEL) -> dict:
    """One Jev call -> the chosen action plus the two side judgements."""
    body = client.ask(state, QUESTIONS, model=model)
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
