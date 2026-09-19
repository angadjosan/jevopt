"""What the optimiser needs to know about a decision, and nothing more.

The method is not about robots. It evolves the option descriptions of any Jev
`choice` question, given labelled states. A Task is the whole interface: the
options to choose between, the states, which options are acceptable in each,
and the vocabulary of conditions the grammar may talk about.

Nothing in here imports a simulator, a domain, or a model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Condition:
    """A fact about a state that a clause can be built from."""
    id: str
    phrase: str                       # slots into "Only choose this when {phrase}."
    test: Callable[[dict], bool]

    def holds(self, state: dict) -> bool:
        try:
            return bool(self.test(state))
        except Exception:             # a state missing the field simply fails it
            return False


@dataclass
class Task:
    name: str
    instructions: str                 # the choice question's instruction text
    options: dict[str, str]           # option id -> naive seed description
    instances: list[dict]             # [{"state": {...}, "acceptable": [ids]}]
    conditions: list[Condition] = field(default_factory=list)
    # Optional human-written option descriptions, as a reference arm to beat.
    reference: dict[str, str] | None = None

    def __post_init__(self) -> None:
        for instance in self.instances:
            bad = set(instance["acceptable"]) - set(self.options)
            if bad:
                raise ValueError(f"{self.name}: unknown option(s) {sorted(bad)}")
            if not instance["acceptable"]:
                raise ValueError(f"{self.name}: instance with no acceptable option")

    # --------------------------------------------------------------- splitting

    def signature(self, state: dict) -> tuple:
        """What the optimiser can actually distinguish about a state.

        Two states on which every condition agrees are the same training
        example, whatever else differs in their JSON. Splitting such states
        across train and test leaks, so the split works on this signature
        rather than on instances.
        """
        return tuple(condition.holds(state) for condition in self.conditions)

    def split(self, fractions=(0.4, 0.3, 0.3), seed: int = 0):
        groups: dict[tuple, list[dict]] = {}
        for instance in self.instances:
            groups.setdefault(self.signature(instance["state"]), []).append(instance)
        keys = list(groups)
        random.Random(seed).shuffle(keys)
        a = int(len(keys) * fractions[0])
        b = a + int(len(keys) * fractions[1])
        return tuple([i for key in part for i in groups[key]]
                     for part in (keys[:a], keys[a:b], keys[b:]))

    # ------------------------------------------------------------------- io

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"name": self.name, "instructions": self.instructions,
                       "options": self.options, "instances": self.instances},
                      fh, indent=1)
