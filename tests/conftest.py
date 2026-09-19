"""Shared fixtures: one tiny synthetic Task, plus the two real ones.

Almost every property under test is a property of *any* Task, so most tests run
against a hand-sized synthetic task whose labels are three one-line rules -- it
is exhaustively checkable and its expected counts can be written down. The real
triage task is then sampled, to catch anything that only shows up in real text
and a real label distribution. Both are session-scoped: building them is cheap
but not free, and nothing here mutates a Task.
"""

from __future__ import annotations

import itertools

import pytest

from jevopt.conditions import derive
from jevopt.evidence import Evidence
from jevopt.task import Task
from jevopt.tasks import triage

COLORS = ("red", "blue")
SIZES = ("small", "large")
ZONES = ("north", "south")
REPLICAS = 3

SYNTHETIC_OPTIONS = {
    "alpha": "Hand the job to the alpha worker.",
    "beta": "Hand the job to the beta worker. Costs twice as much.",
    "gamma": "Hand the job to the gamma worker.",
    "delta": "Hand the job to the delta worker.",
}


def synthetic_acceptable(state: dict) -> list[str]:
    """Deliberately simple rules, so the evidence gates have a known answer.

    alpha and gamma partition the colours, so no state is left unlabelled;
    delta is rare on purpose, to keep an under-evidenced option in the task.
    """
    good = []
    if state["color"] == "red":
        good.append("alpha")
    if state["size"] == "large":
        good.append("beta")
    if state["color"] == "blue":
        good.append("gamma")
    if (state["color"] == "red" and state["size"] == "large"
            and state["position"]["zone"] == "north"):
        good.append("delta")
    return good


def synthetic_instances() -> list[dict]:
    instances = []
    for index, (_rep, color, size, zone) in enumerate(
            itertools.product(range(REPLICAS), COLORS, SIZES, ZONES)):
        state = {
            "color": color,
            "size": size,
            "position": {"zone": zone},          # nested, so flatten() is exercised
            "env": "staging",                    # constant: no condition may read it
            "trace_id": f"t{index:03d}",         # unique: too wide to enumerate
        }
        instances.append({"state": state, "acceptable": synthetic_acceptable(state)})
    return instances


@pytest.fixture(scope="session")
def synthetic_task() -> Task:
    instances = synthetic_instances()
    return Task(
        name="synthetic",
        instructions="A job has arrived. Which worker should take it?",
        options=dict(SYNTHETIC_OPTIONS),
        instances=instances,
        conditions=derive(instances),
    )


@pytest.fixture(scope="session")
def synthetic_evidence(synthetic_task: Task) -> Evidence:
    """Evidence over the whole synthetic set, so positive counts are predictable."""
    return Evidence(synthetic_task, synthetic_task.instances)


@pytest.fixture(scope="session")
def triage_task() -> Task:
    return triage.build()


@pytest.fixture(scope="session", params=["triage"])
def real_task(request) -> Task:
    return request.getfixturevalue(f"{request.param}_task")


@pytest.fixture(scope="session", params=["synthetic", "triage"])
def any_task(request) -> Task:
    return request.getfixturevalue(f"{request.param}_task")
