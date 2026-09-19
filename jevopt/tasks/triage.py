"""Production alert triage -- the optimiser's second, deliberately non-robotic task.

The prompt optimiser in `jevopt/` was extracted from a robot-arm demo, and a
method that only ever ran on one domain proves nothing about the method. So this
task has no geometry, no simulator and no physics: an on-call engineer reads a
structured alert record and decides what to do about it.

The state is a small JSON incident record of seven low-cardinality categorical
fields, which `conditions.derive` enumerates into a grammar for free. The labels
come from `acceptable()` -- a deterministic triage policy written in code and
never shown to the model. It deliberately needs several fields at once (a
duplicate is suppressed even at sev1; a fresh deploy behind customer impact is
rolled back; automation is safe only with a runbook *and* a small blast radius),
so a naive "what the button does" prompt cannot do well where a well-specified
one can. `reference` is the human arm: what a careful engineer would write.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

from ..conditions import derive
from ..task import Condition, Task

SEVERITY = ("sev1", "sev2", "sev3")
SERVICE_TIER = ("tier1", "tier2", "tier3")
ERROR_BUDGET = ("exhausted", "burning", "healthy")
YES_NO = ("yes", "no")

FIELDS = ("severity", "service_tier", "error_budget", "recent_deploy",
          "duplicate_of_open_incident", "business_hours", "known_runbook")

INSTRUCTIONS = (
    "You are the first responder on a production on-call rota. An alert has "
    "fired and the record above is everything the alerting system knows: how "
    "severe the symptom is, how important the affected service is, how much "
    "error budget it has left, whether anything shipped to it recently, whether "
    "an incident someone is already working covers this alert, whether it is the "
    "working day, and whether a written runbook exists. Every response costs "
    "something -- waking a person, a queue of tickets nobody reads, an automated "
    "action taken blind, a needless rollback, or a real problem going unseen. "
    "Which single response should be taken for this alert right now?"
)

# The naive first draft: each option says only what it mechanically does.
OPTIONS = {
    "page_oncall": "Send a phone page to the on-call engineer.",
    "create_ticket": "File a ticket in the team's backlog queue.",
    "auto_remediate": "Run the alert's automated remediation script.",
    "rollback_deploy": "Revert the service to the previously deployed build.",
    "suppress": "Silence the alert and record no follow-up work.",
}

# The human arm: what a careful engineer writes once they have thought about it.
REFERENCE = {
    "page_oncall":
        "Wake the on-call engineer. Choose this when the alert is genuinely new "
        "(duplicate_of_open_incident is \"no\") and a human must look now: any "
        "sev1, a sev2 on a tier1 service, or a sev2 whose error budget is "
        "already exhausted. Never for a duplicate -- someone is awake for that "
        "already -- and never for a sev3, whatever the hour.",
    "create_ticket":
        "Queue the work for the next working day. Choose this when the alert is "
        "new but nothing about it justifies waking anyone, reverting a build or "
        "running a script: typically a sev3, or a sev2 on a tier2/tier3 service "
        "with budget left. Also reasonable alongside a safe automated fix during "
        "business_hours, so someone reviews the cause. Never for a duplicate.",
    "auto_remediate":
        "Run the scripted fix. Choose this only when known_runbook is \"yes\" AND "
        "the blast radius is small -- a tier3 service, or a tier2 service with a "
        "sev3 symptom -- and the severity is not sev1: a sev1 needs a human even "
        "where a script exists. A duplicate may still be auto-remediated, but "
        "only under those same two conditions.",
    "rollback_deploy":
        "Revert the last deployment. Choose this when recent_deploy is \"yes\" and "
        "the deploy is the prime suspect for real damage: a sev1, a sev2 on a "
        "tier1 service, or a sev2 with the error budget burning or exhausted. It "
        "stays right even for a duplicate when that is a sev1 on a tier1 "
        "service. Never revert when nothing was deployed.",
    "suppress":
        "Silence this alert and record no follow-up. Choose this whenever "
        "duplicate_of_open_incident is \"yes\" -- the work is already tracked and "
        "a second page or ticket is pure noise, however high the severity. Also "
        "choose it for genuine noise: a sev3 with a healthy error budget outside "
        "business hours and no runbook worth running.",
}


# ------------------------------------------------------------------- labelling

def _low_blast_radius(state: dict) -> bool:
    """Where an automated script can misfire without hurting much."""
    return state["service_tier"] == "tier3" or (
        state["service_tier"] == "tier2" and state["severity"] == "sev3")


def _customer_impact(state: dict) -> bool:
    return state["severity"] == "sev1" or (
        state["severity"] == "sev2" and state["service_tier"] == "tier1")


def acceptable(state: dict) -> list[str]:
    """The responses a reviewer would sign off on for this alert.

    Written from the state alone, in code -- this is the label, not something the
    model sees. Several responses can be defensible at once, so it is a set.
    """
    sev, budget = state["severity"], state["error_budget"]
    deploy, hours = state["recent_deploy"], state["business_hours"]
    runbook = state["known_runbook"]
    pressure = budget in ("exhausted", "burning")

    # An incident someone is already working must not raise a second alarm.
    if state["duplicate_of_open_incident"] == "yes":
        out = ["suppress"]
        if deploy == "yes" and sev == "sev1" and state["service_tier"] == "tier1":
            out.append("rollback_deploy")      # still kill the build that did it
        elif runbook == "yes" and _low_blast_radius(state):
            out.append("auto_remediate")       # cheap, contained, adds no noise
        return out

    out: list[str] = []

    # A fresh deploy is the prime suspect once the damage is real.
    if deploy == "yes" and (_customer_impact(state) or (sev == "sev2" and pressure)):
        out.append("rollback_deploy")

    # Automation needs both a written procedure and a small blast radius.
    if runbook == "yes" and _low_blast_radius(state) and sev != "sev1":
        out.append("auto_remediate")

    # Wake someone for customer impact, or once a sev2 has spent the budget.
    if _customer_impact(state) or (sev == "sev2" and budget == "exhausted"):
        out.append("page_oncall")

    # Otherwise it is queue work -- unless it is sev3 noise nobody should read.
    if not out and not (sev == "sev3" and budget == "healthy" and hours == "no"):
        out.append("create_ticket")
    elif out == ["auto_remediate"] and hours == "yes":
        out.append("create_ticket")            # someone reviews the fix today

    return out or ["suppress"]


# -------------------------------------------------------------------- instances

def grid(per_label: int = 48, seed: int = 0) -> list[dict]:
    """Every field combination, sampled evenly across the label sets.

    The raw grid is wildly unbalanced -- half of it is duplicates, all
    suppressed -- and a metric dominated by one response would reward a prompt
    that had learnt nothing but the base rate.
    """
    rng = random.Random(seed)
    pool: dict[str, list[dict]] = {}
    for values in itertools.product(SEVERITY, SERVICE_TIER, ERROR_BUDGET,
                                    YES_NO, YES_NO, YES_NO, YES_NO):
        state = dict(zip(FIELDS, values, strict=True))
        good = acceptable(state)
        pool.setdefault("+".join(good), []).append(
            {"state": state, "acceptable": good, "source": "grid"})

    instances: list[dict] = []
    for _, items in sorted(pool.items()):
        rng.shuffle(items)
        instances.extend(items[:per_label])
    rng.shuffle(instances)
    return instances


# Groupings the per-field equalities cannot express on their own.
EXTRA = [
    Condition("blast_radius=low",
              "the blast radius is small -- service_tier is \"tier3\", or "
              "service_tier is \"tier2\" with a sev3 symptom",
              _low_blast_radius),
    Condition("customer_impact=yes",
              "customers are being hurt -- severity is \"sev1\", or severity is "
              "\"sev2\" on a tier1 service",
              _customer_impact),
    Condition("error_budget=under_pressure",
              "error_budget is \"exhausted\" or \"burning\"",
              lambda s: s.get("error_budget") in ("exhausted", "burning")),
]


def build() -> Task:
    instances = grid()
    return Task(
        name="triage",
        instructions=INSTRUCTIONS,
        options=dict(OPTIONS),
        instances=instances,
        conditions=derive(instances, extra=EXTRA),
        reference=dict(REFERENCE),
    )


# ------------------------------------------------------------------------ main

def main() -> None:
    task = build()
    counts = Counter("+".join(i["acceptable"]) for i in task.instances)

    print(f"{len(task.instances)} instances, {len(task.conditions)} conditions")
    print("label mix:")
    for label, n in counts.most_common():
        print(f"  {n:4d}  {label}")

    train, val, test = task.split()
    print(f"split: train {len(train)} / val {len(val)} / test {len(test)}")

    seen = [{task.signature(i["state"]) for i in part} for part in (train, val, test)]
    for a in range(3):
        for b in range(a + 1, 3):
            assert not seen[a] & seen[b], "split leaks a signature across parts"
    assert len(train) + len(val) + len(test) == len(task.instances)
    for instance in task.instances:
        good = set(instance["acceptable"])
        assert good and good <= set(task.options), instance

    # Not degenerate: always answering the single most common response fails a lot.
    votes = Counter(a for i in task.instances for a in i["acceptable"])
    best, hits = votes.most_common(1)[0]
    print(f"majority baseline: always {best!r} -> "
          f"{hits}/{len(task.instances)} = {hits / len(task.instances):.1%}")
    assert hits < len(task.instances), "task is degenerate"


if __name__ == "__main__":
    main()
