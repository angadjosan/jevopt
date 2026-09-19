"""Mutation without generation.

Standard GEPA hands failure traces to a language model and asks it to write a
better instruction. Jev cannot write anything, so this proposer splits that job
in three and gives each part to whatever can actually do it:

  evidence   (code) confusion mass from Jev's own probability distributions says
             which two options failed to separate, and a search over the state
             schema finds conditions that discriminate them on the failing cases
  judgement  (Jev) a choice question picks which candidate repair to apply
  edit       (code) the chosen clause is composed into the component's text

The output is still prompt text -- the candidate carries real English that grew
from the naive seed, and the run writes out the evolved prompt to diff.
"""

from __future__ import annotations

from collections import defaultdict

from .. import client
from ..sim import ACTIONS
from . import prompt

MAX_CLAUSES = 4          # keep components from growing without bound
SHORTLIST = 5            # repairs offered to Jev per mutation


def _confusion(records: list[dict], action: str) -> dict[str, float]:
    """Which other option is stealing (or being stolen by) this one."""
    mass: dict[str, float] = defaultdict(float)
    for record in records:
        if record["role"] == "chosen_wrongly":
            for winner in record["should_have_been"]:
                mass[winner] += record["p_this_action"]
        elif record["role"] == "missed":
            mass[record["model_chose"]] += record["p_model_chose"]
    mass.pop(action, None)
    return dict(mass)


def _candidate_clauses(records: list[dict], action: str, partner: str):
    """Conditions that separate 'this action is right' from 'it is not'.

    Pure set logic over the state schema -- no model involved. A clause only
    survives if it is true on every state where the action is correct and false
    on at least one where it is not (or the mirror image, for exclusions).
    """
    positives = [r["state"] for r in records if action in r["should_have_been"]]
    negatives = [r["state"] for r in records if action not in r["should_have_been"]]
    partner_states = [r["state"] for r in records
                      if partner in r["should_have_been"]]
    if not negatives:
        return []

    scored = []
    for cid, _phrase, _test in prompt.CONDITIONS:
        pos_hold = [prompt.holds(cid, s) for s in positives]
        neg_hold = [prompt.holds(cid, s) for s in negatives]

        # "Only choose this when C": C must be necessary for the action.
        if positives and all(pos_hold):
            excluded = sum(1 for h in neg_hold if not h)
            if excluded:
                scored.append((excluded / len(negatives), "only", cid, None))

        # "Never choose this when C": C must never hold when the action is right.
        if not any(pos_hold):
            caught = sum(1 for h in neg_hold if h)
            if caught:
                scored.append((caught / len(negatives), "never", cid, None))

        # "When C, choose <partner> instead": C marks the partner's territory.
        if partner_states and not any(pos_hold) and \
                all(prompt.holds(cid, s) for s in partner_states):
            caught = sum(1 for h in neg_hold if h)
            if caught:
                scored.append((caught / len(negatives), "prefer", cid, partner))

    scored.sort(key=lambda row: -row[0])
    seen, out = set(), []
    for strength, template, cid, other in scored:
        clause = prompt.render_clause(template, cid, other)
        if clause in seen:
            continue
        seen.add(clause)
        out.append({"clause": clause, "strength": round(strength, 3),
                    "template": template, "condition": cid})
        if len(out) >= SHORTLIST:
            break
    return out


def _digest(records: list[dict], action: str, partner: str, text: str) -> dict:
    def summarise(record):
        where = record["state"].get("where_the_apple_is_relative_to_the_gripper", {})
        return {
            "situation": {
                "forward_back": where.get("along_forward_back_axis"),
                "left_right": where.get("along_left_right_axis"),
                "height": where.get("height"),
                "over_the_apple": record["state"].get(
                    "gripper_is_horizontally_over_the_apple"),
                "right_height_to_grip": record["state"].get(
                    "gripper_is_at_the_right_height_to_grip"),
                "fingers": record["state"].get("fingers"),
                "gripping_something": record["state"].get(
                    "fingers_are_gripping_an_object"),
            },
            "correct_actions": record["should_have_been"],
            "model_chose": record["model_chose"],
        }

    wrong = [summarise(r) for r in records if r["role"] == "chosen_wrongly"][:4]
    missed = [summarise(r) for r in records if r["role"] == "missed"][:4]
    return {
        "action_being_repaired": action,
        "its_current_description": text,
        "most_confused_with": partner,
        "cases_where_it_was_chosen_but_wrong": wrong,
        "cases_where_it_was_right_but_another_option_won": missed,
    }


REPAIR_INSTRUCTION = (
    "A robot arm chooses its next action by picking one option from a list, "
    "where each option carries a description of when to choose it. One option's "
    "description is being repaired because the arm keeps confusing it with "
    "another option. Below are the failing cases and the description as it "
    "stands. Which extra sentence, added to this option's description, would "
    "most reliably stop the confusion without making the option wrong in cases "
    "where it should still be chosen?"
)


class JevProposer:
    """A GEPA ProposalFn that never generates text."""

    def __init__(self, model: str = client.MODEL) -> None:
        self.model = model
        self.log: list[dict] = []

    def __call__(self, candidate, reflective_dataset, components_to_update):
        proposals: dict[str, str] = {}
        for component in components_to_update:
            action = component[len(prompt.PREFIX):]
            records = list(reflective_dataset.get(component, []))
            text = candidate.get(component, prompt.SEED_CRITERIA.get(action, ""))
            if text.count(". ") >= MAX_CLAUSES + 1:
                continue

            failures = [r for r in records if r["role"] != "correct"]
            if not failures:
                continue
            mass = _confusion(records, action)
            if not mass:
                continue
            partner = max(mass, key=mass.get)

            shortlist = _candidate_clauses(records, action, partner)
            if not shortlist:
                continue

            pick = self._choose(shortlist, _digest(records, action, partner, text))
            if pick is None:
                continue
            new_text = prompt.apply_clause(text, pick["clause"])
            if new_text == text:
                continue
            proposals[component] = new_text
            self.log.append({"component": component, "confused_with": partner,
                             "confusion_mass": round(mass[partner], 3),
                             "clause": pick["clause"],
                             "strength": pick["strength"],
                             "considered": [c["clause"] for c in shortlist]})
        return proposals

    def _choose(self, shortlist: list[dict], digest: dict) -> dict | None:
        """Jev picks the repair. One choice question over the shortlisted clauses."""
        if len(shortlist) == 1:
            return shortlist[0]
        options = {f"repair_{i}": entry["clause"] for i, entry in enumerate(shortlist)}
        try:
            body = client.ask(
                digest,
                {"repair": {"type": "choice",
                            "instructions": REPAIR_INSTRUCTION,
                            "criteria": options}},
                model=self.model,
            )
            chosen = body["answers"]["repair"]["choice"]
            return shortlist[int(chosen.split("_")[1])]
        except Exception:
            return shortlist[0]      # fall back to the most discriminative


def seed_candidate() -> dict[str, str]:
    return prompt.components(prompt.SEED_CRITERIA)


def handtuned_candidate() -> dict[str, str]:
    return prompt.components(prompt.HANDTUNED_CRITERIA)


ALL_COMPONENTS = [prompt.PREFIX + a for a in ACTIONS]
