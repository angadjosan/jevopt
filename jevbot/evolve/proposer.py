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


MIN_POSITIVES = 5        # evidence gate: never learn a rule from 1-2 examples
MIN_NEGATIVES = 5
MIN_STRENGTH = 0.15      # a clause must exclude this share of the wrong cases
COVERAGE = 0.95          # an "only" clause must hold on ~all correct cases
PURITY = 0.05            # a "never" clause must hold on ~none of them


class Evidence:
    """Condition truth-tables over the whole training set, computed once.

    Scoring a clause costs no model calls -- it is set logic over labelled
    states -- so there is no reason to judge one on a 6-example minibatch. The
    minibatch still decides *which* component and which confusion to attack;
    only the quality of the repair is measured against everything known.
    """

    def __init__(self, instances: list[dict]) -> None:
        self.states = [i["state"] for i in instances]
        self.good = [set(i["acceptable"]) for i in instances]
        self.truth = {cid: [prompt.holds(cid, s) for s in self.states]
                      for cid, _p, _t in prompt.CONDITIONS}
        self.pos = {a: [i for i, g in enumerate(self.good) if a in g] for a in ACTIONS}
        self.neg = {a: [i for i, g in enumerate(self.good) if a not in g] for a in ACTIONS}

    def _rate(self, cid: str, rows: list[int]) -> float:
        if not rows:
            return 0.0
        truth = self.truth[cid]
        return sum(truth[i] for i in rows) / len(rows)

    def score(self, action: str, template: str, cid: str,
              partner: str | None) -> float | None:
        """Discriminative strength on train, or None if the clause is unsound."""
        pos, neg = self.pos[action], self.neg[action]
        if len(pos) < MIN_POSITIVES or len(neg) < MIN_NEGATIVES:
            return None
        on_pos, on_neg = self._rate(cid, pos), self._rate(cid, neg)

        if template == "only":
            strength = 1.0 - on_neg
            ok = on_pos >= COVERAGE
        elif template == "never":
            strength = on_neg
            ok = on_pos <= PURITY
        else:                                    # prefer <partner> when C
            if partner is None or len(self.pos[partner]) < MIN_POSITIVES:
                return None
            strength = on_neg
            ok = on_pos <= PURITY and self._rate(cid, self.pos[partner]) >= 0.80
        return strength if ok and strength >= MIN_STRENGTH else None

    def shape(self, template: str, cid: str) -> tuple:
        """Which train states a clause would still permit the action on.

        "Never choose this when the gripper is high" and "Only choose this when
        the gripper is not high" are the same constraint, and the grammar can
        produce both. Shortlisting both wastes a slot and narrows what Jev is
        actually choosing between, so they are collapsed by the states they
        allow. A "prefer" clause constrains identically but also names the
        alternative, so it is kept as its own family.
        """
        truth = self.truth[cid]
        allowed = tuple(truth) if template == "only" else tuple(not t for t in truth)
        return (allowed, "redirect" if template == "prefer" else "constrain")

    def clause_value(self, action: str, clause: str) -> float:
        """How much an already-attached clause is earning, for removal."""
        for cid, _phrase, _test in prompt.CONDITIONS:
            for template in ("only", "never"):
                if prompt.render_clause(template, cid) == clause:
                    return self.score(action, template, cid, None) or 0.0
            for other in ACTIONS:
                if prompt.render_clause("prefer", cid, other) == clause:
                    return self.score(action, "prefer", cid, other) or 0.0
        return 0.0


def _candidate_clauses(evidence: Evidence, action: str, partner: str,
                       text: str) -> list[dict]:
    """Shortlist repairs: the strongest sound additions, plus a removal if a
    clause already attached is not earning its place."""
    base, attached = prompt.split_clauses(text)
    scored = []
    for cid, _phrase, _test in prompt.CONDITIONS:
        for template, other in (("only", None), ("never", None), ("prefer", partner)):
            strength = evidence.score(action, template, cid, other)
            if strength is None:
                continue
            clause = prompt.render_clause(template, cid, other)
            if clause in attached:
                continue
            scored.append({"clause": clause, "strength": round(strength, 3),
                           "template": template, "condition": cid, "op": "add"})

    scored.sort(key=lambda row: (-row["strength"], row["clause"]))
    seen, out = set(), []
    for entry in scored:
        shape = evidence.shape(entry["template"], entry["condition"])
        if shape in seen:
            continue
        seen.add(shape)
        out.append(entry)
        if len(out) >= SHORTLIST - 1:
            break

    if attached:
        worst = min(attached, key=lambda c: evidence.clause_value(action, c))
        if evidence.clause_value(action, worst) < MIN_STRENGTH:
            out.append({"clause": f"Remove this rule: {worst}", "strength": 0.0,
                        "template": "remove", "condition": worst, "op": "remove"})
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

    def __init__(self, train: list[dict], model: str = client.MODEL,
                 use_jev_choice: bool = True) -> None:
        self.model = model
        self.evidence = Evidence(train)
        self.use_jev_choice = use_jev_choice
        self.log: list[dict] = []

    def __call__(self, candidate, reflective_dataset, components_to_update):
        proposals: dict[str, str] = {}
        for component in components_to_update:
            action = component[len(prompt.PREFIX):]
            records = list(reflective_dataset.get(component, []))
            text = candidate.get(component, prompt.SEED_CRITERIA.get(action, ""))
            base, attached = prompt.split_clauses(text)
            if len(attached) > MAX_CLAUSES:
                continue

            failures = [r for r in records if r["role"] != "correct"]
            if not failures:
                continue
            mass = _confusion(records, action)
            if not mass:
                continue
            partner = max(mass, key=mass.get)

            shortlist = _candidate_clauses(self.evidence, action, partner, text)
            if not shortlist:
                continue

            pick = self._choose(shortlist, _digest(records, action, partner, text))
            if pick is None:
                continue
            if pick["op"] == "remove":
                new_text = prompt.compose(
                    base, [c for c in attached if c != pick["condition"]])
            else:
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
        if len(shortlist) == 1 or not self.use_jev_choice:
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


MAX_CLAUSES_DOC = MAX_CLAUSES


def seed_candidate() -> dict[str, str]:
    return prompt.components(prompt.SEED_CRITERIA)


def handtuned_candidate() -> dict[str, str]:
    return prompt.components(prompt.HANDTUNED_CRITERIA)


ALL_COMPONENTS = [prompt.PREFIX + a for a in ACTIONS]
