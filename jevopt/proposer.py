"""Mutation without generation.

Standard GEPA hands failure traces to a language model and asks it to write a
better instruction. Jev cannot write anything, so this proposer splits that job
in three and gives each part to whatever can actually do it:

  evidence   (code) confusion mass from Jev's own probability distributions says
             which two options failed to separate, and a search over the task's
             conditions finds ones that discriminate them on the failing cases
  judgement  (Jev) a choice question picks which candidate repair to apply
  edit       (code) the chosen clause is composed into the component's text

The output is still prompt text -- the candidate carries real English that grew
from the naive seed, and the run writes out the evolved prompt to diff.
"""

from __future__ import annotations

from collections import defaultdict

from . import client

from . import grammar
from .evidence import MIN_STRENGTH, Evidence
from .task import Task

MAX_CLAUSES = 4          # keep components from growing without bound
SHORTLIST = 5            # repairs offered to Jev per mutation
DIGEST_CASES = 4         # failing cases of each kind shown to Jev


def _confusion(records: list[dict], option: str) -> dict[str, float]:
    """Which other option is stealing (or being stolen by) this one."""
    mass: dict[str, float] = defaultdict(float)
    for record in records:
        if record["role"] == "chosen_wrongly":
            for winner in record["should_have_been"]:
                mass[winner] += record["p_this_option"]
        elif record["role"] == "missed":
            mass[record["model_chose"]] += record["p_model_chose"]
    mass.pop(option, None)
    return dict(mass)


def _candidate_clauses(task: Task, evidence: Evidence, option: str, partner: str,
                       text: str) -> list[dict]:
    """Shortlist repairs: the strongest sound additions, plus a removal if a
    clause already attached is not earning its place."""
    _base, attached = grammar.split_clauses(text, task)
    scored = []
    for condition in task.conditions:
        for template, other in (("only", None), ("never", None), ("prefer", partner)):
            strength = evidence.score(option, template, condition.id, other)
            if strength is None:
                continue
            clause = grammar.render_clause(task, template, condition.id, other)
            if clause in attached:
                continue
            scored.append({"clause": clause, "strength": round(strength, 3),
                           "template": template, "condition": condition.id, "op": "add"})

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
        worst = min(attached, key=lambda c: evidence.clause_value(option, c))
        if evidence.clause_value(option, worst) < MIN_STRENGTH:
            out.append({"clause": f"Remove this rule: {worst}", "strength": 0.0,
                        "template": "remove", "condition": worst, "op": "remove"})
    return out


def _digest(task: Task, records: list[dict], option: str, partner: str,
            text: str) -> dict:
    """The failure evidence, as the state of the repair question.

    The robot original hand-picked the state fields worth showing. Nothing
    generic can know which those are, so the whole state goes in -- it is the
    same shape Jev already reads when answering the task itself.
    """
    def summarise(record):
        return {"situation": record["state"],
                "correct_options": record["should_have_been"],
                "model_chose": record["model_chose"]}

    wrong = [summarise(r) for r in records if r["role"] == "chosen_wrongly"]
    missed = [summarise(r) for r in records if r["role"] == "missed"]
    return {
        "the_decision_being_made": task.instructions,
        "option_being_repaired": option,
        "its_current_description": text,
        "most_confused_with": partner,
        "cases_where_it_was_chosen_but_wrong": wrong[:DIGEST_CASES],
        "cases_where_it_was_right_but_another_option_won": missed[:DIGEST_CASES],
    }


REPAIR_INSTRUCTION = (
    "A decision is made by picking one option from a list, where each option "
    "carries a description of when to choose it. One option's description is "
    "being repaired because the decision keeps confusing it with another "
    "option. Below are the failing cases and the description as it stands. "
    "Which extra sentence, added to this option's description, would most "
    "reliably stop the confusion without making the option wrong in cases where "
    "it should still be chosen?"
)


class JevProposer:
    """A GEPA ProposalFn that never generates text."""

    def __init__(self, task: Task, train: list[dict], model: str = client.MODEL,
                 use_jev_choice: bool = True) -> None:
        self.task = task
        self.model = model
        self.evidence = Evidence(task, train)
        self.use_jev_choice = use_jev_choice
        self.log: list[dict] = []

    def __call__(self, candidate, reflective_dataset, components_to_update):
        proposals: dict[str, str] = {}
        for component in components_to_update:
            option = component[len(grammar.PREFIX):]
            records = list(reflective_dataset.get(component, []))
            text = candidate.get(component, self.task.options.get(option, ""))
            base, attached = grammar.split_clauses(text, self.task)
            if len(attached) > MAX_CLAUSES:
                continue

            failures = [r for r in records if r["role"] != "correct"]
            if not failures:
                continue
            mass = _confusion(records, option)
            if not mass:
                continue
            partner = max(mass, key=mass.get)

            shortlist = _candidate_clauses(self.task, self.evidence, option,
                                           partner, text)
            if not shortlist:
                continue

            pick = self._choose(
                shortlist, _digest(self.task, records, option, partner, text))
            if pick is None:
                continue
            if pick["op"] == "remove":
                new_text = grammar.compose(
                    base, [c for c in attached if c != pick["condition"]])
            else:
                new_text = grammar.apply_clause(text, pick["clause"])
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
