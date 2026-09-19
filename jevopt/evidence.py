"""Condition truth-tables: how a clause is judged without spending a Jev call.

Scoring a clause is set logic over labelled states, so it costs nothing and
there is no reason to judge one on a six-example minibatch. The minibatch still
decides *which* component and which confusion to attack; only the quality of the
repair is measured against everything known.

The gates below are what stops the search inventing rules. A clause has to be
sound (never wrong on the cases where the option should still be chosen) and
strong (it has to exclude a real share of the cases where it should not), and
the option has to have enough labelled examples either way for those rates to
mean anything.
"""

from __future__ import annotations

from . import grammar
from .task import Task

MIN_POSITIVES = 5        # evidence gate: never learn a rule from 1-2 examples
MIN_NEGATIVES = 5
MIN_STRENGTH = 0.15      # a clause must exclude this share of the wrong cases
COVERAGE = 0.95          # an "only" clause must hold on ~all correct cases
PURITY = 0.05            # a "never" clause must hold on ~none of them


class Evidence:
    """Condition truth-tables over the whole training set, computed once."""

    def __init__(self, task: Task, instances: list[dict]) -> None:
        self.task = task
        self.states = [i["state"] for i in instances]
        self.good = [set(i["acceptable"]) for i in instances]
        self.truth = {c.id: [c.holds(s) for s in self.states] for c in task.conditions}
        self.pos = {o: [i for i, g in enumerate(self.good) if o in g]
                    for o in task.options}
        self.neg = {o: [i for i, g in enumerate(self.good) if o not in g]
                    for o in task.options}

    def _rate(self, cid: str, rows: list[int]) -> float:
        if not rows:
            return 0.0
        truth = self.truth[cid]
        return sum(truth[i] for i in rows) / len(rows)

    def score(self, option: str, template: str, cid: str,
              partner: str | None) -> float | None:
        """Discriminative strength on train, or None if the clause is unsound."""
        pos, neg = self.pos[option], self.neg[option]
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
        """Which train states a clause would still permit the option on.

        "Never choose this when severity is high" and "Only choose this when
        severity is not high" are the same constraint, and the grammar can
        produce both. Shortlisting both wastes a slot and narrows what Jev is
        actually choosing between, so they are collapsed by the states they
        allow. A "prefer" clause constrains identically but also names the
        alternative, so it is kept as its own family.
        """
        truth = self.truth[cid]
        allowed = tuple(truth) if template == "only" else tuple(not t for t in truth)
        return (allowed, "redirect" if template == "prefer" else "constrain")

    def clause_value(self, option: str, clause: str) -> float:
        """How much an already-attached clause is earning, for removal."""
        for condition in self.task.conditions:
            for template in ("only", "never"):
                if grammar.render_clause(self.task, template, condition.id) == clause:
                    return self.score(option, template, condition.id, None) or 0.0
            for other in self.task.options:
                rendered = grammar.render_clause(
                    self.task, "prefer", condition.id, other)
                if rendered == clause:
                    return self.score(option, "prefer", condition.id, other) or 0.0
        return 0.0
