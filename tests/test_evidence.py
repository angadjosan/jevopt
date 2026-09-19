"""The evidence gates: what stops the search inventing rules.

These are the numbers that decide whether a clause is allowed to exist, so every
check here recomputes coverage and purity from the labels themselves rather than
trusting the scorer that produced the shortlist -- a gate that agrees with its
own arithmetic proves nothing. The three properties that matter: an option
without enough labelled examples yields no clause at all, an admitted clause
really does hold where its template claims, and clauses that constrain the
option identically collapse to one shortlist slot.
"""

from __future__ import annotations

import pytest

from jevopt import grammar
from jevopt import proposer as P
from jevopt.evidence import (
    COVERAGE,
    MIN_NEGATIVES,
    MIN_POSITIVES,
    MIN_STRENGTH,
    PURITY,
    Evidence,
)


def _truth(task, instances):
    """An independent truth table, built from the conditions and the raw labels."""
    return {c.id: [c.holds(i["state"]) for i in instances] for c in task.conditions}


def _rows(instances, option):
    positives = [n for n, i in enumerate(instances) if option in i["acceptable"]]
    negatives = [n for n, i in enumerate(instances) if option not in i["acceptable"]]
    return positives, negatives


def _rate(flags, rows):
    return sum(flags[n] for n in rows) / len(rows) if rows else 0.0


# ------------------------------------------------------------ the evidence gate

def test_an_under_evidenced_option_yields_no_clause(synthetic_task, synthetic_evidence):
    """delta is acceptable in one cell of the grid; a rule learned from three
    examples is a rule learned from noise."""
    assert len(synthetic_evidence.pos["delta"]) < MIN_POSITIVES
    for condition in synthetic_task.conditions:
        for template in ("only", "never"):
            assert synthetic_evidence.score("delta", template, condition.id, None) is None
    for partner in synthetic_task.options:
        assert P._candidate_clauses(synthetic_task, synthetic_evidence, "delta",
                                    partner, synthetic_task.options["delta"]) == []


def test_the_gate_holds_on_a_thin_real_option(robot_task):
    """Same gate on real data: the robot's train split barely sees close_gripper."""
    train, _val, _test = robot_task.split(seed=0)
    evidence = Evidence(robot_task, train)
    thin = [o for o in robot_task.options if len(evidence.pos[o]) < MIN_POSITIVES]
    assert thin, "no under-evidenced option in this split -- the gate is untested"
    for option in thin:
        for partner in robot_task.options:
            out = P._candidate_clauses(robot_task, evidence, option, partner,
                                       robot_task.options[option])
            assert out == [], f"{option}: {len(evidence.pos[option])} positives -> {out}"


def test_a_partner_without_evidence_blocks_a_prefer_clause(synthetic_task,
                                                           synthetic_evidence):
    """A `prefer` clause redirects to another option; redirecting to one nobody
    has examples of cannot be justified."""
    for condition in synthetic_task.conditions:
        assert synthetic_evidence.score("alpha", "prefer", condition.id,
                                        "delta") is None


# ------------------------------------------------- what an admitted clause means

def test_admitted_clauses_match_the_labels(any_task):
    """Recompute each template's promise from the data: `only` must cover almost
    every case where the option is still right, `never` must fire on almost none
    of them, and both must exclude a real share of the wrong cases."""
    instances = any_task.instances
    evidence = Evidence(any_task, instances)
    truth = _truth(any_task, instances)
    checked = {"only": 0, "never": 0, "prefer": 0}

    for option in any_task.options:
        positives, negatives = _rows(instances, option)
        for condition in any_task.conditions:
            flags = truth[condition.id]
            on_pos, on_neg = _rate(flags, positives), _rate(flags, negatives)
            partners = [None, *[o for o in any_task.options if o != option]]
            for partner in partners:
                templates = ("only", "never") if partner is None else ("prefer",)
                for template in templates:
                    strength = evidence.score(option, template, condition.id, partner)
                    if strength is None:
                        continue
                    assert len(positives) >= MIN_POSITIVES
                    assert len(negatives) >= MIN_NEGATIVES
                    assert strength >= MIN_STRENGTH
                    if template == "only":
                        assert on_pos >= COVERAGE, f"{option}/{condition.id}: {on_pos}"
                        assert strength == pytest.approx(1.0 - on_neg)
                    else:
                        assert on_pos <= PURITY, f"{option}/{condition.id}: {on_pos}"
                        assert strength == pytest.approx(on_neg)
                    if template == "prefer":
                        # Redirecting only makes sense if the condition really does
                        # describe the partner's own cases.
                        partner_pos, _ = _rows(instances, partner)
                        assert _rate(flags, partner_pos) >= 0.80
                    checked[template] += 1

    assert all(checked.values()), f"a template was never admitted: {checked}"


def test_shortlisted_clauses_clear_min_strength(any_task):
    """Nothing weak reaches Jev: the shortlist is sorted, capped, duplicate-free,
    and every entry clears the strength floor."""
    evidence = Evidence(any_task, any_task.instances)
    checked = 0
    for option in any_task.options:
        for partner in any_task.options:
            if partner == option:
                continue
            out = P._candidate_clauses(any_task, evidence, option, partner,
                                       any_task.options[option])
            assert len(out) <= P.SHORTLIST
            assert len({e["clause"] for e in out}) == len(out)
            assert [e["strength"] for e in out] == sorted(
                (e["strength"] for e in out), reverse=True)
            for entry in out:
                assert entry["op"] == "add", "a pristine seed has nothing to remove"
                assert entry["strength"] >= MIN_STRENGTH, entry
                checked += 1
    assert checked, "no clause was proposed at all -- the gates are untested"


# --------------------------------------------------------------------- shapes

def test_shape_collapses_logically_equivalent_clauses(synthetic_task,
                                                      synthetic_evidence):
    """"Never choose this when the colour is blue" and "only choose this when it
    is red" are one constraint on a two-valued field. Shortlisting both wastes a
    slot and narrows what Jev is actually choosing between."""
    red, blue = "color=red", "color=blue"
    assert synthetic_evidence.shape("only", red) == synthetic_evidence.shape("never",
                                                                             blue)
    assert synthetic_evidence.shape("only", red) != synthetic_evidence.shape("only",
                                                                             blue)


def test_shape_keeps_prefer_as_its_own_family(synthetic_evidence):
    """A `prefer` clause constrains exactly as `never` does but also names the
    alternative, so it is worth offering alongside it."""
    for cid in ("color=red", "size=large"):
        never, prefer = (synthetic_evidence.shape("never", cid),
                         synthetic_evidence.shape("prefer", cid))
        assert never[0] == prefer[0], "they permit the same states"
        assert never != prefer, "but the redirect is a different offer"


def test_the_shortlist_never_repeats_a_shape(any_task):
    evidence = Evidence(any_task, any_task.instances)
    for option in any_task.options:
        for partner in any_task.options:
            if partner == option:
                continue
            out = P._candidate_clauses(any_task, evidence, option, partner,
                                       any_task.options[option])
            shapes = [evidence.shape(e["template"], e["condition"]) for e in out]
            assert len(shapes) == len(set(shapes))


# ---------------------------------------------------------------- clause_value

def test_clause_value_reads_back_what_was_proposed(any_task):
    """Removal re-derives an attached clause's worth by re-rendering it. If the
    two paths disagreed, a clause worth keeping would be dropped as worthless."""
    evidence = Evidence(any_task, any_task.instances)
    checked = 0
    for option in any_task.options:
        for partner in any_task.options:
            if partner == option:
                continue
            for entry in P._candidate_clauses(any_task, evidence, option, partner,
                                              any_task.options[option]):
                value = evidence.clause_value(option, entry["clause"])
                assert value == pytest.approx(entry["strength"], abs=1e-3), entry
                checked += 1
    assert checked


def test_clause_value_is_zero_for_text_the_grammar_cannot_produce(synthetic_task,
                                                                  synthetic_evidence):
    assert synthetic_evidence.clause_value("alpha", "Something a human wrote.") == 0.0


def test_clause_value_is_zero_for_a_clause_the_gate_rejects(synthetic_task,
                                                            synthetic_evidence):
    """An unsound clause scores None; removal must read that as worth nothing
    rather than propagating the None into a comparison."""
    clause = grammar.render_clause(synthetic_task, "only", "size=large")
    assert synthetic_evidence.score("alpha", "only", "size=large", None) is None
    assert synthetic_evidence.clause_value("alpha", clause) == 0.0
