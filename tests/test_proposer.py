"""The mutation path, minus the model: trace -> reflective dataset -> new text.

Everything the proposer does except picking between shortlisted repairs is
ordinary code, so it can be tested with fabricated traces and no API key. What
is worth guarding is the credit assignment: the adapter has to sort a trace into
the right role for each option (a wrong pick and a missed pick are different
evidence), the confusion mass has to point at the option that actually took the
decision away, and the chosen clause has to land in the text as a clause the
grammar can still recognise -- otherwise it can never be removed again.

`use_jev_choice=False` everywhere: it makes `_choose` return the strongest
repair directly, so nothing here reaches the network.
"""

from __future__ import annotations

from types import SimpleNamespace

from jevopt import grammar
from jevopt import proposer as P
from jevopt.adapter import JevAdapter


def _trace(state, acceptable, chosen, probabilities, confidence=0.9):
    return {"state": state, "acceptable": sorted(acceptable), "chosen": chosen,
            "correct": chosen in acceptable, "margin": 0.0,
            "probabilities": probabilities, "confidence": confidence}


def _batch(traces):
    return SimpleNamespace(trajectories=traces)


def _alpha_loses_to_gamma(task):
    """Fabricated evidence that alpha is being taken for gamma and vice versa."""
    red = {"color": "red", "size": "small", "position": {"zone": "north"}}
    blue = {"color": "blue", "size": "small", "position": {"zone": "south"}}
    return [
        _trace(red, ["alpha"], "gamma", {"alpha": 0.30, "gamma": 0.55}),
        _trace(blue, ["gamma"], "alpha", {"alpha": 0.50, "gamma": 0.40}),
        _trace(red, ["alpha"], "alpha", {"alpha": 0.80, "gamma": 0.10}),
    ]


def test_reflective_dataset_sorts_traces_into_roles(synthetic_task):
    adapter = JevAdapter(synthetic_task)
    records = adapter.make_reflective_dataset(
        {}, _batch(_alpha_loses_to_gamma(synthetic_task)),
        ["criteria.alpha"])["criteria.alpha"]

    assert [r["role"] for r in records] == ["missed", "chosen_wrongly", "correct"]
    assert records[0]["model_chose"] == "gamma"
    assert records[0]["p_model_chose"] == 0.55
    assert records[1]["p_this_option"] == 0.50
    assert records[1]["should_have_been"] == ["gamma"]


def test_reflective_dataset_ignores_traces_that_say_nothing_about_the_option(
        synthetic_task):
    """A case where the option was neither right nor chosen is not evidence about
    that option's description, and would only dilute the digest."""
    other = {"color": "blue", "size": "large", "position": {"zone": "north"}}
    traces = [_trace(other, ["beta"], "gamma", {"beta": 0.4, "gamma": 0.5}),
              _trace(other, ["beta"], None, {})]        # an errored call
    adapter = JevAdapter(synthetic_task)
    out = adapter.make_reflective_dataset({}, _batch(traces), ["criteria.alpha"])
    assert out == {"criteria.alpha": []}


def test_confusion_points_at_the_option_that_took_the_decision(synthetic_task):
    records = JevAdapter(synthetic_task).make_reflective_dataset(
        {}, _batch(_alpha_loses_to_gamma(synthetic_task)),
        ["criteria.alpha"])["criteria.alpha"]

    mass = P._confusion(records, "alpha")
    assert "alpha" not in mass, "an option cannot be confused with itself"
    assert max(mass, key=mass.get) == "gamma"
    assert mass["gamma"] == 0.55 + 0.50


def test_a_proposal_appends_a_clause_the_grammar_can_take_back(synthetic_task):
    """The whole loop offline: the edit has to leave the seed intact and the new
    sentence has to split back out, or the removal operator is dead."""
    proposer = P.JevProposer(synthetic_task, synthetic_task.instances,
                             use_jev_choice=False)
    candidate = grammar.components(synthetic_task.options)
    reflective = JevAdapter(synthetic_task).make_reflective_dataset(
        candidate, _batch(_alpha_loses_to_gamma(synthetic_task)), ["criteria.alpha"])

    proposals = proposer(candidate, reflective, ["criteria.alpha"])
    text = proposals["criteria.alpha"]
    base, attached = grammar.split_clauses(text, synthetic_task)
    assert base == synthetic_task.options["alpha"]
    assert len(attached) == 1
    assert attached[0] in grammar.all_clauses(synthetic_task)
    assert proposer.log[0]["confused_with"] == "gamma"


def test_no_failures_means_no_proposal(synthetic_task):
    """A component only gets rewritten on evidence that it is going wrong."""
    red = {"color": "red", "size": "small", "position": {"zone": "north"}}
    traces = [_trace(red, ["alpha"], "alpha", {"alpha": 0.9})]
    candidate = grammar.components(synthetic_task.options)
    reflective = JevAdapter(synthetic_task).make_reflective_dataset(
        candidate, _batch(traces), ["criteria.alpha"])

    proposer = P.JevProposer(synthetic_task, synthetic_task.instances,
                             use_jev_choice=False)
    assert proposer(candidate, reflective, ["criteria.alpha"]) == {}


def test_a_saturated_component_stops_growing(synthetic_task):
    """MAX_CLAUSES is the only thing keeping a description from accreting a
    sentence per generation."""
    text = synthetic_task.options["alpha"]
    clauses = sorted(grammar.all_clauses(synthetic_task))[:P.MAX_CLAUSES + 1]
    for clause in clauses:
        text = grammar.apply_clause(text, clause)
    _base, attached = grammar.split_clauses(text, synthetic_task)
    assert len(attached) > P.MAX_CLAUSES

    candidate = dict(grammar.components(synthetic_task.options),
                     **{"criteria.alpha": text})
    reflective = JevAdapter(synthetic_task).make_reflective_dataset(
        candidate, _batch(_alpha_loses_to_gamma(synthetic_task)), ["criteria.alpha"])
    proposer = P.JevProposer(synthetic_task, synthetic_task.instances,
                             use_jev_choice=False)
    assert proposer(candidate, reflective, ["criteria.alpha"]) == {}


def test_a_worthless_attached_clause_is_offered_for_removal(synthetic_task,
                                                            synthetic_evidence):
    """The shortlist gains a removal entry only when a clause already attached has
    stopped earning its place -- that is the search's only way back."""
    dud = grammar.render_clause(synthetic_task, "only", "size=large")
    assert synthetic_evidence.clause_value("alpha", dud) < P.MIN_STRENGTH

    text = grammar.apply_clause(synthetic_task.options["alpha"], dud)
    out = P._candidate_clauses(synthetic_task, synthetic_evidence, "alpha",
                               "gamma", text)
    removals = [e for e in out if e["op"] == "remove"]
    assert len(removals) == 1
    assert removals[0]["condition"] == dud
    assert dud not in [e["clause"] for e in out if e["op"] == "add"]


def test_the_digest_carries_the_failing_cases_and_the_current_text(synthetic_task):
    records = JevAdapter(synthetic_task).make_reflective_dataset(
        {}, _batch(_alpha_loses_to_gamma(synthetic_task)),
        ["criteria.alpha"])["criteria.alpha"]

    digest = P._digest(synthetic_task, records, "alpha", "gamma",
                       synthetic_task.options["alpha"])
    assert digest["option_being_repaired"] == "alpha"
    assert digest["most_confused_with"] == "gamma"
    assert digest["its_current_description"] == synthetic_task.options["alpha"]
    assert digest["the_decision_being_made"] == synthetic_task.instructions
    assert len(digest["cases_where_it_was_chosen_but_wrong"]) == 1
    assert len(digest["cases_where_it_was_right_but_another_option_won"]) == 1
