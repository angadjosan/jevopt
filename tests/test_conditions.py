"""Deriving the condition vocabulary from the states themselves.

`derive` is what lets a task arrive as nothing but labelled JSON: every leaf
field small enough to enumerate becomes one equality condition per value. The
filters matter as much as the enumeration -- a constant field says nothing about
any decision, and a field with a distinct value per instance is an identifier,
which would hand the grammar a clause that memorises single examples.
"""

from __future__ import annotations

from jevopt.conditions import MAX_VALUES, derive, flatten


def _instances(states) -> list[dict]:
    return [{"state": state, "acceptable": ["a"]} for state in states]


def test_one_condition_per_field_value_pair():
    conditions = derive(_instances([
        {"colour": "red", "size": "small"},
        {"colour": "blue", "size": "large"},
    ]))
    assert {c.id for c in conditions} == {
        "colour=red", "colour=blue", "size=small", "size=large"}
    for condition in conditions:
        field, value = condition.id.split("=")
        assert condition.holds({field: value}) is True
        assert condition.holds({field: "other"}) is False


def test_nested_dicts_are_flattened_into_dotted_paths():
    states = [{"where": {"axis": {"fb": "forward"}}, "flat": 1},
              {"where": {"axis": {"fb": "behind"}}, "flat": 2}]
    assert flatten(states[0]) == {"where.axis.fb": "forward", "flat": 1}

    ids = {c.id for c in derive(_instances(states))}
    assert "where.axis.fb=forward" in ids and "where.axis.fb=behind" in ids
    condition = next(c for c in derive(_instances(states))
                     if c.id == "where.axis.fb=forward")
    assert condition.holds(states[0]) is True
    assert condition.holds(states[1]) is False
    assert condition.holds({"where": "not a dict"}) is False


def test_constant_fields_are_dropped():
    """A field with one value cannot discriminate anything, so a clause about it
    is pure noise in the shortlist."""
    conditions = derive(_instances([{"env": "prod", "tier": "one"},
                                    {"env": "prod", "tier": "two"}]))
    assert {c.id for c in conditions} == {"tier=one", "tier=two"}


def test_wide_fields_are_dropped_as_identifiers():
    states = [{"trace": f"t{i}", "ok": i % 2 == 0} for i in range(MAX_VALUES + 1)]
    ids = {c.id for c in derive(_instances(states))}
    assert not any(i.startswith("trace=") for i in ids)
    assert ids == {"ok=True", "ok=False"}

    # Exactly at the cap it is still enumerable -- the bound is inclusive.
    narrow = [{"trace": f"t{i}"} for i in range(MAX_VALUES)]
    assert len(derive(_instances(narrow), max_values=MAX_VALUES)) == MAX_VALUES


def test_step_like_fields_are_skipped():
    ids = {c.id for c in derive(_instances([{"step": 1, "phase": "a"},
                                            {"step": 2, "phase": "b"}]))}
    assert not any(i.startswith("step=") for i in ids)
    assert ids == {"phase=a", "phase=b"}


def test_extra_conditions_are_appended():
    from jevopt.task import Condition

    extra = Condition("hand.written", "it is so", lambda s: True)
    conditions = derive(_instances([{"x": 1}, {"x": 2}]), extra=[extra])
    assert conditions[-1] is extra


def test_derived_ids_are_unique_and_phrases_slot_into_a_template(any_task):
    """The id is the key the proposer and the removal path both look clauses up
    by, and the phrase is dropped mid-sentence, so it must not end a sentence."""
    ids = [c.id for c in any_task.conditions]
    assert len(ids) == len(set(ids))
    for condition in any_task.conditions:
        assert condition.phrase.strip()
        assert not condition.phrase.endswith(".")


def test_holds_answers_true_or_false_for_every_real_state(any_task):
    """Totality over the real data: one exception or None would poison a whole
    truth table, and Evidence builds those before anything is scored."""
    for instance in any_task.instances:
        for condition in any_task.conditions:
            value = condition.holds(instance["state"])
            assert value is True or value is False, f"{condition.id}: {value!r}"


def test_holds_tolerates_a_missing_field(any_task):
    for condition in any_task.conditions:
        assert condition.holds({}) is False
