"""The Task contract: label validation, state signatures, and the leak-free split.

A Task is the whole interface between a domain and the optimiser, so its two
invariants have to hold before anything downstream means anything. Labels must
name real options and never be empty -- an instance no option answers would
score every arm identically. And the split must be drawn on condition
*signatures*, not on instances: two states the grammar cannot tell apart are the
same training example, so putting one in train and the other in test would leak
the answer and inflate every number the comparison prints.
"""

from __future__ import annotations

import pytest

from jevopt.task import Condition, Task


def _task(instances, conditions=()) -> Task:
    return Task(name="t", instructions="pick", options={"a": "A", "b": "B"},
                instances=instances, conditions=list(conditions))


def test_unknown_option_in_a_label_is_rejected():
    with pytest.raises(ValueError, match="unknown option"):
        _task([{"state": {}, "acceptable": ["a", "nope"]}])


def test_empty_acceptable_set_is_rejected():
    with pytest.raises(ValueError, match="no acceptable option"):
        _task([{"state": {}, "acceptable": []}])


def test_valid_labels_are_accepted():
    task = _task([{"state": {}, "acceptable": ["a"]},
                  {"state": {}, "acceptable": ["a", "b"]}])
    assert len(task.instances) == 2
    assert task.reference is None


def test_condition_holds_is_a_real_bool_and_survives_a_missing_field():
    """`holds` feeds straight into tuples and sums, so a None or a KeyError here
    would poison every truth table built from it."""
    condition = Condition("x", "x is 1", lambda s: s["x"] == 1)
    assert condition.holds({"x": 1}) is True
    assert condition.holds({"x": 2}) is False
    assert condition.holds({}) is False


def test_signature_separates_states_that_differ_on_a_condition(synthetic_task):
    red = {"color": "red", "size": "large", "position": {"zone": "north"}}
    blue = dict(red, color="blue")
    assert synthetic_task.signature(red) != synthetic_task.signature(blue)


def test_signature_collapses_states_no_condition_can_tell_apart(synthetic_task):
    """`env` is constant and `trace_id` is unique, so `derive` drops both. Two
    states differing only there are one example, however different their JSON."""
    reads = {c.id.split("=")[0] for c in synthetic_task.conditions}
    assert "env" not in reads and "trace_id" not in reads

    one = {"color": "red", "size": "small", "position": {"zone": "south"},
           "env": "staging", "trace_id": "t000"}
    two = dict(one, env="production", trace_id="t999")
    assert one != two
    assert synthetic_task.signature(one) == synthetic_task.signature(two)


def test_signature_is_one_flag_per_condition(any_task):
    state = any_task.instances[0]["state"]
    signature = any_task.signature(state)
    assert len(signature) == len(any_task.conditions)
    assert all(isinstance(flag, bool) for flag in signature)


def test_split_parts_are_populated_and_keep_every_instance(any_task):
    parts = any_task.split()
    assert len(parts) == 3
    for part in parts:
        assert part, "an empty split part would make a whole stage vacuous"
    assert sum(len(part) for part in parts) == len(any_task.instances)


def test_split_shares_no_signature_between_parts(any_task):
    """The reason the split exists: a signature in two parts is a leak."""
    seen: dict[tuple, int] = {}
    for index, part in enumerate(any_task.split()):
        for instance in part:
            signature = any_task.signature(instance["state"])
            assert seen.setdefault(signature, index) == index, "signature spans parts"
    assert len(seen) > 1


def test_split_is_deterministic_for_a_fixed_seed(any_task):
    """The comparison re-splits with the seed a run used; a wobble there would
    silently test an arm on instances it was trained on."""
    def shape(parts):
        return [[i["acceptable"] for i in part] for part in parts]

    assert shape(any_task.split(seed=3)) == shape(any_task.split(seed=3))
    assert shape(any_task.split(seed=3)) != shape(any_task.split(seed=4))


def test_save_writes_the_task_back_as_json(synthetic_task, tmp_path):
    import json

    path = tmp_path / "task.json"
    synthetic_task.save(str(path))
    data = json.loads(path.read_text())
    assert data["name"] == synthetic_task.name
    assert data["options"] == synthetic_task.options
    assert len(data["instances"]) == len(synthetic_task.instances)
