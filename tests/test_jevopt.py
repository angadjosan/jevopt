"""Offline guards for the parts of the evolve loop that have no model in them.

The search only works if the machinery around Jev is exact: a clause has to be
recognisable in a component's text after it was appended (otherwise the removal
operator silently stops working and components only ever grow), the frozen
dataset's labels have to be the ones `acceptable()` would compute today
(otherwise every score is measured against a stale oracle), and the evidence
gates have to mean what their names say (otherwise a rule gets learned from two
examples). None of that needs the API, so none of these tests touch it -- they
read `dataset.json` off disk and run set logic. Fast enough to run on every edit.

    python3 -m pytest jevbot/evolve/test_evolve.py -q
    python3 -m jevbot.evolve.test_evolve          # no pytest needed
"""

from __future__ import annotations

import json
import os

from .. import sim
from ..sim import ACTIONS
from . import dataset, prompt
from . import proposer as P

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")


def _load() -> dict:
    with open(DATASET) as fh:
        return json.load(fh)


def _instances() -> list[dict]:
    data = _load()
    return [i for split in ("train", "val", "test") for i in data[split]]


# ------------------------------------------------------------- clause grammar

def test_clause_round_trip():
    """Every producible clause must survive append -> split -> compose.

    Load-bearing: the removal operator finds an attached clause by matching it
    against `all_clauses()`, so any clause that does not come back out of
    `split_clauses()` is one the search can add but never take away.
    """
    clauses = sorted(prompt.all_clauses())
    assert clauses, "the grammar produced nothing"
    for name, seed in prompt.SEED_CRITERIA.items():
        for clause in clauses:
            text = prompt.apply_clause(seed, clause)
            base, found = prompt.split_clauses(text)
            assert base == seed.rstrip(), f"{name}: base mangled by {clause!r} -> {base!r}"
            assert found == [clause], f"{name}: {clause!r} came back as {found!r}"
            assert prompt.compose(base, found) == text, f"{name}: compose did not invert"


def test_apply_clause_is_idempotent():
    clause = prompt.render_clause("only", prompt.CONDITIONS[0][0])
    once = prompt.apply_clause(prompt.SEED_CRITERIA["descend"], clause)
    assert prompt.apply_clause(once, clause) == once


def test_seed_text_has_no_clauses():
    """An untouched seed is all base -- including the two-sentence ones."""
    for name, seed in prompt.SEED_CRITERIA.items():
        base, found = prompt.split_clauses(seed)
        assert found == [], f"{name}: seed text read as clauses {found!r}"
        assert base == seed.rstrip(), f"{name}: base lost text -> {base!r}"

    # open_gripper is two sentences; both belong to the base, neither is a clause.
    base, found = prompt.split_clauses(prompt.SEED_CRITERIA["open_gripper"])
    assert found == []
    assert base.count(".") == 2, base
    assert "Releases anything held." in base


def test_multiple_clauses_round_trip():
    clauses = [prompt.render_clause("only", "fb.forward"),
               prompt.render_clause("never", "fingers.closed"),
               prompt.render_clause("prefer", "hold.yes", "done")]
    text = prompt.SEED_CRITERIA["move_forward"]
    for clause in clauses:
        text = prompt.apply_clause(text, clause)
    base, found = prompt.split_clauses(text)
    assert found == clauses
    assert base == prompt.SEED_CRITERIA["move_forward"]
    assert prompt.compose(base, found) == text


def test_all_clauses_matches_the_grammar():
    """Size is combinatorial, not a magic number: two templates with no partner
    plus one `prefer` per action, for each condition."""
    expected = len(prompt.CONDITIONS) * (len(prompt.TEMPLATES) - 1 + len(ACTIONS))
    assert len(prompt.all_clauses()) == expected
    assert set(prompt.TEMPLATES) == {"only", "never", "prefer"}

    # No two (template, condition, other) triples collide into the same string.
    rendered = [prompt.render_clause(t, cid)
                for t in ("only", "never") for cid, _p, _x in prompt.CONDITIONS]
    rendered += [prompt.render_clause("prefer", cid, other)
                 for cid, _p, _x in prompt.CONDITIONS for other in ACTIONS]
    assert len(rendered) == len(set(rendered)) == expected


# ------------------------------------------------------------------ conditions

def test_condition_ids_are_unique():
    ids = [cid for cid, _p, _t in prompt.CONDITIONS]
    assert len(ids) == len(set(ids))
    assert set(prompt.CONDITION_BY_ID) == set(ids)
    for cid, phrase, _t in prompt.CONDITIONS:
        assert phrase.strip(), cid
        assert not phrase.endswith("."), f"{cid}: phrase must slot into a template"


def test_holds_is_total_over_the_dataset():
    """Every predicate must answer True/False for every stored state -- a
    KeyError or a None here would poison the whole truth table."""
    states = [i["state"] for i in _instances()]
    assert states
    for cid, _phrase, _test in prompt.CONDITIONS:
        for state in states:
            value = prompt.holds(cid, state)
            assert value is True or value is False, f"{cid}: returned {value!r}"


def test_holds_tolerates_a_missing_field():
    for cid, _phrase, _test in prompt.CONDITIONS:
        assert prompt.holds(cid, {}) is False


# --------------------------------------------------------------------- dataset

def test_labels_reproduce_from_the_state():
    """The frozen labels must equal what `acceptable()` computes now."""
    for inst in _instances():
        assert dataset.acceptable(inst["state"]) == inst["acceptable"], inst["state"]


def test_labels_are_non_empty_subsets_of_actions():
    for inst in _instances():
        good = inst["acceptable"]
        assert good, inst["state"]
        assert len(set(good)) == len(good), good
        assert set(good) <= set(ACTIONS), good


def test_splits_are_populated_and_the_grid_is_distinct():
    """Rollouts keep several states per situation bucket, so an identical state
    can land in two splits; the enumerated grid must not repeat itself."""
    data = _load()
    assert set(data) == {"train", "val", "test"}
    grid_keys = set()
    for name in ("train", "val", "test"):
        assert data[name], f"{name} split is empty"
        for inst in data[name]:
            assert set(inst) >= {"state", "acceptable", "source"}
            if inst["source"] != "grid":
                continue
            key = json.dumps(inst["state"], sort_keys=True)
            assert key not in grid_keys, f"grid state repeated in {name}"
            grid_keys.add(key)
    assert grid_keys


def test_acceptable_covers_the_dataset_vocabulary():
    """The grid enumerates the schema, so every axis value is spelled the way
    the conditions expect."""
    for inst in _instances():
        where = inst["state"]["where_the_apple_is_relative_to_the_gripper"]
        assert where["along_forward_back_axis"] in dataset.FB
        assert where["along_left_right_axis"] in dataset.LR
        assert where["height"] in dataset.HEIGHT
        pair = (inst["state"]["fingers"],
                inst["state"]["fingers_are_gripping_an_object"])
        assert pair in dataset.GRIP


# -------------------------------------------------------------- evidence gates

def _train_evidence() -> P.Evidence:
    return P.Evidence(_load()["train"])


def test_thin_evidence_yields_no_clause():
    """(a) An action with fewer than MIN_POSITIVES positives is unlearnable."""
    evidence = _train_evidence()
    thin = [a for a in ACTIONS if len(evidence.pos[a]) < P.MIN_POSITIVES]
    assert thin, "no under-evidenced action in this dataset -- gate untested"
    for action in thin:
        for partner in ACTIONS:
            out = P._candidate_clauses(evidence, action, partner,
                                       prompt.SEED_CRITERIA[action])
            assert out == [], f"{action}: {len(evidence.pos[action])} positives -> {out}"
        for cid, _p, _t in prompt.CONDITIONS:
            for template in ("only", "never"):
                assert evidence.score(action, template, cid, None) is None


def test_proposed_clauses_clear_min_strength():
    """(b) Nothing weak gets shortlisted for a well-evidenced action."""
    evidence = _train_evidence()
    strong = [a for a in ACTIONS if len(evidence.pos[a]) >= P.MIN_POSITIVES]
    assert strong
    checked = 0
    for action in strong:
        for partner in ACTIONS:
            if partner == action:
                continue
            for entry in P._candidate_clauses(evidence, action, partner,
                                              prompt.SEED_CRITERIA[action]):
                assert entry["op"] == "add", "a pristine seed has nothing to remove"
                assert entry["strength"] >= P.MIN_STRENGTH, entry
                checked += 1
    assert checked, "no clause was proposed at all -- the gates are untested"


def test_proposed_clauses_match_the_data():
    """(c) Recompute coverage and purity from the labels, rather than trusting
    the scorer that produced the shortlist."""
    evidence = _train_evidence()
    instances = _load()["train"]
    checked = {"only": 0, "never": 0, "prefer": 0}

    for action in ACTIONS:
        if len(evidence.pos[action]) < P.MIN_POSITIVES:
            continue
        positives = [i["state"] for i in instances if action in i["acceptable"]]
        negatives = [i["state"] for i in instances if action not in i["acceptable"]]
        assert len(positives) >= P.MIN_POSITIVES and len(negatives) >= P.MIN_NEGATIVES

        for partner in ACTIONS:
            if partner == action:
                continue
            for entry in P._candidate_clauses(evidence, action, partner,
                                              prompt.SEED_CRITERIA[action]):
                cid, template = entry["condition"], entry["template"]
                on_pos = sum(prompt.holds(cid, s) for s in positives) / len(positives)
                on_neg = sum(prompt.holds(cid, s) for s in negatives) / len(negatives)

                if template == "only":
                    assert on_pos >= P.COVERAGE, f"{action}/{cid}: covers {on_pos:.3f}"
                    assert 1.0 - on_neg >= P.MIN_STRENGTH
                    assert abs(entry["strength"] - round(1.0 - on_neg, 3)) < 1e-9
                else:
                    assert on_pos <= P.PURITY, f"{action}/{cid}: fires on {on_pos:.3f}"
                    assert on_neg >= P.MIN_STRENGTH
                    assert abs(entry["strength"] - round(on_neg, 3)) < 1e-9
                    if template == "prefer":
                        # It only makes sense to redirect to the partner if the
                        # condition really does describe the partner's cases.
                        partner_pos = [i["state"] for i in instances
                                       if partner in i["acceptable"]]
                        rate = sum(prompt.holds(cid, s) for s in partner_pos) / len(partner_pos)
                        assert rate >= 0.80, f"{action}->{partner}/{cid}: {rate:.3f}"
                checked[template] += 1

    assert checked["only"] and checked["never"], checked


def test_clause_value_reads_back_what_was_proposed():
    """Removal scores an attached clause by re-rendering it; the two paths must
    agree or a good clause gets dropped as worthless."""
    evidence = _train_evidence()
    for action in ACTIONS:
        if len(evidence.pos[action]) < P.MIN_POSITIVES:
            continue
        for entry in P._candidate_clauses(evidence, action, "descend",
                                          prompt.SEED_CRITERIA[action]):
            value = evidence.clause_value(action, entry["clause"])
            assert abs(value - entry["strength"]) < 1e-3, (action, entry, value)


def test_shortlist_is_bounded_and_unique():
    evidence = _train_evidence()
    for action in ACTIONS:
        out = P._candidate_clauses(evidence, action, "descend",
                                   prompt.SEED_CRITERIA[action])
        assert len(out) <= P.SHORTLIST
        assert len({e["clause"] for e in out}) == len(out)
        strengths = [e["strength"] for e in out]
        assert strengths == sorted(strengths, reverse=True)


# ------------------------------------------------------------------- candidate

def test_criteria_of_covers_every_action():
    for candidate in (P.seed_candidate(), P.handtuned_candidate(), {}):
        criteria = prompt.criteria_of(candidate)
        assert list(criteria) == list(ACTIONS)
        for name, text in criteria.items():
            assert isinstance(text, str) and text.strip(), name


def test_questions_offer_one_option_per_action():
    for candidate in (P.seed_candidate(), P.handtuned_candidate()):
        built = prompt.questions(candidate)
        assert set(built) == {"action"}
        question = built["action"]
        assert question["type"] == "choice"
        assert question["instructions"].strip()
        assert list(question["criteria"]) == list(ACTIONS)
        for name, text in question["criteria"].items():
            assert text.strip(), name


def test_components_round_trip_through_the_prefix():
    candidate = P.seed_candidate()
    assert list(candidate) == [prompt.PREFIX + a for a in prompt.SEED_CRITERIA]
    assert prompt.criteria_of(candidate) == {a: prompt.SEED_CRITERIA[a] for a in ACTIONS}
    assert set(P.ALL_COMPONENTS) == set(prompt.components(dict(ACTIONS)))
    assert prompt.HANDTUNED_CRITERIA == dict(sim.ACTIONS)


# ------------------------------------------------------------------ pytest-less

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001 -- report, not raise
            failed.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f"   failed: {', '.join(failed)}" if failed else ""))
    raise SystemExit(1 if failed else 0)
