"""The clause grammar: what the search can say, and what it can take back.

The load-bearing property is round-tripping. The removal operator recognises an
attached clause by matching the component's sentences against `all_clauses()`,
so a clause that does not come back out of `split_clauses()` is one the search
can append but never retract -- components would only ever grow, and a bad
clause would be permanent. Everything else here guards the same boundary: seed
text must not be mistaken for clauses, and the clause set must be exactly the
grammar's combinatorics.
"""

from __future__ import annotations

import pytest

from jevopt import grammar

SAMPLE = 40          # clauses per option on the real tasks; synthetic is exhaustive


def _round_trip(task, option_text: str, clause: str) -> None:
    text = grammar.apply_clause(option_text, clause)
    base, found = grammar.split_clauses(text, task)
    assert base == option_text.rstrip(), f"base mangled by {clause!r} -> {base!r}"
    assert found == [clause], f"{clause!r} came back as {found!r}"
    assert grammar.compose(base, found) == text, "compose did not invert split"


def test_every_clause_round_trips_on_the_synthetic_task(synthetic_task):
    """Exhaustive: every clause x every option, on a task small enough to afford it."""
    clauses = sorted(grammar.all_clauses(synthetic_task))
    assert clauses
    for text in synthetic_task.options.values():
        for clause in clauses:
            _round_trip(synthetic_task, text, clause)


def test_clauses_round_trip_on_the_real_tasks(real_task):
    """Same property, sampled: real phrases carry punctuation and field paths that
    a synthetic vocabulary never would."""
    clauses = sorted(grammar.all_clauses(real_task))
    stride = max(1, len(clauses) // SAMPLE)
    sampled = clauses[::stride]
    assert len(sampled) >= 3
    for text in real_task.options.values():
        for clause in sampled:
            _round_trip(real_task, text, clause)


def test_several_clauses_round_trip_together(synthetic_task):
    """Components accumulate clauses, so the split has to survive more than one."""
    condition = synthetic_task.conditions[0].id
    other = list(synthetic_task.options)[-1]
    clauses = [grammar.render_clause(synthetic_task, "only", condition),
               grammar.render_clause(synthetic_task, "never",
                                     synthetic_task.conditions[1].id),
               grammar.render_clause(synthetic_task, "prefer", condition, other)]
    seed = synthetic_task.options["beta"]
    text = seed
    for clause in clauses:
        text = grammar.apply_clause(text, clause)

    base, found = grammar.split_clauses(text, synthetic_task)
    assert found == clauses
    assert base == seed
    assert grammar.compose(base, found) == text


def test_seed_text_contains_no_clauses(any_task):
    """An untouched option description is all base -- otherwise the first removal
    would delete words a human wrote."""
    for name, seed in any_task.options.items():
        base, found = grammar.split_clauses(seed, any_task)
        assert found == [], f"{name}: seed text read as clauses {found!r}"
        assert base == seed.rstrip(), f"{name}: base lost text -> {base!r}"


def test_multi_sentence_seeds_stay_whole(synthetic_task):
    """`split_clauses` cuts on ". ", so a two-sentence seed is the case that can
    break: both sentences belong to the base, neither is a clause."""
    seed = synthetic_task.options["beta"]
    assert seed.count(". ") == 1, seed              # the fixture must stay two-sentence
    base, found = grammar.split_clauses(seed, synthetic_task)
    assert found == []
    assert base == seed


def test_reference_text_contains_no_clauses(real_task):
    """The human arm is scored as a candidate too; if its prose parsed as clauses
    the proposer could 'remove' a sentence the engineer wrote."""
    if real_task.reference is None:
        pytest.skip("task has no reference arm")
    for name, text in real_task.reference.items():
        _base, found = grammar.split_clauses(text, real_task)
        assert found == [], f"{name}: reference text read as clauses {found!r}"


def test_clause_count_is_the_grammar_combinatorics(any_task):
    """Two partnerless templates plus one `prefer` per option, for each
    condition -- asserted from the task's own sizes, not a magic number."""
    expected = len(any_task.conditions) * (len(grammar.TEMPLATES) - 1
                                           + len(any_task.options))
    clauses = grammar.all_clauses(any_task)
    assert set(grammar.TEMPLATES) == {"only", "never", "prefer"}
    assert len(clauses) == expected

    # No two (template, condition, other) triples may collide into one string:
    # a collision would make one clause unremovable from the other's component.
    rendered = [grammar.render_clause(any_task, template, condition.id)
                for template in ("only", "never") for condition in any_task.conditions]
    rendered += [grammar.render_clause(any_task, "prefer", condition.id, other)
                 for condition in any_task.conditions for other in any_task.options]
    assert len(rendered) == len(set(rendered)) == expected
    assert set(rendered) == clauses


def test_apply_clause_is_idempotent(synthetic_task):
    """The proposer may re-pick a clause a component already carries; applying it
    twice must not duplicate the sentence."""
    clause = grammar.render_clause(synthetic_task, "only",
                                   synthetic_task.conditions[0].id)
    once = grammar.apply_clause(synthetic_task.options["alpha"], clause)
    assert grammar.apply_clause(once, clause) == once


def test_options_of_fills_gaps_from_the_seed(any_task):
    """A candidate need not carry every component; the rest fall back to the seed,
    in the task's own option order."""
    full = grammar.components(any_task.options)
    assert grammar.options_of(any_task, full) == any_task.options
    assert grammar.options_of(any_task, {}) == any_task.options

    name = next(iter(any_task.options))
    partial = {grammar.PREFIX + name: "edited"}
    got = grammar.options_of(any_task, partial)
    assert list(got) == list(any_task.options)
    assert got[name] == "edited"
    assert all(got[o] == any_task.options[o] for o in any_task.options if o != name)


def test_questions_is_one_choice_over_every_option(any_task):
    built = grammar.questions(any_task, grammar.components(any_task.options))
    assert set(built) == {grammar.QUESTION}
    question = built[grammar.QUESTION]
    assert question["type"] == "choice"
    assert question["instructions"] == any_task.instructions
    assert list(question["criteria"]) == list(any_task.options)
    for name, text in question["criteria"].items():
        assert isinstance(text, str) and text.strip(), name


def test_render_prompt_shows_instructions_then_every_option(any_task):
    text = grammar.render_prompt(any_task, {})
    lines = text.splitlines()
    assert lines[0] == any_task.instructions
    assert lines[1] == ""
    assert len(lines) == 2 + len(any_task.options)
    for line, (name, seed) in zip(lines[2:], any_task.options.items(), strict=True):
        assert line == f"{name}: {seed}"
