"""The renderer that once lied quietly.

`report` used to default `--task` to this repo's demo task, so pointing it at any
other run produced a confident, well-formatted document about the wrong options
-- printing "nothing added to the seed criteria" for a run that had added three
clauses. The clause breakdown is the only section that needs the task, which is
why it is the one that goes silently empty. So the tests here are: --task is
required, the committed run really renders its clauses, a task/results mismatch
is *said out loud*, and nothing in an old file can turn into a traceback or an
invented number.

It reads the checked-in artifacts under `runs/` and writes only into tmp_path.
No Jev call is possible from this module -- it is pure reformatting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevopt import report

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RESULTS = str(FIXTURES / "triage.results.json")
COMPARE = str(FIXTURES / "triage.compare.json")
TRIAGE = "jevopt.tasks.triage"


def render(capsys, *argv: str) -> tuple[str, str]:
    report.main(list(argv))
    captured = capsys.readouterr()
    return captured.out, captured.err


def write(tmp_path: Path, name: str, data) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return str(path)


# --------------------------------------------------------------- --task is required

def test_task_is_required(capsys):
    with pytest.raises(SystemExit) as excinfo:
        report.main([RESULTS])

    assert excinfo.value.code == 2
    assert "--task" in capsys.readouterr().err


def test_a_mistyped_task_is_one_line_not_an_importlib_traceback(capsys):
    with pytest.raises(SystemExit) as excinfo:
        report.main([RESULTS, "--task", "jevopt.tasks.no_such_task"])

    assert "cannot load task" in str(excinfo.value.code)


def test_a_module_without_build_is_named_as_such():
    with pytest.raises(SystemExit) as excinfo:
        report.main([RESULTS, "--task", "jevopt.tasks"])

    assert "no build() function" in str(excinfo.value.code)


# ------------------------------------------------- the committed run still renders

def test_the_committed_run_renders_the_clauses_the_search_added(capsys):
    """The regression that matters: this section used to come out empty.

    These are the actual sentences in the recorded run -- if the task
    and the results ever stop lining up, the section silently empties again and
    only real clause text catches it.
    """
    out, err = render(capsys, f"triage={RESULTS}", "--task", TRIAGE)

    learned = out.split("## What the search learned")[1].split("\n## ")[0]
    assert '**page_oncall** — Never choose this when severity is "sev3".' in learned
    assert "**auto_remediate** — Only choose this when the blast radius is small" \
        in learned
    assert "Never choose this when known_runbook is \"no\"." in learned
    assert "**rollback_deploy** — Never choose this when recent_deploy is \"no\"." \
        in learned
    assert "nothing added to the seed criteria" not in learned
    assert "warning" not in err


def test_the_results_table_carries_every_arm_and_the_overfitting_column(capsys):
    out, _err = render(capsys, f"triage={RESULTS}", "--task", TRIAGE)

    assert "| triage | seed | 88 | 56.8% | 89 | 67.4% |" in out
    assert "| triage | GEPA-evolved | 88 | 84.1% | 89 | 87.6% |" in out
    assert "| triage | reference | 88 | 88.6% | 89 | 97.8% |" in out
    assert "**-3.5 pt**" in out           # val - test for the evolved arm


def test_cost_confusions_and_remaining_errors_are_all_populated(capsys):
    out, _err = render(capsys, f"triage={RESULTS}", "--task", TRIAGE)

    assert "6594 Jev calls" in out and "$0.18714" in out
    assert "64 mutations applied" in out
    assert "- create_ticket vs page_oncall: 13" in out
    assert "- **seed**: 29 wrong — chose page_oncall x29" in out


def test_a_bare_path_is_named_after_its_file(capsys):
    out, _err = render(capsys, RESULTS, "--task", TRIAGE)

    assert "| triage.results | seed |" in out


# ------------------------------------------------------ backwards compatibility

def test_an_older_file_without_a_repeat_arm_says_so_instead_of_inventing_one(capsys):
    """The committed JSON predates the seed-scored-twice arm. Saying "no noise
    floor measured" is the only honest option; a number here would be fiction."""
    out, _err = render(capsys, f"triage={RESULTS}", "--task", TRIAGE)

    assert "triage: no noise floor measured" in out
    assert "noise floor +" not in out and "noise floor -" not in out


def test_a_run_that_scored_the_seed_twice_gets_a_measured_floor(tmp_path, capsys):
    path = write(tmp_path, "twice.json", {
        "task": "triage",
        "evolved": {},
        "report": {
            "seed": {"val": {"n": 10, "accuracy": 0.5},
                     "test": {"n": 10, "accuracy": 0.60}},
            "seed (repeat)": {"val": {"n": 10, "accuracy": 0.5},
                              "test": {"n": 10, "accuracy": 0.65}},
            "GEPA-evolved": {"val": {"n": 10, "accuracy": 0.7},
                             "test": {"n": 10, "accuracy": 0.63}},
        },
    })

    out, _err = render(capsys, f"twice={path}", "--task", TRIAGE)

    assert "twice: noise floor +5.0 pt on test" in out
    # 63% is within 5 points of the seed's 60%, so it has not been shown to differ.
    assert "Not shown to differ from the seed: `GEPA-evolved`" in out


def test_missing_numbers_render_as_n_a_rather_than_crashing(tmp_path, capsys):
    path = write(tmp_path, "sparse.json", {
        "task": "triage",
        "evolved": {},
        "report": {"seed": {"val": {}, "test": {"accuracy": "not a number"}}},
    })

    out, _err = render(capsys, f"sparse={path}", "--task", TRIAGE)

    assert "| sparse | seed | n/a | n/a | n/a | n/a |" in out
    assert "n/a Jev calls" in out and "spend n/a" in out


def test_an_evolved_candidate_identical_to_the_seed_is_called_out(tmp_path, capsys):
    path = write(tmp_path, "flat.json", {
        "task": "triage", "evolved": {}, "report": {},
        "evolved_identical_to_seed": True,
    })

    out, _err = render(capsys, f"flat={path}", "--task", TRIAGE)

    assert "the evolved candidate is byte-identical to the seed prompt" in out


# ------------------------------------------------------------- the wrong --task

def test_results_for_another_task_produce_a_visible_warning(tmp_path, capsys):
    """This is the failure mode the default --task used to hide: the clause
    section empties out, and without the warning that reads like a finding."""
    path = write(tmp_path, "other.json", {
        "task": "synthetic",
        "evolved": {"criteria.alpha": "Hand the job to the alpha worker. "
                                      "Never choose this when color is \"blue\"."},
        "report": {"seed": {"val": {"n": 4, "accuracy": 0.5},
                            "test": {"n": 4, "accuracy": 0.5}}},
    })

    out, err = render(capsys, f"other={path}", "--task", TRIAGE)

    assert "warning:" in err
    assert "'synthetic'" in err and "['alpha']" in err
    # The warning must survive into the document itself, not just the terminal.
    assert "> **warning:**" in out
    assert "pass that --task" in out
    # ... precisely because the section it explains is empty.
    assert "nothing added to the seed criteria" in out


def test_a_file_with_no_evolved_candidate_says_there_are_no_clauses(tmp_path, capsys):
    path = write(tmp_path, "bare.json", {"task": "triage", "report": {}})

    out, err = render(capsys, f"bare={path}", "--task", TRIAGE)

    assert "no 'evolved' candidate here" in err
    assert "no 'evolved' candidate here" in out


# --------------------------------------------------------------------- --compare

def test_compare_folds_in_the_pairwise_evidence(capsys):
    out, _err = render(capsys, "--task", TRIAGE, "--compare", f"cmp={COMPARE}")

    assert "## Comparison: cmp" in out
    assert "89 shared test instances" in out
    assert "16 arms, 120 pairs" in out
    assert "### Pairwise (McNemar exact, two-sided)" in out
    assert "| seed vs reference | -29.2 pt | 0 | 26 | 0.0000 |" in out
    assert "**65 of 120 pairs separate at p < 0.05**" in out
    # The unpaired intervals must keep their health warning next to them.
    assert "they are NOT the test" in out


def test_compare_calls_out_the_noise_floor_arm(capsys):
    """`noise_twin` is the control; an arm tied with it has not been shown to work
    -- and here that includes `gepa`, which is the whole point of the section."""
    out, _err = render(capsys, "--task", TRIAGE, "--compare", f"cmp={COMPARE}")

    floor = out.split("**Noise floor:")[1]
    assert floor.startswith(" `noise_twin` at 88.8%.")
    assert "not distinguishable from the floor (7 of the 15 arms tested" in floor
    assert "`gepa`" in floor


def test_a_comparison_without_a_noise_arm_simply_omits_the_callout(tmp_path, capsys):
    path = write(tmp_path, "plain.json", {
        "arms": {"seed": {"accuracy": 0.5}, "gepa": {"accuracy": 0.9}},
        "pairs": [{"a": "seed", "b": "gepa", "mcnemar_p": 0.01,
                   "accuracy_diff": -0.4, "significant": True}],
    })

    out, _err = render(capsys, "--task", TRIAGE, "--compare", f"plain={path}")

    assert "Noise floor:" not in out
    assert "| seed vs gepa |" in out


def test_runs_and_comparisons_render_into_one_document(capsys):
    out, _err = render(capsys, f"triage={RESULTS}", "--task", TRIAGE,
                       "--compare", f"cmp={COMPARE}")

    assert out.index("## Results") < out.index("## What the search learned")
    assert out.index("## What the search learned") < out.index("## Comparison: cmp")


# ---------------------------------------------------------------- unreadable input

@pytest.mark.parametrize("content,fragment", [
    ("{ not json", "Expecting"),
    ("[1, 2]", "expected a JSON object at the top level"),
])
def test_a_malformed_file_is_skipped_with_a_message(tmp_path, capsys,
                                                    content, fragment):
    bad = write(tmp_path, "bad.json", content)

    out, err = render(capsys, f"triage={RESULTS}", f"bad={bad}", "--task", TRIAGE)

    assert f"skipping bad={bad}" in err
    assert fragment in err
    assert "## Results" in out          # the good file still rendered


def test_a_missing_file_is_skipped_with_a_message(tmp_path, capsys):
    gone = str(tmp_path / "nope.json")

    out, err = render(capsys, f"triage={RESULTS}", f"gone={gone}", "--task", TRIAGE)

    assert f"skipping gone={gone}" in err
    assert "No such file" in err
    assert "| triage | seed |" in out


def test_nothing_readable_at_all_exits_non_zero(tmp_path, capsys):
    gone = str(tmp_path / "nope.json")

    with pytest.raises(SystemExit) as excinfo:
        report.main([f"gone={gone}", "--task", TRIAGE])

    assert excinfo.value.code == 1
    assert "no readable results or comparison files" in capsys.readouterr().err


def test_no_files_at_all_exits_non_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        report.main(["--task", TRIAGE])

    assert excinfo.value.code == 1


# ------------------------------------------------------------------ --out/--force

def test_out_writes_markdown_and_keeps_stdout_clean(tmp_path, capsys):
    out_path = tmp_path / "report.md"

    stdout, err = render(capsys, f"triage={RESULTS}", "--task", TRIAGE,
                         "--out", str(out_path))

    assert stdout == ""
    assert f"markdown -> {out_path}" in err
    text = out_path.read_text()
    assert text.startswith("# jevopt report — task `triage`")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "## What the search learned" in text


def test_out_refuses_to_clobber_an_existing_file(tmp_path, capsys):
    out_path = tmp_path / "report.md"
    out_path.write_text("published evidence")

    with pytest.raises(SystemExit) as excinfo:
        report.main([f"triage={RESULTS}", "--task", TRIAGE, "--out", str(out_path)])

    assert "refusing to overwrite" in str(excinfo.value.code)
    assert out_path.read_text() == "published evidence"


def test_force_replaces_it(tmp_path, capsys):
    out_path = tmp_path / "report.md"
    out_path.write_text("published evidence")

    render(capsys, f"triage={RESULTS}", "--task", TRIAGE,
           "--out", str(out_path), "--force")

    assert out_path.read_text().startswith("# jevopt report")


def test_the_refusal_happens_before_any_file_is_read(tmp_path, capsys):
    """check_writable runs first, so a bad --out does not half-render anything."""
    out_path = tmp_path / "report.md"
    out_path.write_text("published evidence")

    with pytest.raises(SystemExit):
        report.main([f"triage={RESULTS}", "--task", TRIAGE, "--out", str(out_path)])

    assert capsys.readouterr().out == ""
