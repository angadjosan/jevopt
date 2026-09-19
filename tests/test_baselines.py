"""What `baselines` refuses to do, and that its two control arms are real.

`baselines` shares `optimize`'s preflight -- the same load_task, check_writable,
check_split and check_failures -- so the same four bugs could ship through it,
and until now nothing here was tested either. The first half of this file pins
those guards on this command; the second half pins what is only true of this
command, that greedy and random are constructed from the evidence gate and
actually scored.

Everything runs offline through `stubjev`, which replaces `jevopt.client.ask`.
"""

from __future__ import annotations

import json

import pytest
from stubjev import always_first, stub_jev

from jevopt import baselines, grammar

# --task takes a dotted path that main() imports, so these task modules are
# written into tmp_path rather than built as fixtures. HEALTHY has three fields
# the grammar can see; COLLAPSED has one identifier field, which
# conditions.MAX_VALUES (8) drops, leaving every state the same signature.

HEALTHY = '''
"""Eight states over three fields the grammar can see -- a hand-sized task."""
import itertools

from jevopt.conditions import derive
from jevopt.task import Task


def build():
    instances = [
        {"state": {"color": color, "size": size, "zone": zone},
         "acceptable": ["alpha"] if color == "red" else ["beta"]}
        for color, size, zone in itertools.product(
            ("red", "blue"), ("small", "large"), ("north", "south"))]
    return Task(name="hand", instructions="Which worker should take the job?",
                options={"alpha": "Give it to alpha.", "beta": "Give it to beta."},
                instances=instances, conditions=derive(instances))
'''

COLLAPSED = '''
"""One identifier field: no conditions, one signature, an unfillable split."""
from jevopt.conditions import derive
from jevopt.task import Task


def build():
    instances = [{"state": {"ticket_id": f"T{n:03d}"}, "acceptable": ["alpha"]}
                 for n in range(12)]
    return Task(name="collapsed", instructions="Which worker?",
                options={"alpha": "Give it to alpha.", "beta": "Give it to beta."},
                instances=instances, conditions=derive(instances))
'''


def write_task(tmp_path, monkeypatch, name: str, source: str) -> str:
    """Put a task module where --task can import it; return its dotted path."""
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


@pytest.fixture
def healthy(tmp_path, monkeypatch) -> str:
    return write_task(tmp_path, monkeypatch, "baseguard_healthy", HEALTHY)


@pytest.fixture
def collapsed(tmp_path, monkeypatch) -> str:
    return write_task(tmp_path, monkeypatch, "baseguard_collapsed", COLLAPSED)


@pytest.fixture
def out_path(tmp_path):
    """The one output path, under tmp_path: no test may write into runs/."""
    return tmp_path / "baselines.json"


def argv(task: str, out_path, *extra: str) -> list[str]:
    return ["--task", task, "--k", "1", "--random-seeds", "2",
            "--out", str(out_path), *extra]


# ------------------------------------------------- the shared spending guards

def test_a_run_whose_calls_all_failed_refuses_to_report_it(healthy, out_path):
    """A failed call scores -1.0, the same as a confident wrong answer, so a
    dead key would otherwise leave a file of 0% arms behind it."""
    with stub_jev(fail=RuntimeError("HTTP 500 boom")), \
            pytest.raises(SystemExit) as exit_info:
        baselines.main(argv(healthy, out_path))

    message = str(exit_info.value.code)
    assert exit_info.value.code != 0
    assert "mostly failed" in message
    assert "HTTP 500 boom" in message            # the first error, quoted
    assert not out_path.exists()


def test_an_empty_split_aborts_before_a_single_call(collapsed, out_path):
    with stub_jev(always_first) as calls, pytest.raises(SystemExit) as exit_info:
        baselines.main(argv(collapsed, out_path))

    message = str(exit_info.value.code)
    assert exit_info.value.code != 0
    assert "the train split is empty" in message
    assert "too many distinct values" in message   # and why it collapsed
    assert calls == []                             # nothing was spent
    assert not out_path.exists()


def test_existing_results_are_not_overwritten_without_force(healthy, out_path):
    out_path.write_text('{"keep": "me"}')

    with stub_jev(always_first) as calls, pytest.raises(SystemExit) as exit_info:
        baselines.main(argv(healthy, out_path))

    assert exit_info.value.code != 0
    assert str(out_path) in str(exit_info.value.code)
    assert out_path.read_text() == '{"keep": "me"}'   # content, not mtime
    assert calls == []                                # refused before spending


def test_force_overwrites_existing_results(healthy, out_path):
    out_path.write_text('{"keep": "me"}')

    with stub_jev(always_first):
        baselines.main(argv(healthy, out_path, "--force"))

    assert json.loads(out_path.read_text())["task"] == "hand"


def test_task_is_required():
    with pytest.raises(SystemExit) as exit_info:
        baselines.main(["--k", "1"])
    assert exit_info.value.code == 2                  # argparse's usage error


# ------------------------------------------------------- the arms themselves

def test_a_healthy_run_scores_greedy_and_every_random_arm(healthy, out_path):
    with stub_jev(always_first) as calls:
        baselines.main(argv(healthy, out_path))

    report = json.loads(out_path.read_text())["report"]
    assert set(report) == {"greedy", "random.s0", "random.s1", "random.mean"}
    for name in ("greedy", "random.s0", "random.s1"):
        arm = report[name]
        assert set(arm["clauses"]) == {"alpha", "beta"}
        assert arm["prompt"].strip()
        for split in ("val", "test"):
            assert arm[split]["n"] > 0
            assert 0.0 <= arm[split]["accuracy"] <= 1.0
    assert calls                                      # the arms were really scored

    averaged = [report[f"random.s{seed}"]["test"]["accuracy"] for seed in (0, 1)]
    assert report["random.mean"]["test"]["accuracy"] == pytest.approx(
        sum(averaged) / len(averaged))


def test_the_reported_call_total_is_the_number_of_calls_made(healthy, out_path,
                                                             capsys):
    with stub_jev(always_first) as calls:
        baselines.main(argv(healthy, out_path))

    written = json.loads(out_path.read_text())
    per_arm = (written["report"]["greedy"]["val"]["n"]
               + written["report"]["greedy"]["test"]["n"])
    assert len(calls) == 3 * per_arm                  # greedy + two random seeds
    assert written["jev_calls"] == len(calls)
    assert written["jev_calls_breakdown"] == {"measurement_evaluations": len(calls),
                                              "total": len(calls)}
    assert f"Jev calls: {len(calls)} total" in capsys.readouterr().out


def test_greedy_attaches_the_strongest_clauses_the_evidence_admits(
        synthetic_task, synthetic_evidence):
    """Greedy is "the top-K sound clauses" and nothing else: if it ever stops
    being that, it is no longer the control the search is measured against."""
    candidate, picks = baselines.greedy_arm(synthetic_task, synthetic_evidence, 2)

    assert any(picks.values())
    for option in synthetic_task.options:
        pool = baselines.sound_clauses(synthetic_task, synthetic_evidence, option)
        assert picks[option] == pool[:2]
        strengths = [row["strength"] for row in picks[option]]
        assert strengths == sorted(strengths, reverse=True)
        for row in picks[option]:
            assert row["clause"] in candidate[grammar.PREFIX + option]


def test_the_random_arm_is_seeded_and_stays_inside_the_sound_pool(
        synthetic_task, synthetic_evidence):
    """Random is the "even the ranking is decoration" control, so it has to be
    a draw from the same admissible pool -- and reproducible from its seed."""
    _first, picks = baselines.random_arm(synthetic_task, synthetic_evidence, 2, 3)
    _again, repeat = baselines.random_arm(synthetic_task, synthetic_evidence, 2, 3)
    assert picks == repeat

    for option, rows in picks.items():
        pool = baselines.sound_clauses(synthetic_task, synthetic_evidence, option)
        assert len(rows) == min(2, len(pool))
        assert all(row in pool for row in rows)
