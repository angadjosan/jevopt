"""What `optimize` refuses to do, and what it must report when it does run.

This is the module that spends money, and nothing tested it: every guard below
is a bug that shipped and was caught by hand -- a dead key reported as a table
of results, an empty split that logged forever instead of stopping, a run that
quietly overwrote published evidence, a call total that disagreed with itself.
Each test pins one of them by name, so a failure says which one came back.

Nothing here touches the network. `stubjev` replaces `jevopt.client.ask`, the
one call every module goes through, so a complete run costs nothing and takes
milliseconds -- which is also why these tests can exist at all.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from types import SimpleNamespace

import pytest
from stubjev import always_first, stub_jev

from jevopt import optimize

# --------------------------------------------------- task modules on disk
#
# --task takes a dotted path that main() imports, so a Task built in a fixture
# cannot be reached. Each source below is one shape of task a guard cares about;
# they are written into tmp_path and imported from there.

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
"""One field, a distinct value per state. conditions.MAX_VALUES is 8, so the
field is read as an identifier and dropped: every state ends up with the same
(empty) signature, and split() cannot fill three parts from one signature."""
from jevopt.conditions import derive
from jevopt.task import Task


def build():
    instances = [{"state": {"ticket_id": f"T{n:03d}"}, "acceptable": ["alpha"]}
                 for n in range(12)]
    return Task(name="collapsed", instructions="Which worker?",
                options={"alpha": "Give it to alpha.", "beta": "Give it to beta."},
                instances=instances, conditions=derive(instances))
'''

CONFLICTING = '''
"""Two states per signature, labelled differently: the field the decision turns
on ("tie") is not in the state at all, so no wording can tell them apart."""
import itertools

from jevopt.conditions import derive
from jevopt.task import Task


def build():
    instances = []
    for color, size in itertools.product(("red", "blue"), ("small", "large")):
        for tie in ("alpha", "beta"):
            instances.append({"state": {"color": color, "size": size},
                              "acceptable": ["alpha"] if color == "red" else [tie]})
    return Task(name="conflicting", instructions="Which worker?",
                options={"alpha": "Give it to alpha.", "beta": "Give it to beta."},
                instances=instances, conditions=derive(instances))
'''

NO_BUILD = '''
"""A module that imports cleanly and is not a task."""
OPTIONS = {"alpha": "Give it to alpha."}
'''


def write_task(tmp_path, monkeypatch, name: str, source: str) -> str:
    """Put a task module where --task can import it; return its dotted path."""
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


@pytest.fixture
def healthy(tmp_path, monkeypatch) -> str:
    return write_task(tmp_path, monkeypatch, "optguard_healthy", HEALTHY)


@pytest.fixture
def collapsed(tmp_path, monkeypatch) -> str:
    return write_task(tmp_path, monkeypatch, "optguard_collapsed", COLLAPSED)


@pytest.fixture
def conflicting(tmp_path, monkeypatch) -> str:
    return write_task(tmp_path, monkeypatch, "optguard_conflicting", CONFLICTING)


@pytest.fixture
def no_build(tmp_path, monkeypatch) -> str:
    return write_task(tmp_path, monkeypatch, "optguard_no_build", NO_BUILD)


@pytest.fixture
def outputs(tmp_path):
    """Both output paths, under tmp_path: no test may write into runs/."""
    return tmp_path / "prompt.txt", tmp_path / "results.json"


def argv(task: str, outputs, *extra: str) -> list[str]:
    out, results = outputs
    return ["--task", task, "--budget", "6", "--minibatch", "4", "--quiet",
            "--out", str(out), "--results", str(results), *extra]


@pytest.fixture
def finished_run(healthy, outputs, capsys):
    """One complete stubbed run: its JSON, its console output, its calls."""
    out, results = outputs
    with stub_jev(always_first) as calls:
        optimize.main(argv(healthy, outputs))
    return SimpleNamespace(results=json.loads(results.read_text()),
                           prompt=out.read_text(),
                           console=capsys.readouterr(),
                           calls=list(calls))


# ------------------------------------- 1. failed calls are refused, not reported

def test_a_run_whose_calls_all_failed_refuses_to_report_it(healthy, outputs):
    """A failed call scores -1.0, exactly like a confident wrong answer, so a
    dead key used to print a full table and exit 0."""
    out, results = outputs
    with stub_jev(fail=RuntimeError("HTTP 401 no key")), \
            pytest.raises(SystemExit) as exit_info:
        optimize.main(argv(healthy, outputs))

    message = str(exit_info.value.code)
    assert exit_info.value.code != 0
    assert "mostly failed" in message
    assert "HTTP 401 no key" in message          # the first error, quoted
    assert not out.exists()
    assert not results.exists()


def test_a_healthy_run_writes_both_files_and_exits_zero(healthy, outputs):
    out, results = outputs
    with stub_jev(always_first):
        optimize.main(argv(healthy, outputs))    # no SystemExit == exit 0

    assert out.read_text().strip()
    assert json.loads(results.read_text())["task"] == "hand"


def _fails_once():
    """A responder that raises on its first call only -- a 429 that outlived
    its retries, which must not be mistaken for a broken key."""
    lock, counter = threading.Lock(), itertools.count()

    def respond(state, questions):
        with lock:
            first = next(counter) == 0
        if first:
            raise RuntimeError("HTTP 429 slow down")
        return always_first(state, questions)

    return respond


def test_a_single_failed_call_is_forgiven_and_still_counted(healthy, outputs, capsys):
    out, results = outputs
    with stub_jev(_fails_once()):
        optimize.main(argv(healthy, outputs))

    assert out.exists()
    assert json.loads(results.read_text())["jev_call_failures"] == 1
    assert "1 of" in capsys.readouterr().err     # the count is always announced


# --------------------------------------------- 2. an empty split aborts at once

def test_an_empty_split_aborts_before_a_single_call(collapsed, outputs):
    """GEPA swallows the AssertionError an empty valset raises and a failed
    iteration spends no budget, so this used to log until it was killed."""
    out, results = outputs
    started = time.perf_counter()
    with stub_jev(always_first) as calls, pytest.raises(SystemExit) as exit_info:
        optimize.main(argv(collapsed, outputs))
    elapsed = time.perf_counter() - started

    message = str(exit_info.value.code)
    assert exit_info.value.code != 0
    assert "the train split is empty" in message
    assert "too many distinct values" in message   # and why it collapsed
    assert calls == []                             # nothing was spent
    assert not out.exists()
    assert not results.exists()
    assert elapsed < 5.0                           # it stops, it does not loop


def test_check_split_passes_when_every_part_is_filled(synthetic_task):
    train, val, test = synthetic_task.split(seed=0)
    optimize.check_split(synthetic_task, {"train": train, "val": val, "test": test})


def test_check_split_names_the_part_that_is_empty(synthetic_task):
    train, _val, test = synthetic_task.split(seed=0)
    with pytest.raises(SystemExit) as exit_info:
        optimize.check_split(synthetic_task,
                             {"train": train, "val": [], "test": test})
    assert "the val split is empty" in str(exit_info.value.code)


# ------------------------------------------ 3. contradictory training data warns

def test_contradictory_training_data_is_warned_about(conflicting, outputs, capsys):
    with stub_jev(always_first):
        optimize.main(argv(conflicting, outputs))

    warning = capsys.readouterr().err
    assert "contradictory training data" in warning
    assert "alpha vs beta" in warning            # the labels that disagree
    assert '"color": "blue"' in warning          # and a state to go and look at


def test_a_task_its_conditions_can_explain_warns_about_nothing(synthetic_task, capsys):
    optimize.warn_conflicts(synthetic_task)
    assert capsys.readouterr().err == ""


# ------------------------------------------------- 4. --force protects the files

def test_existing_output_is_not_overwritten_without_force(healthy, outputs):
    """The default paths are runs/<task>.*, where this repo's published
    evidence lives, so the quickstart as written used to destroy it."""
    out, results = outputs
    out.write_text("PREVIOUS PROMPT")
    results.write_text('{"keep": "me"}')

    with stub_jev(always_first) as calls, pytest.raises(SystemExit) as exit_info:
        optimize.main(argv(healthy, outputs))

    message = str(exit_info.value.code)
    assert exit_info.value.code != 0
    assert str(out) in message
    assert str(results) in message
    assert out.read_text() == "PREVIOUS PROMPT"      # content, not mtime
    assert results.read_text() == '{"keep": "me"}'
    assert calls == []                               # refused before spending


def test_force_overwrites_existing_output(healthy, outputs):
    out, results = outputs
    out.write_text("PREVIOUS PROMPT")
    results.write_text('{"keep": "me"}')

    with stub_jev(always_first):
        optimize.main(argv(healthy, outputs, "--force"))

    assert "PREVIOUS PROMPT" not in out.read_text()
    assert json.loads(results.read_text())["task"] == "hand"


# ------------------------------------------- 5. --task is required and specific

def test_task_is_required():
    """It used to default to the demo task, so a forgotten --task bought a
    confident report about somebody else's data."""
    with pytest.raises(SystemExit) as exit_info:
        optimize.main(["--budget", "6", "--quiet"])
    assert exit_info.value.code == 2                 # argparse's usage error


def test_an_unimportable_task_is_one_line_not_a_traceback(outputs):
    with pytest.raises(SystemExit) as exit_info:
        optimize.main(argv("no.such.task.module", outputs))

    message = str(exit_info.value.code)
    assert "cannot load task 'no.such.task.module'" in message
    assert "ModuleNotFoundError" in message
    assert "Traceback" not in message
    assert len(message.splitlines()) <= 3


def test_a_task_module_without_build_says_exactly_that(no_build, outputs):
    with pytest.raises(SystemExit) as exit_info:
        optimize.main(argv(no_build, outputs))

    message = str(exit_info.value.code)
    assert "has no build() function" in message
    assert "Traceback" not in message
    assert len(message.splitlines()) <= 3


# --------------------------------------------------- 6. the noise-floor machinery

def test_the_results_json_records_the_noise_floor_machinery(finished_run):
    results = finished_run.results
    assert set(results) >= {"noise_floor", "majority_option",
                            "evolved_identical_to_seed"}
    assert set(results["noise_floor"]) == {"val", "test"}
    # the recorded option is the one the free majority arm actually answered
    assert (results["report"]["majority"]["val"]["always_answers"]
            == results["majority_option"])


def test_scoring_one_prompt_twice_puts_the_noise_floor_at_zero(finished_run):
    """The stub is deterministic, so the seed and its repeat are the same
    measurement: any gap here is the arithmetic, not the model."""
    report = finished_run.results["report"]
    assert finished_run.results["noise_floor"] == {"val": 0.0, "test": 0.0}
    assert report["seed"] == report["seed (repeat)"]
    assert "0.0 pt on val and 0.0 pt on test" in finished_run.console.out


def test_an_evolved_candidate_equal_to_the_seed_is_not_a_second_arm(finished_run):
    """Two scores for one prompt are two samples of the same thing; tabling
    them as rival arms is how a 4.8-point "regression" got reported."""
    results = finished_run.results
    assert results["evolved_identical_to_seed"] is True
    assert "GEPA-evolved" not in results["report"]
    assert "BYTE-IDENTICAL to the seed" in finished_run.console.out


# ----------------------------------------------- 7. the call total reconciles

def test_the_reported_call_total_is_the_sum_of_its_parts(finished_run):
    breakdown = finished_run.results["jev_calls_breakdown"]
    parts = ("search_evaluations", "proposer_repair_calls",
             "measurement_evaluations")
    assert sum(breakdown[part] for part in parts) == breakdown["total"]
    assert breakdown["total"] == finished_run.results["jev_calls"]


def test_the_reported_call_total_is_the_number_of_calls_made(finished_run):
    """The console said 88 and the JSON said 619 for one run: the proposer's
    repair calls reached no counter at all."""
    total = finished_run.results["jev_calls"]
    assert total == len(finished_run.calls)
    assert f"Jev calls: {total} total" in finished_run.console.out


def test_a_full_triage_run_reconciles_its_call_total(outputs, capsys):
    """The one test on the real task: six hundred-odd calls, all of them stubbed."""
    out, results = outputs
    with stub_jev(always_first) as calls:
        optimize.main(["--task", "jevopt.tasks.triage", "--budget", "6", "--quiet",
                       "--out", str(out), "--results", str(results)])

    written = json.loads(results.read_text())
    assert written["task"] == "triage"
    assert len(calls) > 500                      # a real run, not a toy one
    assert written["jev_calls"] == len(calls)
    assert "reference" in written["report"]      # triage carries a human arm
    assert f"Jev calls: {len(calls)} total" in capsys.readouterr().out
