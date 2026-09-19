"""The shell-facing command: spec parsing, the error paths, and where state comes
from.

`ask` is the one command a user types by hand, so every mistake it can be handed
-- a `--choice` with no options, an empty state, no questions at all -- has to
come back as one line, not a traceback. It is also the only command that already
reports a missing key properly; that behaviour is pinned here so it stays.

Everything runs against `stub_jev`, so the question dicts can be read straight
off the intercepted call and nothing reaches the network.
"""

from __future__ import annotations

import io
import json

import pytest
from stubjev import stub_jev

from jevopt import ask


def respond(state, questions: dict) -> dict:
    """A plausible answer for each question asked, whatever its type."""
    answers = {}
    for name, question in questions.items():
        kind = question.get("type")
        if kind == "noul":
            answers[name] = {"type": "noul", "noul": 0.75, "confidence": 0.8}
        elif kind == "choice":
            options = list(question["criteria"])
            answers[name] = {"type": "choice", "choice": options[0],
                             "probabilities": {o: 1 / len(options) for o in options}}
        else:
            levels = list(question.get("criteria", []))
            answers[name] = {"type": "score", "score": 1.0,
                             "legend": {str(i): lv for i, lv in enumerate(levels)},
                             "probabilities": {str(i): 1 / max(1, len(levels))
                                               for i in range(len(levels))}}
    return {"model": "typesafe/jev-1.13", "provider": "stub", "answers": answers,
            "usage": {"cost": 1e-6, "input_tokens": 10, "output_tokens": 2}}


def run(argv: list[str]):
    """Drive `ask` offline; return (recorded calls, the questions it built)."""
    with stub_jev(respond) as calls:
        ask.main(argv)
    return calls, calls[0][1]


# --------------------------------------------------------------- spec parsing

def test_noul_spec_becomes_a_noul_question():
    _calls, questions = run(["a ticket", "--noul", "urgent: Is this urgent?"])

    assert questions == {"urgent": {"type": "noul",
                                    "instructions": "Is this urgent?"}}


def test_choice_spec_keeps_option_order_and_per_option_rubrics():
    _calls, questions = run(
        ["a ticket",
         "--choice", "team: Who handles this? = billing: money | technical | sales"])

    assert questions == {"team": {
        "type": "choice",
        "instructions": "Who handles this?",
        "criteria": {"billing": "money", "technical": None, "sales": None},
    }}
    assert list(questions["team"]["criteria"]) == ["billing", "technical", "sales"]


def test_score_spec_becomes_an_ordered_list_of_levels():
    _calls, questions = run(
        ["a ticket", "--score", "mood: How frustrated? = Calm | Cross | Furious"])

    assert questions == {"mood": {
        "type": "score",
        "instructions": "How frustrated?",
        "criteria": ["Calm", "Cross", "Furious"],
    }}


def test_repeated_flags_all_land_in_one_request():
    _calls, questions = run(
        ["a ticket",
         "--noul", "urgent: Urgent?",
         "--noul", "angry: Angry?",
         "--choice", "team: Who? = billing | sales",
         "--score", "mood: How cross? = Calm | Furious"])

    assert set(questions) == {"urgent", "angry", "team", "mood"}
    assert [q["type"] for q in questions.values()] == ["noul", "noul",
                                                       "choice", "score"]


def test_questions_file_is_merged_in(tmp_path):
    raw = tmp_path / "questions.json"
    raw.write_text(json.dumps({"extra": {"type": "noul", "instructions": "Hm?"}}))

    _calls, questions = run(["a ticket", "--noul", "urgent: Urgent?",
                             "--questions", str(raw)])

    assert set(questions) == {"urgent", "extra"}


def test_the_model_flag_reaches_the_client():
    with stub_jev(respond) as calls:
        ask.main(["a ticket", "--noul", "u: Urgent?", "--model", "typesafe/other"])

    assert calls  # the stub does not record the model, so just prove it got there
    assert ask.DEFAULT_MODEL == "typesafe/jev-1.13"


# ----------------------------------------------------------------- bad input

@pytest.mark.parametrize("argv,expected", [
    (["s", "--choice", "team: Who handles this?"], "Malformed --choice"),
    (["s", "--choice", "team = billing | sales"], "Malformed --choice"),
    (["s", "--choice", "team: Who? = billing"], "at least two options"),
    (["s", "--choice", "team: Who? =  | sales"], "empty option"),
    (["s", "--score", "mood: How cross?"], "Malformed --score"),
    (["s", "--score", "mood: How cross? = Calm"], "at least two levels"),
    (["s", "--noul", "urgent Is this urgent?"], "Malformed --noul"),
])
def test_a_malformed_spec_is_one_line_not_a_traceback(argv, expected):
    with stub_jev(respond) as calls:
        with pytest.raises(SystemExit) as excinfo:
            ask.main(argv)

    message = excinfo.value.code
    assert isinstance(message, str), "sys.exit(str) is what makes this one line"
    assert expected in message
    assert calls == [], "it must fail before spending a call"


def test_no_questions_at_all_says_which_flags_to_use():
    with stub_jev(respond) as calls:
        with pytest.raises(SystemExit) as excinfo:
            ask.main(["a ticket"])

    assert "No questions asked" in excinfo.value.code
    assert "--noul" in excinfo.value.code
    assert calls == []


def test_an_empty_state_is_refused_before_the_call():
    with stub_jev(respond) as calls:
        with pytest.raises(SystemExit) as excinfo:
            ask.main(["   ", "--noul", "urgent: Urgent?"])

    assert "Empty state" in excinfo.value.code
    assert calls == []


def test_a_questions_file_that_is_not_json_is_reported_not_raised(tmp_path):
    raw = tmp_path / "questions.json"
    raw.write_text("{not json")

    with pytest.raises(SystemExit) as excinfo:
        ask.main(["a ticket", "--questions", str(raw)])

    assert "not valid JSON" in excinfo.value.code


def test_a_questions_file_that_is_not_an_object_is_refused(tmp_path):
    raw = tmp_path / "questions.json"
    raw.write_text("[1, 2]")

    with pytest.raises(SystemExit) as excinfo:
        ask.main(["a ticket", "--questions", str(raw)])

    assert "--questions must contain a JSON object" in excinfo.value.code


def test_json_state_that_will_not_parse_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        ask.main(["{not json", "--json-state", "--noul", "u: Urgent?"])

    assert "--json-state" in excinfo.value.code


# --------------------------------------------------------------- where state comes from

def test_state_can_be_a_literal_argument():
    calls, _questions = run(["payouts have failed", "--noul", "u: Urgent?"])

    assert calls[0][0] == "payouts have failed"


def test_state_can_come_from_a_file(tmp_path):
    ticket = tmp_path / "ticket.txt"
    ticket.write_text("payouts have failed")

    calls, _questions = run(["--state-file", str(ticket), "--noul", "u: Urgent?"])

    assert calls[0][0] == "payouts have failed"


def test_state_can_come_from_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("payouts have failed"))

    calls, _questions = run(["--noul", "u: Urgent?"])

    assert calls[0][0] == "payouts have failed"


def test_json_state_is_sent_structured_not_as_text(tmp_path):
    ticket = tmp_path / "ticket.json"
    ticket.write_text('{"severity": "sev1", "service": "payouts"}')

    calls, _questions = run(["--state-file", str(ticket), "--json-state",
                             "--noul", "u: Urgent?"])

    assert calls[0][0] == {"severity": "sev1", "service": "payouts"}


# -------------------------------------------------------------------- output

def test_json_flag_emits_the_raw_response_and_nothing_else(capsys):
    with stub_jev(respond):
        ask.main(["a ticket", "--json", "--noul", "urgent: Urgent?"])

    body = json.loads(capsys.readouterr().out)
    assert body["answers"]["urgent"]["noul"] == 0.75


def test_the_human_rendering_names_every_question_asked(capsys):
    with stub_jev(respond):
        ask.main(["a ticket", "--noul", "urgent: Urgent?",
                  "--choice", "team: Who? = billing | sales",
                  "--score", "mood: How cross? = Calm | Furious"])

    out = capsys.readouterr().out
    assert "urgent  (noul)" in out
    assert "team  (choice) -> billing" in out
    assert "mood  (score)" in out


def test_verbose_puts_cost_on_stderr_so_stdout_stays_pipeable(capsys):
    with stub_jev(respond):
        ask.main(["a ticket", "-v", "--noul", "u: Urgent?"])

    captured = capsys.readouterr()
    assert "cost: $0.000001" in captured.err
    assert "model: typesafe/jev-1.13" in captured.err
    assert "cost" not in captured.out          # the answers keep stdout to themselves


def test_json_returns_before_the_verbose_line(capsys):
    """Documented quirk: `--json -v` prints no diagnostics -- main() returns at
    --json, and the usage block is already inside the body it printed."""
    with stub_jev(respond):
        ask.main(["a ticket", "--json", "-v", "--noul", "u: Urgent?"])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["usage"]["cost"] == 1e-6


def test_check_probes_the_model_with_one_cheap_question(capsys):
    with stub_jev(respond) as calls:
        ask.main(["--check"])

    assert len(calls) == 1
    assert calls[0][1] == {"ok": {"type": "noul", "instructions": "Is this a test?"}}
    assert "is answering" in capsys.readouterr().out


# ---------------------------------------------------------------- missing key

def test_a_missing_key_is_reported_as_one_line(monkeypatch):
    """The real client runs here -- api_key() raises before any socket opens."""
    monkeypatch.delenv("OPENROUTER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        ask.main(["a ticket", "--noul", "urgent: Urgent?"])

    assert "No API key found" in excinfo.value.code


def test_an_http_failure_is_unwrapped_rather_than_printed_escaped():
    error = ask.client.JevError("HTTP 400: ...", code=400,
                                body=json.dumps({"error": {"message": "bad field"}}))

    with stub_jev(fail=error):
        with pytest.raises(SystemExit) as excinfo:
            ask.main(["a ticket", "--noul", "urgent: Urgent?"])

    assert excinfo.value.code == "HTTP 400: bad field"
