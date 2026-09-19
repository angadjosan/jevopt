"""The transport, pinned offline: what goes on the wire, and what retries.

`client.ask` is the one function in the package that spends money, so the two
things worth guarding are the shape of the request (wrong URL, missing bearer
token or a mangled body all look identical from the outside until a key is
burned) and the retry policy. A permanent 400 that retried five times would be
five times the failure and five times the bill, so the call count is asserted,
not just the exception. `urlopen` is monkeypatched throughout: no test here
touches the network or needs a key.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from jevopt import client

OK_BODY = {"answers": {"decision": {"type": "choice", "choice": "a"}}}
QUESTIONS = {"decision": {"type": "choice", "instructions": "pick",
                          "criteria": {"a": "first", "b": "second"}}}


@pytest.fixture(autouse=True)
def no_wall_clock(monkeypatch):
    """Backoff must cost nothing in a test. Records the delays for assertions."""
    slept: list[float] = []
    monkeypatch.setattr(client.time, "sleep", slept.append)
    return slept


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _responses(monkeypatch, *outcomes):
    """Serve `outcomes` in order; each is a dict to return or an exception factory.

    Returns (requests, calls): the Request objects seen, and a one-item counter.
    """
    seen: list[urllib.request.Request] = []
    remaining = list(outcomes)

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        outcome = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if callable(outcome):
            raise outcome()
        return io.BytesIO(json.dumps(outcome).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def http_error(code: int, body: str = "nope"):
    """A fresh HTTPError per raise -- `exc.read()` drains its fp exactly once."""
    return lambda: urllib.error.HTTPError(
        "https://openrouter.ai/api/alpha/decisions", code, "err", {},
        io.BytesIO(body.encode()))


# ------------------------------------------------------------------- the request

def test_ask_posts_json_to_the_decisions_endpoint(monkeypatch, key):
    seen = _responses(monkeypatch, OK_BODY)

    assert client.ask("a ticket", QUESTIONS) == OK_BODY

    (request,) = seen
    assert request.full_url == client.API_ROOT + "/alpha/decisions"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"model": client.MODEL,
                                        "state": "a ticket",
                                        "questions": QUESTIONS}


def test_ask_carries_the_bearer_token_and_json_content_type(monkeypatch, key):
    seen = _responses(monkeypatch, OK_BODY)

    client.ask("s", QUESTIONS)

    # Request lowercases header names it was handed as capitalised keys.
    headers = {k.lower(): v for k, v in seen[0].header_items()}
    assert headers["Authorization".lower()] == "Bearer sk-or-test"
    assert headers["Content-type".lower()] == "application/json"


def test_ask_sends_the_model_it_was_given(monkeypatch, key):
    seen = _responses(monkeypatch, OK_BODY)

    client.ask("s", QUESTIONS, model="typesafe/jev-other")

    assert json.loads(seen[0].data)["model"] == "typesafe/jev-other"


# --------------------------------------------------------------------- the key

def test_api_key_prefers_openrouter_over_openrouter_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER", "first")
    monkeypatch.setenv("OPENROUTER_API_KEY", "second")
    assert client.api_key() == "first"


def test_api_key_falls_back_to_openrouter_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "second")
    assert client.api_key() == "second"


def test_api_key_ignores_an_empty_variable(monkeypatch):
    monkeypatch.setenv("OPENROUTER", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "second")
    assert client.api_key() == "second"


def test_api_key_says_which_variables_to_set_when_there_is_none(monkeypatch):
    monkeypatch.delenv("OPENROUTER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(client.JevError) as excinfo:
        client.api_key()

    message = str(excinfo.value)
    assert "No API key found" in message
    assert "OPENROUTER" in message and "OPENROUTER_API_KEY" in message


def test_ask_refuses_to_open_a_socket_without_a_key(monkeypatch):
    """The key is read while building the request, so nothing is sent."""
    monkeypatch.delenv("OPENROUTER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = _responses(monkeypatch, OK_BODY)

    with pytest.raises(client.JevError, match="No API key found"):
        client.ask("s", QUESTIONS)

    assert seen == []


# ------------------------------------------------------------------- retrying

@pytest.mark.parametrize("code", [429, 529])
def test_ask_retries_a_throttle_until_retries_is_spent(monkeypatch, key,
                                                       no_wall_clock, code):
    seen = _responses(monkeypatch, http_error(code, "slow down"))

    with pytest.raises(client.JevError) as excinfo:
        client.ask("s", QUESTIONS, retries=3)

    assert len(seen) == 4                      # the first try plus three retries
    assert no_wall_clock == [1.0, 2.0, 4.0]    # exponential, not a busy loop
    assert excinfo.value.code == code
    assert excinfo.value.body == "slow down"


def test_ask_returns_the_answer_when_a_retry_succeeds(monkeypatch, key,
                                                      no_wall_clock):
    seen = _responses(monkeypatch, http_error(429), OK_BODY)

    assert client.ask("s", QUESTIONS, retries=2) == OK_BODY
    assert len(seen) == 2
    assert no_wall_clock == [1.0]


def test_ask_does_not_retry_a_permanent_error(monkeypatch, key, no_wall_clock):
    """A 400 is a bug in the request; retrying it four times is four bills."""
    seen = _responses(monkeypatch, http_error(400, '{"error":{"message":"bad"}}'))

    with pytest.raises(client.JevError) as excinfo:
        client.ask("s", QUESTIONS, retries=4)

    assert len(seen) == 1
    assert no_wall_clock == []
    assert excinfo.value.code == 400
    assert excinfo.value.body == '{"error":{"message":"bad"}}'
    assert "HTTP 400" in str(excinfo.value)


@pytest.mark.parametrize("code", [401, 402, 404, 500])
def test_ask_does_not_retry_any_non_throttle_status(monkeypatch, key,
                                                    no_wall_clock, code):
    seen = _responses(monkeypatch, http_error(code))

    with pytest.raises(client.JevError):
        client.ask("s", QUESTIONS, retries=4)

    assert len(seen) == 1


def test_ask_with_zero_retries_gives_up_on_the_first_throttle(monkeypatch, key,
                                                              no_wall_clock):
    """`ask --check` and the ask CLI pass retries=0 so the shell fails fast."""
    seen = _responses(monkeypatch, http_error(429))

    with pytest.raises(client.JevError):
        client.ask("s", QUESTIONS, retries=0)

    assert len(seen) == 1
    assert no_wall_clock == []


def test_ask_retries_an_unreachable_api_then_names_it(monkeypatch, key,
                                                      no_wall_clock):
    seen = _responses(monkeypatch,
                      lambda: urllib.error.URLError("connection refused"))

    with pytest.raises(client.JevError) as excinfo:
        client.ask("s", QUESTIONS, retries=2)

    assert len(seen) == 3
    assert client.API_ROOT in str(excinfo.value)
    assert excinfo.value.code is None          # nothing HTTP to report


def test_jev_error_defaults_to_no_code_and_an_empty_body():
    error = client.JevError("something went wrong")
    assert (error.code, error.body) == (None, "")
    assert isinstance(error, RuntimeError)
