"""A fake Jev, so the spending commands can be tested without spending.

Every module that talks to the model goes through `jevopt.client.ask`, which
makes it the one place to intercept. That single choke point is why `optimize`,
`baselines` and `compare` can be driven end to end offline -- and why they were
untested before: without this, any test of them costs money and needs a key.

    with stub_jev(lambda state, qs: first_option(qs)):
        optimize.main([...])            # a run that always picks the first option

    with stub_jev(fail=RuntimeError("HTTP 401")):
        optimize.main([...])            # every call raises, as a dead key does
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from jevopt import client


def options_of(questions: dict) -> list[str]:
    """The option ids in whichever choice question this request carries."""
    for question in questions.values():
        if question.get("type") == "choice":
            return list(question["criteria"])
    return []


def answer(chosen: str, options: list[str], confidence: float = 0.9) -> dict:
    """A well-formed decisions response putting most mass on `chosen`."""
    rest = (1.0 - confidence) / max(1, len(options) - 1)
    probs = {o: (confidence if o == chosen else rest) for o in options}
    return {"answers": {"decision": {"type": "choice", "choice": chosen,
                                     "probabilities": probs,
                                     "confidence": confidence}},
            "usage": {"cost": 1e-6}}


def always_first(state: dict, questions: dict) -> dict:
    options = options_of(questions)
    return answer(options[0], options)


def oracle(labels: Callable[[dict], list[str]]):
    """A perfect responder: answers each state with its own first label."""
    def respond(state: dict, questions: dict) -> dict:
        options = options_of(questions)
        good = labels(state) or options
        return answer(good[0], options)
    return respond


@contextlib.contextmanager
def stub_jev(respond: Callable[[dict, dict], dict] | None = None,
             fail: BaseException | None = None):
    """Replace client.ask for the duration. Records calls on `.calls`."""
    calls: list[tuple] = []
    respond = respond or always_first

    def fake_ask(state, questions, model=None, retries=0):
        calls.append((state, questions))
        if fail is not None:
            raise fail
        return respond(state, questions)

    real = client.ask
    client.ask = fake_ask
    try:
        yield calls
    finally:
        client.ask = real
