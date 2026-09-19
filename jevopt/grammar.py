"""The thing being optimised: the text that describes each option.

A GEPA candidate is `{"criteria.<option>": "<text>", ...}` -- one component per
option of the task's choice question. The instructions and the option set are
held fixed across every arm, so the experiment isolates one variable: how each
option is described.

Because Jev cannot write text, new wording has to come from somewhere other than
a generative model. It comes from a *grammar*: the task's conditions are turned
into English by three clause templates. Nothing in here encodes which option is
right -- only the vocabulary for saying when an option might apply. Finding the
right binding is the search problem.
"""

from __future__ import annotations

from .task import Task

PREFIX = "criteria."
QUESTION = "decision"          # the question's name in the Jev request/response

TEMPLATES = {
    "only": "Only choose this when {phrase}.",
    "never": "Never choose this when {phrase}.",
    "prefer": "When {phrase}, choose {other} instead of this.",
}

# Rendering needs each condition's phrase, and split_clauses needs every string
# the grammar can produce; both are pure functions of the task and both sit in
# loops hot enough to matter. A Task is a mutable dataclass and so unhashable,
# hence the id-keyed memo -- which also holds the task, so the object stays
# alive and its id cannot be recycled under us.
_memo: dict[int, dict] = {}


def _entry(task: Task) -> dict:
    got = _memo.get(id(task))
    if got is None or got["task"] is not task:
        got = {"task": task,
               "phrases": {c.id: c.phrase for c in task.conditions},
               "clauses": None}
        _memo[id(task)] = got
    return got


def _render(phrases: dict[str, str], template: str, condition_id: str,
            other: str | None = None) -> str:
    return TEMPLATES[template].format(phrase=phrases[condition_id], other=other or "")


def render_clause(task: Task, template: str, condition_id: str,
                  other: str | None = None) -> str:
    return _render(_entry(task)["phrases"], template, condition_id, other)


def all_clauses(task: Task) -> set[str]:
    """Every string the grammar can produce -- lets us find added clauses in a
    component's text, so they can be removed again as well as appended."""
    entry = _entry(task)
    if entry["clauses"] is None:
        phrases = entry["phrases"]
        out = set()
        for condition_id in phrases:
            out.add(_render(phrases, "only", condition_id))
            out.add(_render(phrases, "never", condition_id))
            for other in task.options:
                out.add(_render(phrases, "prefer", condition_id, other))
        entry["clauses"] = out
    return entry["clauses"]


def split_clauses(text: str, task: Task) -> tuple[str, list[str]]:
    """Separate the seed description from clauses the search has appended."""
    known = all_clauses(task)
    parts, base, clauses = [p.strip() for p in text.split(". ")], [], []
    for part in parts:
        sentence = part if part.endswith(".") else part + "."
        (clauses if sentence in known else base).append(sentence)
    return " ".join(base), clauses


def compose(base: str, clauses: list[str]) -> str:
    return " ".join([base] + clauses).strip()


def apply_clause(text: str, clause: str) -> str:
    """Append a clause, keeping the component readable and duplicate-free."""
    text = text.rstrip()
    if clause in text:
        return text
    return f"{text} {clause}"


# ------------------------------------------------------------------- candidates

def components(options: dict[str, str]) -> dict[str, str]:
    return {PREFIX + name: text for name, text in options.items()}


def options_of(task: Task, candidate: dict[str, str]) -> dict[str, str]:
    """Candidate -> the criteria map Jev expects, in the task's option order."""
    return {name: candidate.get(PREFIX + name, seed)
            for name, seed in task.options.items()}


def questions(task: Task, candidate: dict[str, str]) -> dict:
    """The one choice question the optimiser evolves."""
    return {
        QUESTION: {
            "type": "choice",
            "instructions": task.instructions,
            "criteria": options_of(task, candidate),
        }
    }


def render_prompt(task: Task, candidate: dict[str, str]) -> str:
    """The evolved prompt as readable text, for diffing and for the repo."""
    lines = [task.instructions, ""]
    for name, text in options_of(task, candidate).items():
        lines.append(f"{name}: {text}")
    return "\n".join(lines)
