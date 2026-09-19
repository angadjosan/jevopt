"""Build the condition vocabulary from the states themselves.

A task should not have to hand-write its grammar. Every leaf field in the state
whose value set is small enough to enumerate becomes one condition per value,
phrased with the field's own name -- which is exactly what Jev reads in the
state, so referring to it literally is unambiguous.

Tasks can add their own conditions for groupings the data cannot suggest, such
as treating "slightly to the left" and "far to the left" as one fact.
"""

from __future__ import annotations

from .task import Condition

MAX_VALUES = 8            # a field with more distinct values is an identifier
SKIP_SUFFIXES = ("step",)

# Why a field yielded no conditions. `diagnose` reports these verbatim, so they
# are named once here rather than spelled out at each use.
TOO_MANY = "too many distinct values"
CONSTANT = "constant"
SKIPPED = "skipped name"


def flatten(state: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in state.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, prefix=f"{path}."))
        elif isinstance(value, (str, bool, int, float)) and not isinstance(value, bool):
            out[path] = value
        elif isinstance(value, bool):
            out[path] = value
    return out


def _at(path: str):
    parts = path.split(".")

    def get(state: dict):
        node = state
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node
    return get


def _scan(instances: list[dict]) -> dict[str, set]:
    """Every flattened field across the states, with the values it actually takes.

    Skipped names are collected too: `diagnose` has to be able to say how wide a
    field it refused was.
    """
    seen: dict[str, set] = {}
    for instance in instances:
        for path, value in flatten(instance["state"]).items():
            seen.setdefault(path, set()).add(value)
    return seen


def _rejection(path: str, values: set, max_values: int) -> str | None:
    """Why this field yields no conditions, or None if it yields some.

    `derive` and `diagnose` both decide through here. A warning that disagreed
    with what was actually built would be worse than no warning at all.
    """
    if path.endswith(SKIP_SUFFIXES):
        return SKIPPED
    if len(values) < 2:
        return CONSTANT               # a constant field says nothing
    if len(values) > max_values:
        return TOO_MANY               # wide ones are identifiers, not facts
    return None


def derive(instances: list[dict], extra: list[Condition] | None = None,
           max_values: int = MAX_VALUES) -> list[Condition]:
    """One equality condition per (field, value) the states actually contain."""
    conditions: list[Condition] = []
    for path, values in sorted(_scan(instances).items()):
        if _rejection(path, values, max_values) is not None:
            continue
        getter = _at(path)
        for value in sorted(values, key=str):
            conditions.append(Condition(
                id=f"{path}={value}",
                phrase=f'{path} is "{value}"',
                test=(lambda g, v: (lambda s: g(s) == v))(getter, value),
            ))
    conditions.extend(extra or [])
    return conditions


def diagnose(instances: list[dict], max_values: int = MAX_VALUES) -> dict:
    """Which state fields became conditions, and which were discarded, and why.

    `derive` returning a short list looks the same whether the states are simple
    or whether the one field the decision turns on was thrown away for having
    too many values. Nothing downstream can tell those apart, so the field scan
    is reported here as data for a caller to warn about.
    """
    kept: dict[str, int] = {}
    dropped: dict[str, tuple[int, str]] = {}
    for path, values in sorted(_scan(instances).items()):
        reason = _rejection(path, values, max_values)
        if reason is None:
            kept[path] = len(values)
        else:
            dropped[path] = (len(values), reason)
    return {"kept": kept, "dropped": dropped}
