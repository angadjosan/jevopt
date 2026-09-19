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


def derive(instances: list[dict], extra: list[Condition] | None = None,
           max_values: int = MAX_VALUES) -> list[Condition]:
    """One equality condition per (field, value) the states actually contain."""
    seen: dict[str, set] = {}
    for instance in instances:
        for path, value in flatten(instance["state"]).items():
            if path.endswith(SKIP_SUFFIXES):
                continue
            seen.setdefault(path, set()).add(value)

    conditions: list[Condition] = []
    for path, values in sorted(seen.items()):
        if not 2 <= len(values) <= max_values:
            continue                  # constant fields say nothing; wide ones are ids
        getter = _at(path)
        for value in sorted(values, key=str):
            conditions.append(Condition(
                id=f"{path}={value}",
                phrase=f'{path} is "{value}"',
                test=(lambda g, v: (lambda s: g(s) == v))(getter, value),
            ))
    conditions.extend(extra or [])
    return conditions
