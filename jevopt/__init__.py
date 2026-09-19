"""Prompt optimisation for `typesafe/jev-1.13`, a decision model that returns a
typed choice rather than text.

`jevopt` evolves the option descriptions of one Jev `choice` question from
labelled states, using GEPA's Pareto search with a proposer that never generates
text. Describe a decision as a `Task` -- the options, the labelled states, and
the `Condition` vocabulary a clause may be built from, which `derive` can read
off the states themselves -- and the rest of the package does not care what the
decision is about.

    from jevopt import Task, derive

Everything that talks to the API lives behind the `jevopt` command; see
`jevopt --help`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .conditions import derive
from .task import Condition, Task

try:
    __version__ = version("jevopt")
except PackageNotFoundError:          # a source checkout that was never installed
    __version__ = "0.1.0"

__all__ = ["Condition", "Task", "__version__", "derive"]
