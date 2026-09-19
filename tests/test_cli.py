"""The front door: `jevopt <subcommand> ...` must reach the right module intact.

Dispatch is the one place where a mistake is invisible -- every subcommand still
"works", it just receives the wrong argv, or swallows a flag, or silently does
nothing. So these tests replace each target `main` and assert on the argv it was
handed. Nothing here imports a stub or spends anything: the real mains are never
called.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from jevopt import cli

SUBCOMMANDS = sorted(cli.COMMANDS)


@pytest.fixture
def spy(monkeypatch):
    """Replace one subcommand's main(); return the list of argvs it receives."""
    def install(command: str, takes_argv: bool = True):
        seen: list = []
        module = importlib.import_module(cli.COMMANDS[command][0])
        if takes_argv:
            monkeypatch.setattr(module, "main", lambda argv: seen.append(argv))
        else:
            monkeypatch.setattr(module, "main", lambda: seen.append(list(sys.argv)))
        return seen
    return install


# ------------------------------------------------------------------- dispatch

@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_each_subcommand_reaches_its_own_module(spy, command):
    seen = spy(command)

    assert cli.main([command]) == 0
    assert seen == [[]]


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_the_rest_of_the_command_line_arrives_untouched(spy, command):
    """Flags the top-level parser has never heard of must survive verbatim."""
    rest = ["--task", "jevopt.tasks.triage", "--budget", "1500",
            "name=runs/x.json", "--force", "-v", "--", "trailing"]
    seen = spy(command)

    assert cli.main([command, *rest]) == 0
    assert seen == [rest]


def test_a_leading_dash_argument_is_not_eaten_by_the_top_level_parser(spy):
    """`--version` belongs to jevopt, but only before the subcommand."""
    seen = spy("ask")

    assert cli.main(["ask", "--version"]) == 0
    assert seen == [["--version"]]


def test_dispatch_rewrites_sys_argv_for_a_main_that_still_reads_it(spy):
    seen = spy("report", takes_argv=False)

    cli.main(["report", "--task", "t"])

    assert seen == [["jevopt report", "--task", "t"]]


def test_dispatch_restores_sys_argv_afterwards(spy):
    before = list(sys.argv)
    spy("report")

    cli.main(["report", "--task", "t"])

    assert sys.argv == before


def test_sys_argv_is_restored_even_when_the_subcommand_exits(monkeypatch):
    before = list(sys.argv)

    def boom(argv):
        raise SystemExit(3)

    monkeypatch.setattr("jevopt.report.main", boom)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["report"])

    assert excinfo.value.code == 3
    assert sys.argv == before


def test_argv_defaults_to_sys_argv_when_none_is_passed(spy, monkeypatch):
    seen = spy("compare")
    monkeypatch.setattr(sys, "argv", ["jevopt", "compare", "--limit", "2"])

    assert cli.main() == 0
    assert seen == [["--limit", "2"]]


# ---------------------------------------------------------------- no subcommand

def test_bare_jevopt_prints_help_and_fails(capsys):
    code = cli.main([])

    assert code != 0
    help_text = capsys.readouterr().err
    assert "usage: jevopt" in help_text
    # The help is only useful if it says what the subcommands are.
    for command in SUBCOMMANDS:
        assert command in help_text


def test_version_prints_the_package_version(capsys):
    from jevopt import __version__

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"jevopt {__version__}"


def test_an_unknown_subcommand_fails_and_names_the_real_ones(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["summarise"])

    assert excinfo.value.code != 0
    complaint = capsys.readouterr().err
    assert "summarise" in complaint
    for command in SUBCOMMANDS:
        assert command in complaint


def test_every_advertised_command_maps_to_an_importable_main():
    """The table is hand-written; a typo in it only shows up at dispatch time."""
    for command, (dotted, help_) in cli.COMMANDS.items():
        module = importlib.import_module(dotted)
        assert callable(module.main), command
        assert help_ and help_[0].islower(), command
