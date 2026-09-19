"""One entry point for the whole tool: `jevopt <subcommand> [options]`.

Each subcommand is a module that already owns its own argument parser, so this
file declares no flags of its own beyond `--version`. It resolves the first
word to a module, hands the rest of the command line to that module's
`main(argv)`, and stays out of the way -- which is what keeps
`python3 -m jevopt.optimize ...` working identically.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys

from . import __version__

COMMANDS: dict[str, tuple[str, str]] = {
    "optimize": ("jevopt.optimize", "evolve a task's option descriptions with GEPA"),
    "baselines": ("jevopt.baselines", "greedy and random clause controls, no search"),
    "compare": ("jevopt.compare", "paired comparison of prompt arms on shared instances"),
    "report": ("jevopt.report", "turn results JSON into a markdown comparison"),
    "ask": ("jevopt.ask", "ask Jev noul/choice/score questions directly"),
}


def build_parser() -> argparse.ArgumentParser:
    width = max(len(name) for name in COMMANDS)
    listing = "\n".join(f"  {name:{width}s}  {help_}"
                        for name, (_module, help_) in COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog="jevopt",
        description="Prompt optimisation for the Jev decision model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"subcommands:\n{listing}\n\n"
               "Every subcommand takes --help of its own:\n"
               "  jevopt optimize --help\n")
    parser.add_argument("--version", action="version",
                        version=f"jevopt {__version__}")
    parser.add_argument("command", nargs="?", choices=list(COMMANDS),
                        metavar="COMMAND", help="one of the subcommands below")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        metavar="...", help="arguments for the subcommand")
    return parser


def run(command: str, argv: list[str]) -> None:
    """Dispatch to a subcommand module's main().

    argv[0] is rewritten so the subcommand's own parser reports usage as
    `jevopt optimize ...` rather than `jevopt ...`, and so a module whose main()
    still reads sys.argv sees what it expects.
    """
    module = importlib.import_module(COMMANDS[command][0])
    saved = sys.argv
    sys.argv = [f"jevopt {command}", *argv]
    try:
        takes_argv = bool(inspect.signature(module.main).parameters)
        module.main(argv) if takes_argv else module.main()
    finally:
        sys.argv = saved


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    run(args.command, args.args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
