"""CLI entry point for NonLinear PF — delegates to prg.run_filter."""

from prg.run_filter import main as _main


def main() -> None:
    _main("pf")


if __name__ == "__main__":
    main()
