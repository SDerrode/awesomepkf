"""CLI entry point for NonLinear PPF — delegates to prg.run_filter."""

from prg.run_filter import main as _main


def main() -> None:
    _main("ppf")


if __name__ == "__main__":
    main()
