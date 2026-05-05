"""CLI entry point for Linear PKF — delegates to prg.run_filter."""

from prg.run_filter import main as _main


def main() -> None:
    _main("pkf")


if __name__ == "__main__":
    main()
