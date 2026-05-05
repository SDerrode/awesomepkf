"""CLI entry point for NonLinear UPKF — delegates to prg.run_filter."""

from prg.run_filter import main as _main


def main() -> None:
    _main("upkf")


if __name__ == "__main__":
    main()
