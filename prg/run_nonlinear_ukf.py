"""CLI entry point for NonLinear UKF — delegates to prg.run_filter."""

from prg.run_filter import main as _main


def main() -> None:
    _main("ukf")


if __name__ == "__main__":
    main()
