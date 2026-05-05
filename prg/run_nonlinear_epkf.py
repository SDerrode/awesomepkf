import argparse
import sys

from prg.base_classes.filter_runner import FilterRunner
from prg.utils.exceptions import FilterError, NumericalError, ParamError, PKFError
from prg.utils.parser import add_arguments


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NonLinear EPKF")

    add_arguments(
        parser,
        ["nonLinearModelName", "linearModelName", "N", "sKey", "dataFileName"],
    )

    args = parser.parse_args()

    if args.dataFileName is not None and args.N is not None:
        parser.error("--N should not be used with --dataFileName")

    if args.dataFileName is None and args.N is None:
        parser.error("--N must be used when --dataFileName is not specified")

    if args.linearModelName is not None and args.nonLinearModelName is not None:
        parser.error(
            "--nonLinearModelName should not be used with --linearModelName. One or the other!"
        )
    if args.linearModelName is None and args.nonLinearModelName is None:
        parser.error("--nonLinearModelName OR --linearModelName must be used.")

    model_name = (
        args.nonLinearModelName
        if args.linearModelName is None
        else args.linearModelName
    )

    return args, model_name


def main() -> None:
    args, model_name = parse_arguments()
    mode = "from_file" if args.dataFileName is not None else "simulation"

    try:
        runner = FilterRunner(
            filter_name="epkf",
            model_name=model_name,
            mode=mode,
            N=args.N,
            sKey=args.sKey,
            data_filename=args.dataFileName,
            verbose=args.verbose,
            plot=args.plot,
            save_history=args.saveHistory,
        )
        runner.run()

    except NumericalError as e:
        print(f"[NUMERICAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except FilterError as e:
        print(f"[FILTER ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except PKFError as e:
        print(f"[PKF ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ParamError as e:
        print(f"[PARAMETER ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[PARAMETER ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"[RUNTIME ERROR] {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
