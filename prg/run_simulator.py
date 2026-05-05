import argparse
import sys

from prg.base_classes.simulator_linear import LinearDataSimulator
from prg.base_classes.simulator_nonlinear import NonLinearDataSimulator
from prg.utils.parser import add_arguments

# =============================================================
# Parser
# =============================================================


def _print_model_list() -> None:
    """Print the available linear and nonlinear models then exit."""
    # Deferred imports: avoid the (slow) factory discovery cost when the
    # user only wants to simulate (the common path).
    from prg.models.linear import ModelFactoryLinear  # noqa: PLC0415
    from prg.models.nonLinear import ModelFactoryNonLinear  # noqa: PLC0415

    nl_models = sorted(ModelFactoryNonLinear.list_models())
    lin_models = sorted(ModelFactoryLinear.list_models())

    print("\nAvailable nonlinear models:")
    for name in nl_models:
        print(f"  {name}")

    print("\nAvailable linear models:")
    for name in lin_models:
        print(f"  {name}")

    print()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate linear or non-linear data")

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available models and exit",
    )

    add_arguments(
        parser,
        [
            "linear-model-name",
            "nonlinear-model-name",
            "N",
            "s-key",
            "data-filename",
            "without-x",
        ],
    )

    args = parser.parse_args()

    if args.list_models:
        _print_model_list()
        sys.exit(0)

    if args.linear_model_name is not None and args.nonlinear_model_name is not None:
        parser.error(
            "--nonlinear-model-name should not be used with --linear-model-name. One or the other!"
        )
    if args.linear_model_name is None and args.nonlinear_model_name is None:
        parser.error("--nonlinear-model-name OR --linear-model-name must be used.")

    return args


# =============================================================
# Main
# =============================================================


def main() -> None:
    args = parse_arguments()

    if args.linear_model_name and args.nonlinear_model_name:
        raise ValueError(
            "Please provide only one of --linear-model-name or --nonlinear-model-name"
        )

    if args.linear_model_name:
        simulator = LinearDataSimulator(
            model_name=args.linear_model_name,
            N=args.N,
            sKey=args.s_key,
            data_file_name=args.data_filename,
            verbose=args.verbose,
            withoutX=args.without_x,
        )

    elif args.nonlinear_model_name:
        simulator = NonLinearDataSimulator(
            model_name=args.nonlinear_model_name,
            N=args.N,
            sKey=args.s_key,
            data_file_name=args.data_filename,
            verbose=args.verbose,
            withoutX=args.without_x,
        )

    else:
        raise ValueError(
            "Please provide either --linear-model-name or --nonlinear-model-name"
        )

    simulator.run()


if __name__ == "__main__":
    main()
