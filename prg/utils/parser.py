import argparse
import warnings


def int_ge_1(value: str) -> int:
    """Parse *value* as ``int`` and verify it is ≥ 1."""
    try:
        ivalue = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from e

    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} must be an integer ≥ 1")

    return ivalue


# ---------------------------------------------------------------------------
# Optional options configuration
# ---------------------------------------------------------------------------

_OPTION_CONFIG: dict = {
    "N": {
        "type": int_ge_1,
        "default": None,
        "help": "Number of samples to process (default: None)",
    },
    "n_particles": {
        "type": int_ge_1,
        "default": 300,
        "help": "Number of particles to use (default: 300)",
    },
    "sKey": {
        "type": int,
        "default": None,
        "help": "Random generator seed (default: None)",
    },
    "linearModelName": {
        "type": str,
        "default": None,
        "help": "Linear model to use (default: None)",
    },
    "sigmaSet": {
        "choices": ["wan2000", "cpkf", "lerner2002", "ito2000"],
        "default": "wan2000",
        "help": "Sigma points set for UPKF (default: wan2000)",
    },
    "nonLinearModelName": {
        "type": str,
        "default": None,
        "help": "Nonlinear model to use (default: None)",
    },
    "dataFileName": {
        "type": str,
        "default": None,
        "help": "Path to the trajectory file (default: None)",
    },
    "withoutX": {
        "action": "store_true",
        "help": "Do not save the true state X (default: False)",
    },
    "filter": {
        "choices": ["EPKF", "UPKF", "PPF", "UKF", "PKF"],
        "default": None,
        "help": "Filter type to use (default: None)",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser, list_options: list[str]) -> None:
    """
    Add arguments to an ``ArgumentParser``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to enrich.
    list_options : list[str]
        Names of the optional options to add (keys of ``_OPTION_CONFIG``).

    Raises
    ------
    None — unknown keys emit a ``UserWarning`` instead of being
    silently ignored.
    """

    # --- Always-available options ---
    parser.add_argument(
        "--verbose",
        type=int,
        choices=range(0, 3),
        default=0,
        help="Verbosity level (0=silent, 2=maximum, default=0)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display and save signals to disk (default: False)",  # FIX: was "True if not specified" → inverted
    )
    parser.add_argument(
        "--saveHistory",
        action="store_true",
        help="Save the parameter trace to disk (default: False)",
    )

    # --- Configurable optional options ---
    for opt in list_options:
        if opt not in _OPTION_CONFIG:
            # FIX: unknown option → explicit warning instead of silently ignoring
            warnings.warn(
                f"add_arguments: unknown option {opt!r} ignored "
                f"(available options: {list(_OPTION_CONFIG)})",
                UserWarning,
                stacklevel=2,
            )
            continue

        kwargs = _OPTION_CONFIG[opt].copy()
        # FIX: dest=opt removed — argparse infers it automatically from --opt
        parser.add_argument(f"--{opt}", **kwargs)
