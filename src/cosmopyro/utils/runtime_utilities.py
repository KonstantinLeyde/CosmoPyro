import argparse
import datetime

__all__ = [
    "get_config",
    "get_time_stamp",
    "get_unit_sec_from_unit_string",
]


def get_unit_sec_from_unit_string(unit):

    if unit == "min":
        return 60
    elif unit == "hr":
        return 60**2
    elif unit == "day":
        return 60**2 * 24
    else:
        raise "Unit not recognized. "


def get_time_stamp(unit, include_last_digits=False):
    # Get current UTC time
    current_utc_time = datetime.datetime.now(datetime.UTC)

    # Define the Unix epoch
    unix_epoch = datetime.datetime(2025, 10, 15, 0, 0, 0, tzinfo=datetime.UTC)

    time_difference = current_utc_time - unix_epoch

    unit_sec = get_unit_sec_from_unit_string(unit)

    # Convert to total minutes
    total_time_since_epoch = int(time_difference.total_seconds() // unit_sec)

    s = str(total_time_since_epoch)

    if include_last_digits:
        last_digits_milliseconds = str(time_difference.microseconds)[-2:]
        s += "_" + last_digits_milliseconds

    return s


def get_config():
    """
    Parse command-line arguments and return them as a dictionary.
    Example:
        python run_analysis.py --use_skymap true --num_events 10 --num_posterior_samples 1000 --debug false
    """
    parser = argparse.ArgumentParser(description="Parse analysis configuration.")

    parser.add_argument(
        "--num_events",
        type=int,
        default=None,
        help="Number of events (default: use all)",
    )
    parser.add_argument(
        "--num_posterior_samples",
        type=int,
        default=None,
        help="Number of posterior samples (default: use all)",
    )
    parser.add_argument(
        "--debug",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable debug mode (true/false)",
    )
    parser.add_argument("--job_id", type=str, required=True, help="Job id")
    parser.add_argument(
        "--path_posterior_samples",
        type=str,
        required=False,
        help="Path to posterior samples data file",
    )
    parser.add_argument(
        "--path_injections",
        type=str,
        required=False,
        help="Path to injections data file",
    )
    parser.add_argument(
        "--path_kwargs",
        type=str,
        required=False,
        help="Path to kwargs analysis yaml file",
    )
    parser.add_argument(
        "--path_skymap", type=str, required=False, help="Path to skymap file"
    )
    parser.add_argument(
        "--sampler", type=str, default="nuts", help="Sampler to use (e.g., nuts, mclmc)"
    )
    parser.add_argument(
        "--num_injections",
        type=int,
        default=None,
        help="Number of injections to use (if not using all)",
    )

    args = parser.parse_args()
    return vars(args)
