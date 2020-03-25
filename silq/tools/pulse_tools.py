import numpy as np
from typing import List, Union
import logging
from matplotlib import pyplot as plt


__all__ = ["pulse_to_waveform_sequence"]

logger = logging.getLogger(__name__)


def pulse_to_waveform_sequence(
    max_points: int,
    frequency: float,
    sampling_rate: float,
    frequency_threshold: float,
    total_duration=None,
    final_delay_threshold: float = None,
    min_points: int = 1,
    sample_points_multiple: int = 1,
    point_offsets: List[int] = [-1, 0, 1, 2],
    filters=[],
    plot=False,
) -> Union[dict, None]:
    """
    This method can be used when generating a periodic signal with an AWG device.
    Given a frequency and duration of the desired signal, a general AWG can produce
    that signal by repeating one waveform (waveform_1) for a number of times
    (cycles) and ending with a second waveform (waveform_2). This is a practical
    way of generating periodic signals that are long in duration without using
    a lot of the available RAM of the AWG.

    Because of the finite sampling rate and restrictions on amount of
    sample points (which must generally be a multiple of a certain number),
    there will be an error in the period of the generated signal. This error goes
    down with the number of periods (n) that one cycle of the repeated waveform contains.

    This function calculates the minimum number n for which the error threshold
    is satisfied. Therefore minimizing the total amount of sample points that
    need to be stored by the AWG.

    Args:
        duration: duration of the signal in seconds
        frequency: frequency of the signal in Hz
        sampling_rate: the sampling rate of the waveform
        threshold: threshold in relative period error
        n_min: minimum number of signal periods that the waveform must contain
        n_max: maximum number of signal periods that a waveform can contain
        sample_points_multiple: the number of samples must be a multiple of

    Returns:
        Dict containing:
        'optimum': Dict of optimal settings
            'error': Relative frequency error of optimal settings
            'final_delay': Remaining duration of pulse after waveform repetitions.
                Cannot be negative.
            'periods': Periods within waveform (using modified frequency)
            'repetitions': Repetitions of waveform to (almost) reach end of pulse
            'duration': Duration of main waveform
            'points': Number of waveform points
            'idx': index of optimal result. first index corresponds to period
                index of periods between min_periods and max_periods.
                Second index is the point offset index.
            'modified_frequency': Frequency close to target frequency whose
                period perfectly fits in the number of points
        'filtered_results': Array of settings that satisfy filters
        'repetitions_multiple': Repetition array for all settings
        'final_delays': Array of final_delays for all settings
        'errors': Array of relative frequency errors for all settings
        'periods_range': Range of periods that have been considered

        If the minimum number of periods (set by min_points) exceeds the maximum
        number of periods (set by max_points), None is returned
    """
    t_period = 1 / abs(frequency)
    max_periods = int(max_points / sampling_rate / t_period)  # Bounded by max_points
    min_periods = int(
        np.ceil(min_points / sampling_rate / t_period)
    )  # Bounded by min_points

    if min_periods >= max_periods:
        return None

    periods = np.arange(min_periods, max_periods + 1)
    t_periods = periods * t_period
    # Always consider neighbouring points as well, as they may have more favourable settings
    point_offsets_multiple = np.array(point_offsets, dtype=int) * sample_points_multiple

    # Calculate frequency errors
    points_periods = (
        t_periods * sampling_rate
    )  # Unrounded points for each periods value (1D)
    points_cutoff = (
        sample_points_multiple * (points_periods // sample_points_multiple)
    ).astype(
        int
    )  # Rounded (floor) points for each periods value (1D)
    # Rounded (floor) points for each periods value with offsets (2D)
    points_cutoff_multiple = points_cutoff[:, np.newaxis] + point_offsets_multiple
    # Durations of rounded (floor) points for each periods value with offsets (2D)
    t_cutoff_multiple = points_cutoff_multiple / sampling_rate
    # Relative frequency error (2D)
    errors = np.abs(1 - t_cutoff_multiple / t_periods[:, np.newaxis])

    # Calculate final delays
    final_delays = total_duration * np.ones(points_cutoff_multiple.shape)
    repetitions_multiple = ((total_duration + 1e-13) // t_cutoff_multiple).astype(int)
    t_periods_cutoff_multiple = t_cutoff_multiple * repetitions_multiple
    final_delays -= t_periods_cutoff_multiple

    # Filter results
    filtered_results = np.ones(errors.shape, dtype=bool)
    # Ensure that all results have at least one repetition
    filtered_results[repetitions_multiple < 1] = False
    filter_arrs = {
        "frequency": errors,
        "final_delay": final_delays,
        "points": points_cutoff_multiple,
    }
    # Prepend thresholds to filters
    if final_delay_threshold is not None:
        filters = [("final_delay", final_delay_threshold)] + filters
    filters = [("frequency", frequency_threshold)] + filters

    for k, (filter_name, threshold) in enumerate(filters):
        filter_arr = filter_arrs[filter_name]
        if np.any(filtered_results[filter_arr <= threshold]):
            logger.debug(
                f"Found {np.sum(filtered_results[filter_arr <= threshold])} "
                f"satisfying {filter_name} < {threshold}"
            )
            filtered_results[filter_arr > threshold] = False
        else:
            min_val = filter_arr[filtered_results].min()
            remaining_results = filtered_results.copy()
            remaining_results[filter_arr != min_val] = 0
            # min_idx is a tuple containing the first element that minimizes
            # according to the filter
            min_idx = np.unravel_index(
                remaining_results.argmax(), remaining_results.shape
            )
            log_str = (
                f"Could not find any sine waveform decomposition with "
                f"{filter_name} error < {threshold}, "
            )

            # Print log message, either as warning or as debug message
            if k == 1 or k == 2 and final_delay_threshold is not None:
                logger.warning(log_str)
            else:
                logger.debug(log_str)
            break

    else:
        logger.debug("Satisfied all filters, choosing lowest frequency error")
        min_val = errors[filtered_results].min()
        remaining_results = filtered_results.copy()
        remaining_results[errors != min_val] = 0
        min_idx = np.unravel_index(remaining_results.argmax(), remaining_results.shape)
    modified_frequency = 1 / (
        points_cutoff_multiple[min_idx] / periods[min_idx[0]] / sampling_rate
    )
    if frequency < 0:
        modified_frequency *= -1

    optimum = {
        "error": errors[min_idx],
        "final_delay": final_delays[min_idx],
        "periods": min_periods + min_idx[0],
        "repetitions": repetitions_multiple[min_idx],
        "duration": points_cutoff_multiple[min_idx] / sampling_rate,
        "points": points_cutoff_multiple[min_idx],
        "idx": min_idx,
        "modified_frequency": modified_frequency,
    }

    if plot:
        fig, axes = plt.subplots(3, sharex=True)
        axes[0].semilogy(periods, repetitions_multiple)
        axes[1].semilogy(periods, errors)
        axes[2].semilogy(periods, final_delays)
        axes[0].set_ylabel("Repetitions")
        axes[1].set_ylabel("Rel. frequency error")
        axes[2].set_ylabel("Final delay (s)")
        axes[2].set_xlabel("Periods")

        handle = axes[0].vlines(
            optimum["periods"], *axes[0].get_ylim(), linestyle="--", lw=2
        )
        axes[1].vlines(optimum["periods"], *axes[1].get_ylim(), linestyle="--", lw=2)
        axes[2].vlines(optimum["periods"], *axes[2].get_ylim(), linestyle="--", lw=2)
        axes[0].legend([handle], ["optimum"])

        plt.subplots_adjust(hspace=0)

    return {
        "errors": errors,
        "periods_range": periods,
        "final_delays": final_delays,
        "repetitions_multiple": repetitions_multiple,
        "points_per_period": points_cutoff_multiple,
        "filtered_results": filtered_results,
        "optimum": optimum,
    }
