import numpy as np
from typing import List
import logging
from matplotlib import pyplot as plt


__all__ = ['pulse_to_waveform_sequence']

logger = logging.getLogger(__name__)

def pulse_to_waveform_sequence(max_points: int,
                               frequency: float,
                               sampling_rate: float,
                               frequency_threshold: float,
                               total_duration=None,
                               final_delay_threshold: float = None,
                               min_points: int = 1,
                               sample_points_multiple: int = 1,
                               point_offsets: List[int] = [-1, 0, 1, 2],
                               filters=[],
                               plot=False):
    """ 
    This method can be used when generating a periodic signal with an AWG device. Given a frequency and duration of the
    desired signal, a general AWG can produce that signal by repeating one waveform (waveform_1) for a number of times
    (cycles) and ending with a second waveform (waveform_2). This is a practical way of generating periodic signals that
    are long in duration without using a lot of the available RAM of the AWG.

    Because of the finite sampling rate and restrictions on amount of sample points (which must generally be a multiple
    of a certain number), there will be an error in the period of the generated signal. This error goes down with the
    number of periods (n) that one cycle of the repeated waveform contains.

    This function calculates the minimum number n for which the error threshold is satisfied. Therefore minimizing the
    total amount of sample points that need to be stored by the AWG.

        Args:
            duration (float): duration of the signal in seconds
            frequency (float): frequency of the signal in Hz
            sampling_rate (float): the sampling rate of the waveform
            threshold (float): threshold in relative period error
            n_min (int): minimum number of signal periods that the waveform must contain
            n_max (int): maximum number of signal periods that a waveform can contain
            sample_points_multiple (int): the number of samples must be a multiple of

        Returns:
            (tuple):
                n (int):        number of signal periods that are in one cycle of the repeating waveform
                error (float):  relative error in the signal period
                samples (int):  number of samples in one cycle of the repeating waveform

    """
    t_period = 1 / abs(frequency)
    max_periods = int(max_points / sampling_rate / t_period)
    min_periods = int(np.ceil(min_points / sampling_rate / t_period))
    periods = np.arange(min_periods, max_periods + 1)
    t_periods = periods * t_period
    point_offsets_multiple = np.array(point_offsets, dtype=int) * sample_points_multiple

    # Calculate frequency errors
    points_periods = t_periods * sampling_rate
    points_cutoff = (sample_points_multiple * (points_periods // sample_points_multiple)).astype(int)
    points_cutoff_multiple = points_cutoff[:, np.newaxis] + point_offsets_multiple
    t_cutoff_multiple = points_cutoff_multiple / sampling_rate
    errors = np.abs(1 - t_cutoff_multiple / t_periods[:, np.newaxis])

    # Calculate final delays
    final_delays = total_duration * np.ones((max_periods + 1 - min_periods, len(point_offsets)))
    repetitions_multiple = (total_duration // t_cutoff_multiple).astype(int)
    t_periods_cutoff_multiple = t_cutoff_multiple * repetitions_multiple
    final_delays -= t_periods_cutoff_multiple

    # Filter results
    filtered_results = np.ones(errors.shape, dtype=bool)
    filter_arrs = {'frequency': errors,
                  'final_delay': final_delays,
                  'points': points_cutoff_multiple}
    # Prepend thresholds to filters
    if final_delay_threshold is not None:
        filters = [('final_delay', final_delay_threshold)] + filters
    filters = [('frequency', frequency_threshold)] + filters

    for k, (filter_name, threshold) in enumerate(filters):
        filter_arr = filter_arrs[filter_name]
        if np.any(filtered_results[filter_arr <= threshold]):
            logger.debug(f'Found {np.sum(filtered_results[filter_arr <= threshold])} '
                         f'satisfying {filter_name} < {threshold}')
            filtered_results[filter_arr > threshold] = False
        else:
            min_val = filter_arr[filtered_results].min()
            remaining_results = filtered_results.copy()
            remaining_results[filter_arr != min_val] = 0
            min_idx = np.unravel_index(remaining_results.argmax(),
                                       remaining_results.shape)
            log_str = f"Could not find any sine waveform decomposition with " \
                      f"{filter_name} error < {threshold}, "

            # Print log message, either as warning or as debug message
            if k == 1 or k == 2 and final_delay_threshold is not None:
                logger.warning(log_str)
            else:
                logger.debug(log_str)
            break

    else:
        logger.debug('Satisfied all filters, choosing lowest frequency error')
        min_val = errors[filtered_results].min()
        remaining_results = filtered_results.copy()
        remaining_results[errors != min_val] = 0
        min_idx = np.unravel_index(remaining_results.argmax(),
                                   remaining_results.shape)
    modified_frequency = 1 / (points_cutoff_multiple[min_idx] / periods[min_idx[0]] / sampling_rate)
    if frequency < 0:
        modified_frequency *= -1

    optimum= {'error': errors[min_idx],
              'final_delay': final_delays[min_idx],
              'periods': min_periods + min_idx[0],
              'repetitions': repetitions_multiple[min_idx],
              'points': points_cutoff_multiple[min_idx],
              'idx': min_idx,
              'modified_frequency': modified_frequency}

    if plot:
        fig, axes = plt.subplots(2, sharex=True)
        axes[0].semilogy(periods, errors)
        axes[1].semilogy(periods, final_delays)
        axes[0].set_ylabel('Rel. frequency error')
        axes[1].set_ylabel('Final delay (s)')
        axes[1].set_xlabel('Periods')

        handle = axes[0].vlines(optimum['periods'], *axes[0].get_ylim(),
                                linestyle='--', lw=2)
        axes[1].vlines(optimum['periods'], *axes[1].get_ylim(),
                       linestyle='--', lw=2)
        axes[0].legend([handle], ['optimum'])

        plt.subplots_adjust(hspace=0)

    return {'errors': errors,
            'final_delays': final_delays,
            'repetitions_multiple': repetitions_multiple,
            'filtered_results': filtered_results,
            'optimum': optimum}
