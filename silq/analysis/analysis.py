import numpy as np
import peakutils
import logging
from typing import Union, Dict, Any, List, Sequence
import collections

__all__ = ['smooth', 'find_high_low', 'edge_voltage', 'find_up_proportion',
           'count_blips', 'analyse_traces', 'analyse_EPR', 'analyse_flips']

logger = logging.getLogger(__name__)

from silq import config
if 'analysis' not in config:
    config['analysis'] = {}
analysis_config = config['analysis']


def smooth(x: np.ndarray,
           window_len: int = 11,
           window: str = 'hanning') -> np.ndarray:
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    # TODO: Somehow it shifts the center.
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', "
                         "'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def find_high_low(traces: np.ndarray,
                  plot: bool = False,
                  threshold_peak: float = 0.02,
                  attempts: int = 8,
                  threshold_method: str = 'config',
                  min_SNR: Union[float, None] = None) -> Dict[str, Any]:
    """ Find high and low voltages of traces using histograms

    This function determines the high and low voltages of traces by binning them
    into 30 bins, and trying to discern two peaks.
    Useful for determining the threshold value for measuring a blip.

    If no two peaks can be discerned after all attempts, None is returned for
    each of the returned dict keys except DC_voltage.

    Args:
        traces: 2D array of acquisition traces
        plot: Whether to plot the histograms
        threshold_peak: threshold for discerning a peak. Will be varied if too
            many/few peaks are found
        attempts: Maximum number of attempts for discerning two peaks.
            Each attempt the threshold_peak is decreased/increased depending on
            if too many/few peaks were found
        threshold_method: Method used to determine the threshold voltage.
            Allowed methods are:
                `mean`: average of high and low voltage.
                `{n}*std_low`: n standard deviations above mean low voltage
                `{n}*std_high`: n standard deviations below mean high voltage
                `config`: Use threshold method provided in
                    `config.analysis.threshold_method` (`mean` if not specified)
        min_SNR: Minimum SNR between high and low voltages required to determine
            a threshold voltage.

    Returns:
        Dict containing:
            low (float): mean low voltage, `None` if two peaks cannot be
                discerned
            `high` (float): mean high voltage, `None` if no two peaks cannot be
                discerned
            threshold_voltage (float): threshold voltage for a blip. If SNR is
            below `min_SNR` or no two peaks can be discerned, returns `None`.
            voltage_difference (float): difference between low and high voltage.
                If not two peaks can be discerned, returns `None`.
            DC_voltage (float): average voltage of traces.

    """
    assert attempts > 0, f'Attempts {attempts} must be at least 1'

    # Calculate DC (mean) voltage
    DC_voltage = np.mean(traces)

    # Determine threshold method
    if threshold_method == 'config':
        threshold_method = analysis_config.get('threshold_method', 'mean')

    hist, bin_edges = np.histogram(np.ravel(traces), bins=30)

    # Find two peaks
    for k in range(attempts):
        peaks_idx = np.sort(peakutils.indexes(hist, thres=threshold_peak,
                                              min_dist=5))
        if len(peaks_idx) == 2:
            break
        elif len(peaks_idx) == 1:
            # print('One peak found instead of two, lowering threshold')
            threshold_peak /= 1.5
        elif len(peaks_idx) > 2:
            # print(f'Found {len(peaks_idx)} peaks instead of two, '
            #        'increasing threshold')
            threshold_peak *= 1.5
        else:
            return {'low': None,
                    'high': None,
                    'threshold_voltage': None,
                    'voltage_difference': None,
                    'DC_voltage': DC_voltage}

    # Find mean voltage, used to determine which points are low/high
    mean_voltage_idx = int(np.round(np.mean(peaks_idx)))
    mean_voltage = bin_edges[mean_voltage_idx]

    # Create dictionaries containing information about the low, high state
    low, high = {}, {}
    low['traces'] = traces[traces < mean_voltage]
    high['traces'] = traces[traces > mean_voltage]
    for signal in [low, high]:
        signal['mean'] = np.mean(signal['traces'])
        signal['std'] = np.std(signal['traces'])
    voltage_difference = (high['mean'] - low['mean'])

    if threshold_method == 'mean':
        # Threshold_method is midway between low and high mean
        threshold_voltage = (high['mean'] - low['mean']) / 2
    elif 'std_low' in threshold_method:
        # Threshold_method is {factor} standard deviations above low mean
        factor = float(threshold_method.split('*')[0])
        threshold_voltage = low['mean'] + factor * low['std']
    elif 'std_high' in threshold_method:
        # Threshold_method is {factor} standard deviations below high mean
        factor = float(threshold_method.split('*')[0])
        threshold_voltage = high['mean'] - factor * high['std']

    SNR = voltage_difference / np.sqrt(high['std'] ** 2 + low['std'] ** 2)
    if min_SNR is None:
        min_SNR = analysis_config.min_SNR
    if SNR < min_SNR:
        logger.info(f'Signal to noise ratio {SNR} is too low')
        threshold_voltage = None

    # Plotting
    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        for k, signal in enumerate([low, high]):
            sub_hist, sub_bin_edges = np.histogram(np.ravel(signal['traces']), bins=10)
            plt.bar(sub_bin_edges[:-1], sub_hist, width=0.05, color='bg'[k])
            plt.plot(signal['mean'], hist[peaks_idx[k]], 'or', ms=12)

    return {'low': low,
            'high': high,
            'threshold_voltage': threshold_voltage,
            'voltage_difference': voltage_difference,
            'SNR': SNR,
            'DC_voltage': DC_voltage}


def edge_voltage(traces: np.ndarray,
                 edge: str,
                 state: str,
                 threshold_voltage: Union[float, None] = None,
                 points: int = 5,
                 start_idx: int = 0) -> np.ndarray:
    """ Test traces for having a high/low voltage at begin/end

    Args:
        traces: 2D array of acquisition traces
        edge: which side of traces to test, either `begin` or `end`
        state: voltage that the edge must have, either `low` or `high`
        threshold_voltage: threshold voltage for a `high` voltage (blip).
            If not specified, `find_high_low` is used to determine threshold
        points: Number of data points to average over to determine state
        start_idx: index of first point to use. Useful if there is some
            capacitive voltage spike occuring at the start.
            Only used if edge is `begin`

    Returns:
        1D boolean array, True if the trace has the correct state at the edge
    """
    assert edge in ['begin', 'end'], f'Edge {edge} must be `begin` or `end`'
    assert state in ['low', 'high'], f'State {state} must be `low` or `high`'

    if edge == 'begin':
        if start_idx > 0:
            idx_list = slice(start_idx, start_idx + points)
        else:
            idx_list = slice(None, points)
    else:
        idx_list = slice(-points, None)

    # Determine threshold voltage if not provided
    if threshold_voltage is None:
        threshold_voltage = find_high_low(traces)['threshold_voltage']

    if threshold_voltage is None:
        # print('Could not find two peaks for empty and load state')
        success = np.array([False] * len(traces))
    elif state == 'low':
        success = [np.mean(trace[idx_list]) < threshold_voltage
                   for trace in traces]
    else:
        success = [np.mean(trace[idx_list]) > threshold_voltage
                   for trace in traces]
    return np.array(success)


def find_up_proportion(traces: np.ndarray,
                       threshold_voltage: Union[float, None] = None,
                       return_array: bool = False,
                       start_idx: int = 0,
                       filter_window: int = 0) -> Union[float, np.ndarray]:
    """ Determine the up proportion of traces (traces that have blips)

    Args:
        traces: 2D array of acquisition traces
        threshold_voltage: threshold voltage for a `high` voltage (blip).
            If not specified, `find_high_low` is used to determine threshold
        return_array: whether to return the boolean array or the up proportion
        start_idx: index of first point to use. Useful if there is some
            capacitive voltage spike occuring at the start.
            Only used if edge is `begin`
        filter_window: number of points of smoothing (0 means no smoothing)

    Returns:
        if return_array is False:
            (float) The proportion of traces with a blip
        else:
            Boolean array, True if the trace has a blip

    """
    # trace has to contain read stage only
    # TODO Change start point to start time (sampling rate independent)
    if threshold_voltage is None:
        threshold_voltage = find_high_low(traces)['threshold_voltage']

    if threshold_voltage is None:
        return 0

    if filter_window > 0:
        traces = [np.convolve(trace, np.ones(filter_window) / filter_window,
                              mode='valid')
                  for trace in traces]

    # Filter out the traces that contain one or more peaks
    traces_up_electron = [np.any(trace[start_idx:] > threshold_voltage)
                          for trace in traces]

    if not return_array:
        return sum(traces_up_electron) / len(traces)
    else:
        return traces_up_electron


def count_blips(traces: np.ndarray,
                threshold_voltage: float,
                sample_rate: float,
                t_skip: float) -> Dict[str, Any]:
    """ Count number of blips and durations in high/low state

    Args:
        traces: 2D array of acquisition traces
        threshold_voltage: threshold voltage for a `high` voltage (blip).
        sample_rate: acquisition sample rate (per second)
        t_skip: initial time to skip for each trace (ms)

    Returns:
        Dict containing:
            blips (float): number of blips per trace
            blips_per_second (float): number of blips per second
            low_blip_duration (np.ndarray): durations in low-voltage state
            high_blip_duration (np.ndarray): durations in high-voltage state
            mean_low_blip_duration (float): average duration in low state
            mean_high_blip_duration (float): average duration in high state
    """
    low_blip_pts, high_blip_pts = [], []
    start_idx = round(t_skip * sample_rate)

    for k, trace in enumerate(traces):
        idx = start_idx
        while idx < len(trace):
            if trace[idx] < threshold_voltage:
                next_idx = np.argmax(trace[idx:] > threshold_voltage)
                blip_list = low_blip_pts
            else:
                next_idx = np.argmax(trace[idx:] < threshold_voltage)
                blip_list = high_blip_pts
            if next_idx == 0:
                blip_list.append(len(trace) - idx)
                break
            else:
                blip_list.append(next_idx)
                idx += next_idx

    low_blip_duration = np.array(low_blip_pts) / sample_rate
    high_blip_duration = np.array(high_blip_pts) / sample_rate

    blips = len(low_blip_durations) / len(traces)

    duration = len(traces[0]) / sample_rate
    return {'blips': blips,
            'blips_per_second': blips / duration,
            'low_blip_durations': low_blip_durations,
            'high_blip_durations': high_blip_durations,
            'mean_low_blip_duration': np.mean(low_blip_durations),
            'mean_high_blip_duration': np.mean(high_blip_durations)}


def analyse_traces(traces: np.ndarray,
                   sample_rate: float,
                   filter: Union[str, None] = None,
                   min_filter_proportion: float = 0.5,
                   t_skip: float = 0,
                   t_read: Union[float, None] = None,
                   segment: str = 'begin',
                   threshold_voltage: Union[float, None] = None,
                   threshold_method: str='config') -> Dict[str, Any]:
    """ Analyse voltage, up proportions, and blips of acquisition traces

    Args:
        traces: 2D array of acquisition traces.
        sample_rate: acquisition sample rate (per second).
        filter: only use traces that begin in low or high voltage.
            Allowed values are `low`, `high` or `None` (do not filter).
        min_filter_proportion: minimum proportion of traces that satisfy filter.
            If below this value, up_proportion etc. are not calculated.
        t_skip: initial time to skip for each trace (ms).
        t_read: duration of each trace to use for calculating up_proportion etc.
            e.g. for a long trace, you want to compare up proportion of start
            and end segments.
        segment: Use beginning or end of trace for `t_read`.
            Allowed values are `begin` and `end`.
        threshold_voltage: threshold voltage for a `high` voltage (blip).
            If not specified, `find_high_low` is used to determine threshold.
        threshold_method: Method used to determine the threshold voltage.
            Allowed methods are:
                `mean`: average of high and low voltage.
                `{n}*std_low`: n standard deviations above mean low voltage.
                `{n}*std_high`: n standard deviations below mean high voltage.
                `config`: Use threshold method provided in
                    `config.analysis.threshold_method` (`mean` if not specified)

    Returns:
        Dict containing:
            up_proportion (float): proportion of traces that has a blip
            end_high (float): proportion of traces that end with high voltage
            end_low (float): proportion of traces that end with low voltage
            num_traces (int): Number of traces that satisfy filter
            filtered_traces_idx (np.ndarray): 1D bool array, True if that trace
                satisfies filter
            voltage_difference (float): voltage difference between high and low
                voltages
            average_voltage (float): average voltage over all traces
            threshold_voltage (float): threshold voltage for counting a blip
                (high voltage). Is calculated if not provided as input arg.
            blips (float): average blips per trace.
            mean_low_blip_duration (float): average duration in low state
            mean_high_blip_duration (float): average duration in high state

    Note:
        If no threshold voltage is provided, and no two peaks can be discerned,
            all results except average_voltage are set to an initial value
            (either 0 or undefined)
        If the filtered trace proportion is less than min_filter_proportion,
            the results `up_proportion`, `end_low`, `end_high` are set to an
            initial value
    """
    assert filter in [None, 'low', 'high'], 'filter must be None, `low`, or `high`'

    assert segment in ['begin', 'end'], 'segment must be either `begin` or `end`'

    # Initialize all results to None
    results = {'up_proportion': 0,
               'end_high': 0,
               'end_low': 0,
               'num_traces': 0,
               'filtered_traces_idx': None,
               'voltage_difference': None,
               'average_voltage': np.mean(traces),
               'threshold_voltage': None,
               'blips': None,
               'mean_low_blip_duration': None,
               'mean_high_blip_duration': None}

    # Histogram trace voltages to find two peaks corresponding to high and low
    high_low_results = find_high_low(traces, threshold_method=threshold_method)
    results['voltage_difference'] = high_low_results['voltage_difference']
    if threshold_voltage is None:
        # Use threshold voltage from high_low_results
        threshold_voltage = high_low_results['threshold_voltage']

    results['threshold_voltage'] = threshold_voltage

    if threshold_voltage is None:
        logger.debug('Could not determine threshold voltage')
        return results

    # Analyse blips
    blips_results = count_blips(traces=traces,
                                sample_rate=sample_rate,
                                threshold_voltage=threshold_voltage,
                                t_skip=t_skip)
    results['blips'] = blips_results['blips']
    results['mean_low_blip_duration'] = blips_results['mean_low_blip_duration']
    results['mean_high_blip_duration'] = blips_results['mean_high_blip_duration']

    # minimum trace idx to include (to discard initial capacitor spike)
    start_idx = round(t_skip * sample_rate)

    if filter == 'low':
        # Filter all traces that do not start with low voltage
        filtered_traces_idx = edge_voltage(traces, edge='begin', state='low',
                                           start_idx=start_idx,
                                           threshold_voltage=threshold_voltage)
    elif filter == 'high':
        # Filter all traces that do not start with high voltage
        filtered_traces_idx = edge_voltage(traces, edge='begin', state='high',
                                           start_idx=start_idx,
                                           threshold_voltage=threshold_voltage)
    else:
        filtered_traces_idx = np.ones(len(traces), dtype=bool)

    results['filtered_traces_idx'] = filtered_traces_idx
    filtered_traces = traces[filtered_traces_idx]
    results['num_traces'] = len(filtered_traces)

    if len(filtered_traces) / len(traces) < min_filter_proportion:
        logger.debug(f'Not enough traces start {filter}')
        return results

    if t_read is not None:
        # Only use a time segment of each trace
        read_pts = int(round(t_read * sample_rate))
        if segment == 'begin':
            segmented_filtered_traces = filtered_traces[:, :read_pts]
        else:
            segmented_filtered_traces = filtered_traces[:, -read_pts:]
    else:
        segmented_filtered_traces = filtered_traces

    # Calculate up proportion of traces
    up_proportion = find_up_proportion(segmented_filtered_traces,
                                       start_idx=start_idx,
                                       threshold_voltage=threshold_voltage)
    results['up_proportion'] = up_proportion

    # Calculate ratio of traces that end up with low voltage
    idx_end_low = edge_voltage(segmented_filtered_traces,
                               edge='end',
                               state='low',
                               threshold_voltage=threshold_voltage)
    results['end_low'] = np.sum(idx_end_low) / len(segmented_filtered_traces)

    # Calculate ratio of traces that end up with high voltage
    idx_end_high = edge_voltage(segmented_filtered_traces,
                                edge='end',
                                state='high',
                                threshold_voltage=threshold_voltage)
    results['end_high'] = np.sum(idx_end_high) / len(segmented_filtered_traces)

    return results


def analyse_EPR(empty_traces: np.ndarray,
                plunge_traces: np.ndarray,
                read_traces: np.ndarray,
                sample_rate: float,
                t_skip: float,
                t_read: float,
                min_filter_proportion: float = 0.5) -> Dict[str, float]:
    """ Analyse an empty-plunge-read sequence

    Args:
        empty_traces: 2D array of acquisition traces in empty (ionized) state
        plunge_traces: 2D array of acquisition traces in plunge (neutral) state
        read_traces: 2D array of acquisition traces in read state
        sample_rate: acquisition sample rate (per second).
        t_skip: initial time to skip for each trace (ms).
        t_read: duration of each trace to use for calculating up_proportion etc.
            e.g. for a long trace, you want to compare up proportion of start
            and end segments.
        min_filter_proportion: minimum proportion of traces that satisfy filter.
            If below this value, up_proportion etc. are not calculated.

    Returns:
        Dict containing:
            fidelity_empty (float): proportion of empty traces that end ionized
                (high voltage). Traces are filtered out that do not start
                neutral (low voltage).
            voltage_difference_empty (float): voltage difference between high
                and low state in empty traces
            fidelity_load (float): proportion of plunge traces that end neutral
                (low voltage). Traces are filtered out that do not start ionized
                (high voltage).
            voltage_difference_load (float): voltage difference between high and
                low state in plunge traces
            up_proportion (float): proportion of read traces that have blips
                For each trace, only up to t_read is considered.
            dark_counts (float): proportion of read traces that have dark
                counts. For each trace, only the final t_read is considered.
            contrast (float): =up_proportion - dark_counts
            voltage_difference_read (float): voltage difference between high and
                low state in read traces
                blips (float): average blips per trace in read traces.
                mean_low_blip_duration (float): average duration in low state.
                mean_high_blip_duration (float): average duration in high state.
    """
    # Analyse empty stage
    results_empty = analyse_traces(traces=empty_traces,
                                   sample_rate=sample_rate,
                                   filter='low',
                                   min_filter_proportion=min_filter_proportion,
                                   t_skip=t_skip)

    # Analyse plunge stage
    results_load = analyse_traces(traces=plunge_traces,
                                  sample_rate=sample_rate,
                                  filter='high',
                                  min_filter_proportion=min_filter_proportion,
                                  t_skip=t_skip)

    # Analyse read stage
    results_read = analyse_traces(traces=read_traces,
                                  sample_rate=sample_rate,
                                  filter='low',
                                  min_filter_proportion=min_filter_proportion,
                                  t_skip=t_skip)
    results_read_begin = analyse_traces(traces=read_traces,
                                        sample_rate=sample_rate,
                                        filter='low',
                                        min_filter_proportion=min_filter_proportion,
                                        t_read=t_read,
                                        segment='begin',
                                        t_skip=t_skip)
    results_read_end = analyse_traces(traces=read_traces,
                                      sample_rate=sample_rate,
                                      t_read=t_read,
                                      segment='end',
                                      t_skip=t_skip)

    return {'fidelity_empty': results_empty['end_high'],
            'voltage_difference_empty': results_empty['voltage_difference'],

            'fidelity_load': results_load['end_low'],
            'voltage_difference_load': results_load['voltage_difference'],

            'up_proportion': results_read_begin['up_proportion'],
            'contrast': (results_read_begin['up_proportion'] -
                         results_read_end['up_proportion']),
            'dark_counts': results_read_end['up_proportion'],

            'voltage_difference_read': results_read['voltage_difference'],
            'blips': results_read['blips'],
            'mean_low_blip_duration': results_read['mean_low_blip_duration'],
            'mean_high_blip_duration': results_read['mean_high_blip_duration']}


def analyse_flips(up_proportions_arrs: List[np.ndarray],
                  threshold_up_proportion: Union[Sequence, float]) -> \
        Dict[str, Any]:
    """ Analyse flipping between NMR states

    For each up_proportion array, it will count the number of flips
    (transition between high/low up proportion) for each sample.

    If more than one up_proportion array is passed, combined flipping events
    between each pair of states is also considered (i.e. one goes from low to
    high up proportion, the other from high to low). Furthermore, filtering is
    also performed where flips are only counted if the nuclear state remains in
    the subspace spanned by the pair of states for all samples.

    Args:
        up_proportions_arrs: 3D Arrays of up proportions, calculated from
            `analyse_traces`. Up proportion is the proportion of traces that
            contain blips. First and second dimensions are arbitrary (can be a
            singleton), third dimension is samples.
        threshold_up_proportion: Threshold for when an up_proportion is high
            enough to count the nucleus as being in that state (consistently
            able to flip the electron). Can also be a tuple with two elements,
            in which case the first is a lower threshold, below which we can
            say the nucleus is not in the state, and the second is an upper
            threshold . If any up proportion is not in that state, it is in an
            undefined state, and not counted for flipping. For any filtering
            on up_proportion pairs, the whole row is considered invalid (NaN).

    Returns:
        Dict containing:
            flips(_{idx}) (int): Number of flips between high/low up_proportion.
                If more than one up_proportion_arr is provided, a zero-based
                index is added to specify the up_proportion_array.
                If an up_proportion is between the lower and upper threshold,
                it is not counted.
            flip_probability(_{idx}) (float): proportion of flips compared to
                maximum number of flips (samples - 1). If more than one
                up_proportion_arr is provided, a zero-based index is added to
                specify the up_proportion_array

        The following results are between neighbouring pairs of
        up_proportion_arrs (|idx1-idx2| == 1), and are only returned if more
        than one up_proportion_arr is given:
            'combined_flips_{idx1}{idx2}: combined flipping events between
                up_proportion_arrs idx1 and idx2, where one of the
                up_proportions switches from high to low, and the other from
                low to high.
            'combined_flip_probability_{idx1}{idx2}: Flipping probability of the
                combined flipping events.

            Additionally, each of the above results will have another result
            with the same name, but prepended with `filtered_`, and appended
            with `_{idx1}{idx2}` if not already present. Here, all the values
            are filtered out where the corresponding pair of up_proportion
            samples do not have exactly one high and one low for each sample.
            The values that do not satisfy the filter are set to np.nan.

            filtered_scans_{idx1}{idx2}: 2D bool array, True if pair of
                up_proportion rows remain in subspace
    """
    if isinstance(threshold_up_proportion, collections.Sequence):
        if len(threshold_up_proportion) != 2:
            raise SyntaxError(f'threshold_up_proportion must be either single '
                              'value, or two values (low and high threshold)')
        threshold_up_proportion_low = threshold_up_proportion[0]
        threshold_up_proportion_high = threshold_up_proportion[1]
    else:
        threshold_up_proportion_low = threshold_up_proportion
        threshold_up_proportion_high = threshold_up_proportion

    if not isinstance(up_proportions_arrs, np.ndarray):
        # Convert to numpy array to allow
        up_proportions_arrs = np.array(up_proportions_arrs)

    max_flips = up_proportions_arrs.shape[-1] - 1  # number of samples - 1

    results = {}

    # Boolean arrs equal to True if up proportion is above/below threshold
    with np.errstate(invalid='ignore'):
        # errstate used because up_proportions may contain NaN
        up_proportions_high = up_proportions_arrs > threshold_up_proportion_high
        up_proportions_low = up_proportions_arrs < threshold_up_proportion_low

    # State of up proportions by threshold (above: 1, below: -1, between: 0)
    state_arrs = np.zeros(up_proportions_arrs.shape)
    state_arrs[up_proportions_high] = 1
    state_arrs[up_proportions_low] = -1
    # results['state_arrs'] = state_arrs

    for f_idx, state_arr in enumerate(state_arrs):
        # TODO verify that sum is over correct axis
        flips = np.sum(np.abs(np.diff(state_arr)) == 2, axis=-1, dtype=float)
        # Flip probability by dividing flips by number of samples - 1
        flip_probability = flips / max_flips

        # Add suffix if more than one up_proportion array is provided
        suffix = f'_{f_idx}' if len(up_proportions_arrs) > 1 else ''
        results['flips' + suffix] = flips
        results['flip_probability' + suffix] = flip_probability

        # Only do this if more than one up_proportions is provided
        for f_idx2 in range(f_idx + 1, len(up_proportions_arrs)):
            state_arr2 = state_arrs[f_idx2]

            # Calculate relative states (default zero)
            relative_state_arr = np.zeros(state_arr.shape)
            # state1 low, state2 high: 1
            relative_state_arr[(state_arr2 - state_arr) == 2] = 1
            # state1 high, state2 low: -1
            relative_state_arr[(state_arr2 - state_arr) == -2] = -1
            # results[f'relative_state_arr_{f_idx}{f_idx2}'] = relative_state_arr

            # Combined flips, happens if relative_state_arr changes by 2
            # (high, low) -> (low, high) and (low, high) -> (high, low) equal 1
            combined_flips_arr = np.sum(
                np.abs(np.diff(relative_state_arr)) == 2,
                axis=-1, dtype=float)
            results[f'combined_flips_{f_idx}{f_idx2}'] = combined_flips_arr
            combined_flip_probability = combined_flips_arr / max_flips
            results[f'combined_flip_probability_{f_idx}{f_idx2}'] = \
                combined_flip_probability

            # Filter out scans that are not entirely in subspace i.e. not all
            # up proportion combinations have exactly one high, one low state
            # For this, the multiplied states must equal -1 (one +1, one -1)
            filtered_scans = np.all(state_arr * state_arr2 == -1, axis=-1)
            results[f'filtered_scans_{f_idx}{f_idx2}'] = filtered_scans

            # Add filtered version of combined flips
            for arr_name in [f'combined_flips_{f_idx}{f_idx2}',
                             f'combined_flip_probability_{f_idx}{f_idx2}']:
                filtered_arr = results[arr_name].copy()
                filtered_arr[~filtered_scans] = np.nan
                results['filtered_' + arr_name] = filtered_arr

        # Also filter flips and flip_probability
        # This is done afterwards since otherwise these arrays may not exist
        # e.g. flips_1 does not exist when filtered_scans_01 is created
        for arr_name in [f'flips_{f_idx}', f'flip_probability_{f_idx}']:
            # Only loop over nearest indices
            for f_idx2 in [f_idx - 1, f_idx + 1]:
                if f_idx2 < 0 or f_idx2 == len(up_proportions_arrs):
                    # f_idx is either first or last element
                    continue
                suffix = f'_{min(f_idx,f_idx2)}{max(f_idx, f_idx2)}'
                filtered_scans = results[f'filtered_scans' + suffix]
                filtered_arr = results[arr_name].copy()
                filtered_arr[~filtered_scans] = np.nan
                results[f'filtered_{arr_name}' + suffix] = filtered_arr

    return results