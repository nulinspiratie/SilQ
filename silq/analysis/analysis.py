import numpy as np
import matplotlib
import peakutils
import logging
from typing import Union, Dict, List, Sequence, Union
import collections
from matplotlib import pyplot as plt

from silq import config

from qcodes import MatPlot

__all__ = ['find_high_low', 'edge_voltage', 'find_up_proportion',
           'count_blips', 'analyse_traces', 'analyse_EPR', 'analyse_flips']

logger = logging.getLogger(__name__)


def find_high_low(traces: np.ndarray,
                  plot: bool = False,
                  threshold_peak: float = 0.02,
                  attempts: int = 8,
                  threshold_method: str = 'config',
                  min_voltage_difference: Union[float, str] = 'config',
                  threshold_requires_high_low: Union[bool, str] = 'config',
                  min_SNR: Union[float, None] = None):
    """ Find high and low voltages of traces using histograms

    This function determines the high and low voltages of traces by binning them
    into 30 bins, and trying to discern two peaks.
    Useful for determining the threshold value for measuring a blip.

    If no two peaks can be discerned after all attempts, None is returned for
    each of the returned dict keys except DC_voltage.

    Args:
        traces: 2D array of acquisition traces
        plot: Whether to plot the histograms
        threshold_peak: Threshold for discerning a peak. Will be varied if too
            many/few peaks are found
        attempts: Maximum number of attempts for discerning two peaks.
            Each attempt the threshold_peak is decreased/increased depending on
            if too many/few peaks were found
        threshold_method: Method used to determine the threshold voltage.
            Allowed methods are:

            * **mean**: average of high and low voltage.
            * **{n}\*std_low**: n standard deviations above mean low voltage,
              where n is a float (ignore slash in raw docstring).
            * **{n}\*std_high**: n standard deviations below mean high voltage,
              where n is a float (ignore slash in raw docstring).
            * **config**: Use threshold method provided in
              ``config.analysis.threshold_method`` (``mean`` if not specified)

        min_voltage_difference: minimum difference between high and low voltage.
            If not satisfied, all results are None.
            Will try to retrieve from config.analysis.min_voltage_difference,
            else defaults to 0.3V
        threshold_requires_high_low: Whether or not both a high and low voltage
            must be discerned before returning a threshold voltage.
            If set to False and threshold_method is not ``mean``, a threshold
            voltage is always determined, even if no two voltage peaks can be
            discerned. In this situation, there usually aren't any blips, or the
            blips are too short-lived to have a proper high current.
            When the threshold_method is ``std_low`` (``std_high``), the top
            (bottom) 20% of voltages are scrapped to ensure any short-lived blips
            with a high (low) current aren't included.
            The average is then taken over the remaining 80% of voltages, which
            is then the average low (high) voltage.
            Default is True.
            Can be set by config.analysis.threshold_requires_high_low
        min_SNR: Minimum SNR between high and low voltages required to determine
            a threshold voltage (default None).

    Returns:
        Dict[str, Any]:
        * **low** (float): Mean low voltage, ``None`` if two peaks cannot be
          discerned
        * **high** (float): Mean high voltage, ``None`` if no two peaks cannot
          be discerned
        * **threshold_voltage** (float): Threshold voltage for a blip. If SNR is
          below ``min_SNR`` or no two peaks can be discerned, returns ``None``.
        * **voltage_difference** (float): Difference between low and high
          voltage. If not two peaks can be discerned, returns ``None``.
        * **DC_voltage** (float): Average voltage of traces.
    """
    if attempts < 1:
        raise ValueError(
            f'Attempts {attempts} to find high and low voltage must be at least 1'
        )

    # Retrieve properties from config.analysis
    analysis_config = config.get('analysis', {})
    if threshold_method == 'config':
        threshold_method = analysis_config.get('threshold_method', 'mean')
    if min_voltage_difference == 'config':
        min_voltage_difference = analysis_config.get('min_voltage_difference', 0.3)
    if threshold_requires_high_low == 'config':
        threshold_requires_high_low = analysis_config.get('threshold_requires_high_low', True)
    if min_SNR is None:
        min_SNR = analysis_config.get('min_SNR', None)

    # Calculate DC (mean) voltage
    DC_voltage = np.mean(traces)

    # Perform a histogram over all voltages in all traces. These bins will be
    # used to determine two peaks, corresponding to low/high voltage
    hist, bin_edges = np.histogram(np.ravel(traces), bins=30)

    # Determine minimum number of bins between successive histogram peaks
    if min_voltage_difference is not None:
        min_dist = int(np.ceil(min_voltage_difference / np.diff(bin_edges)[0]))
    else:
        min_dist = 5

    # Find two peaks by changing the threshold dependent on the number of peaks foudn
    for k in range(attempts):
        peaks_idx = np.sort(peakutils.indexes(hist, thres=threshold_peak,
                                              min_dist=min_dist))
        if len(peaks_idx) == 2:
            break
        elif len(peaks_idx) == 1:
            # print('One peak found instead of two, lowering threshold')
            threshold_peak /= 1.5
        elif len(peaks_idx) > 2:
            # print(f'Found {len(peaks_idx)} peaks instead of two, '
            #        'increasing threshold')
            threshold_peak *= 1.5
    else:  # Could not identify two peaks after all attempts
        results = {
            'low': None,
            'high': None,
            'threshold_voltage': None,
            'voltage_difference': None,
            'DC_voltage': DC_voltage
        }

        if not threshold_requires_high_low and threshold_method != 'mean':
            # Still return threshold voltage even though no two peaks were observed
            # Use {factor}*std_low/high to determine threshold voltage
            factor = float(threshold_method.split('*')[0])
            voltages = sorted(traces.flatten())
            if threshold_method.endswith('std_low'):
                # Remove top 20 percent (high voltage)
                cutoff_slice = slice(None, int(0.8 * len(voltages)))
            else:
                factor *= -1  # standard deviations should be subtracted from mean
                # Remove bottom 20 percent of voltages (low voltage)
                cutoff_slice = slice(int(0.8 * len(voltages)), None)
            voltages_cutoff = voltages[cutoff_slice]
            voltage_mean = np.mean(voltages_cutoff)
            voltage_std = np.std(voltages_cutoff)
            threshold_voltage = voltage_mean + factor * voltage_std
            results['threshold_voltage'] = threshold_voltage

        return results

    # Find mean voltage, used to determine which points are low/high
    # Note that this is slightly odd, since we might use another threshold_method
    # later on to distinguish between high and low voltage
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
        threshold_voltage = (high['mean'] + low['mean']) / 2
    elif threshold_method.endswith('std_low'):
        # Threshold_method is {factor} standard deviations above low mean
        factor = float(threshold_method.split('*')[0])
        threshold_voltage = low['mean'] + factor * low['std']
    elif threshold_method.endswith('std_high'):
        # Threshold_method is {factor} standard deviations below high mean
        factor = float(threshold_method.split('*')[0])
        threshold_voltage = high['mean'] - factor * high['std']
    else:
        raise RuntimeError(f'Threshold method {threshold_method} not understood')

    SNR = voltage_difference / np.sqrt(high['std'] ** 2 + low['std'] ** 2)

    if min_SNR is not None and SNR < min_SNR:
        logger.info(f'Signal to noise ratio {SNR} is too low')
        threshold_voltage = None

    # Plotting
    if plot is not False:
        if plot is True:
            plt.figure()
        else:
            plt.sca(plot)
        for k, signal in enumerate([low, high]):
            sub_hist, sub_bin_edges = np.histogram(np.ravel(signal['traces']), bins=10)
            plt.bar(sub_bin_edges[:-1], sub_hist, width=0.05, color='bg'[k])
            if k < len(peaks_idx):
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
        edge: which side of traces to test, either ``begin`` or ``end``
        state: voltage that the edge must have, either ``low`` or ``high``
        threshold_voltage: threshold voltage for a ``high`` voltage (blip).
            If not specified, ``find_high_low`` is used to determine threshold
        points: Number of data points to average over to determine state
        start_idx: index of first point to use. Useful if there is some
            capacitive voltage spike occuring at the start.
            Only used if edge is ``begin``.

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
        threshold_voltage: threshold voltage for a ``high`` voltage (blip).
            If not specified, ``find_high_low`` is used to determine threshold
        return_array: whether to return the boolean array or the up proportion
        start_idx: index of first point to use. Useful if there is some
            capacitive voltage spike occuring at the start.
            Only used if edge is ``begin``
        filter_window: number of points of smoothing (0 means no smoothing)

    Returns:

        if return_array is False
            (float) The proportion of traces with a blip
        else
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
    traces_up_electron = np.array([np.any(trace[start_idx:] > threshold_voltage)
                                   for trace in traces])

    if not return_array:
        return sum(traces_up_electron) / len(traces)
    else:
        return traces_up_electron


def count_blips(traces: np.ndarray,
                threshold_voltage: float,
                sample_rate: float,
                t_skip: float,
                ignore_final: bool = False):
    """ Count number of blips and durations in high/low state.

    Args:
        traces: 2D array of acquisition traces.
        threshold_voltage: Threshold voltage for a ``high`` voltage (blip).
        sample_rate: Acquisition sample rate (per second).
        t_skip: Initial time to skip for each trace (ms).

    Returns:
        Dict[str, Any]:
        * **blips** (float): Number of blips per trace.
        * **blips_per_second** (float): Number of blips per second.
        * **low_blip_duration** (np.ndarray): Durations in low-voltage state.
        * **high_blip_duration** (np.ndarray): Durations in high-voltage state.
        * **mean_low_blip_duration** (float): Average duration in low state.
        * **mean_high_blip_duration** (float): Average duration in high state.
    """
    low_blip_pts, high_blip_pts = [], []
    start_idx = round(t_skip * sample_rate)

    blip_events = [[] for _ in range(len(traces))]
    for k, trace in enumerate(traces):
        idx = start_idx
        trace_above_threshold = trace > threshold_voltage
        trace_below_threshold = ~trace_above_threshold
        while idx < len(trace):
            if trace[idx] < threshold_voltage:
                next_idx = np.argmax(trace_above_threshold[idx:])
                blip_list = low_blip_pts
            else:
                next_idx = np.argmax(trace_below_threshold[idx:])
                blip_list = high_blip_pts

            if next_idx == 0:  # Reached end of trace
                next_idx = len(trace) - idx
                blip_list.append(next_idx)
                if not ignore_final:
                    blip_events[k].append((int(trace[idx] >= threshold_voltage), next_idx))
                break
            else:
                blip_list.append(next_idx)
                blip_events[k].append((int(trace[idx] >= threshold_voltage), next_idx))
                idx += next_idx

    low_blip_durations = np.array(low_blip_pts) / sample_rate
    high_blip_durations = np.array(high_blip_pts) / sample_rate

    mean_low_blip_duration = np.NaN if not len(low_blip_durations) \
        else np.mean(low_blip_durations)
    mean_high_blip_duration = np.NaN if not len(high_blip_durations) \
        else np.mean(high_blip_durations)

    blips = len(low_blip_durations) / len(traces)

    duration = len(traces[0]) / sample_rate
    return {'blips': blips,
            'blip_events': blip_events,
            'blips_per_second': blips / duration,
            'low_blip_durations': low_blip_durations,
            'high_blip_durations': high_blip_durations,
            'mean_low_blip_duration': mean_low_blip_duration,
            'mean_high_blip_duration': mean_high_blip_duration}


def analyse_traces(traces: np.ndarray,
                   sample_rate: float,
                   filter: Union[str, None] = None,
                   min_filter_proportion: float = 0.5,
                   t_skip: float = 0,
                   t_read: Union[float, None] = None,
                   segment: str = 'begin',
                   threshold_voltage: Union[float, None] = None,
                   threshold_method: str = 'config',
                   plot: Union[bool, matplotlib.axis.Axis] = False,
                   plot_high_low: Union[bool, matplotlib.axis.Axis] = False):
    """ Analyse voltage, up proportions, and blips of acquisition traces

    Args:
        traces: 2D array of acquisition traces.
        sample_rate: acquisition sample rate (per second).
        filter: only use traces that begin in low or high voltage.
            Allowed values are ``low``, ``high`` or ``None`` (do not filter).
        min_filter_proportion: minimum proportion of traces that satisfy filter.
            If below this value, up_proportion etc. are not calculated.
        t_skip: initial time to skip for each trace (ms).
        t_read: duration of each trace to use for calculating up_proportion etc.
            e.g. for a long trace, you want to compare up proportion of start
            and end segments.
        segment: Use beginning or end of trace for ``t_read``.
            Allowed values are ``begin`` and ``end``.
        threshold_voltage: threshold voltage for a ``high`` voltage (blip).
            If not specified, ``find_high_low`` is used to determine threshold.
        threshold_method: Method used to determine the threshold voltage.
            Allowed methods are:

            * **mean**: average of high and low voltage.
            * **{n}\*std_low**: n standard deviations above mean low voltage,
              where n is a float (ignore slash in raw docstring).
            * **{n}\*std_high**: n standard deviations below mean high voltage,
              where n is a float (ignore slash in raw docstring).
            * **config**: Use threshold method provided in
              ``config.analysis.threshold_method`` (``mean`` if not specified)

        plot: Whether to plot traces with results.
            If True, will create a MatPlot object and add results.
            Can also pass a MatPlot axis, in which case that will be used.
            Each trace is preceded by a block that can be green (measured blip
            during start), red (no blip measured), or white (trace was filtered
            out).

    Returns:
        Dict[str, Any]:
        * **up_proportion** (float): proportion of traces that has a blip
        * **end_high** (float): proportion of traces that end with high voltage
        * **end_low** (float): proportion of traces that end with low voltage
        * **num_traces** (int): Number of traces that satisfy filter
        * **filtered_traces_idx** (np.ndarray): 1D bool array,
          True if that trace satisfies filter
        * **voltage_difference** (float): voltage difference between high and
          low voltages
        * **average_voltage** (float): average voltage over all traces
        * **threshold_voltage** (float): threshold voltage for counting a blip
          (high voltage). Is calculated if not provided as input arg.
        * **blips** (float): average blips per trace.
        * **mean_low_blip_duration** (float): average duration in low state
        * **mean_high_blip_duration** (float): average duration in high state

    Note:
        If no threshold voltage is provided, and no two peaks can be discerned,
            all results except average_voltage are set to an initial value
            (either 0 or undefined)
        If the filtered trace proportion is less than min_filter_proportion,
            the results ``up_proportion``, ``end_low``, ``end_high`` are set to an
            initial value
    """
    assert filter in [None, 'low',
                      'high'], 'filter must be None, `low`, or `high`'

    assert segment in ['begin',
                       'end'], 'segment must be either `begin` or `end`'

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

    # minimum trace idx to include (to discard initial capacitor spike)
    start_idx = round(t_skip * sample_rate)

    if plot is not False:  # Create plot for traces
        ax = MatPlot()[0] if plot is True else plot
        t_list = np.linspace(0, len(traces[0]) / sample_rate, len(traces[0]))
        print(ax.get_xlim())
        ax.add(traces, x=t_list, y=np.arange(len(traces), dtype=float),
               cmap='seismic')
        print(ax.get_xlim())
        # Modify x-limits to add blips information
        xlim = ax.get_xlim()
        xpadding = 0.05 * (xlim[1] - xlim[0])
        if segment == 'begin':
            xpadding_range = [-xpadding + xlim[0], xlim[0]]
            ax.set_xlim(-xpadding + xlim[0], xlim[1])
        else:
            xpadding_range = [xlim[1], xlim[1] + xpadding]
            ax.set_xlim(xlim[0], xlim[1] + xpadding)

    if threshold_voltage is None:  # Calculate threshold voltage if not provided
        # Histogram trace voltages to find two peaks corresponding to high and low
        high_low_results = find_high_low(traces[:, start_idx:],
                                         threshold_method=threshold_method,
                                         plot=plot_high_low)
        results['high_low_results'] = high_low_results
        results['voltage_difference'] = high_low_results['voltage_difference']
        # Use threshold voltage from high_low_results
        threshold_voltage = high_low_results['threshold_voltage']

        results['threshold_voltage'] = threshold_voltage

        if threshold_voltage is None:
            logger.debug('Could not determine threshold voltage')
            if plot is not False:
                ax.text(np.mean(xlim), len(traces) + 0.5,
                        'Unknown threshold voltage',
                        horizontalalignment='center')
            return results
    else:
        # We don't know voltage difference since we skip a high_low measure.
        results['voltage_difference'] = None

    # Analyse blips (disabled because it's very slow)
    # blips_results = count_blips(traces=traces,
    #                             sample_rate=sample_rate,
    #                             threshold_voltage=threshold_voltage,
    #                             t_skip=t_skip)
    # results['blips'] = blips_results['blips']
    # results['mean_low_blip_duration'] = blips_results['mean_low_blip_duration']
    # results['mean_high_blip_duration'] = blips_results['mean_high_blip_duration']

    if filter == 'low':  # Filter all traces that do not start with low voltage
        filtered_traces_idx = edge_voltage(traces, edge='begin', state='low',
                                           start_idx=start_idx,
                                           threshold_voltage=threshold_voltage)
    elif filter == 'high':  # Filter all traces that do not start with high voltage
        filtered_traces_idx = edge_voltage(traces, edge='begin', state='high',
                                           start_idx=start_idx,
                                           threshold_voltage=threshold_voltage)
    else:  # Do not filter traces
        filtered_traces_idx = np.ones(len(traces), dtype=bool)

    results['filtered_traces_idx'] = filtered_traces_idx
    filtered_traces = traces[filtered_traces_idx]
    results['num_traces'] = len(filtered_traces)

    if len(filtered_traces) / len(traces) < min_filter_proportion:
        logger.debug(f'Not enough traces start {filter}')

        if plot is not False:
            ax.pcolormesh(xpadding_range, np.arange(len(traces) + 1) - 0.5,
                          filtered_traces.reshape(1, -1), cmap='RdYlGn')
            ax.text(np.mean(xlim), len(traces) + 0.5,
                    f'filtered traces: {len(filtered_traces)} / {len(traces)} = '
                    f'{len(filtered_traces) / len(traces):.2f} < {min_filter_proportion}',
                    horizontalalignment='center')
        return results

    if t_read is not None:  # Only use a time segment of each trace
        read_pts = int(round(t_read * sample_rate))
        if segment == 'begin':
            segmented_filtered_traces = filtered_traces[:, :read_pts]
        else:
            segmented_filtered_traces = filtered_traces[:, -read_pts:]
    else:
        segmented_filtered_traces = filtered_traces

    # Calculate up proportion of traces
    up_proportion_idxs = find_up_proportion(segmented_filtered_traces,
                                            start_idx=start_idx,
                                            threshold_voltage=threshold_voltage,
                                            return_array=True)
    results['up_proportion'] = sum(up_proportion_idxs) / len(traces)

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

    if plot is not False:
        # Plot information on up proportion
        up_proportion_arr = 2 * up_proportion_idxs - 1
        up_proportion_arr[~filtered_traces_idx] = 0
        up_proportion_arr = up_proportion_arr.reshape(-1, 1)  # Make array 2D

        mesh = ax.pcolormesh(xpadding_range, np.arange(len(traces) + 1) - 0.5,
                             up_proportion_arr, cmap='RdYlGn')
        mesh.set_clim(-1, 1)

        # Add vertical line for t_read
        ax.vlines(t_read, -0.5, len(traces + 0.5), lw=2, linestyle='--',
                  color='orange')
        ax.text(t_read, len(traces) + 0.5, f't_read={t_read*1e3} ms',
                horizontalalignment='center', verticalalignment='bottom')
        ax.text(t_skip, len(traces) + 0.5, f't_skip={t_skip*1e3} ms',
                horizontalalignment='center', verticalalignment='bottom')

    return results


def analyse_EPR(empty_traces: np.ndarray,
                plunge_traces: np.ndarray,
                read_traces: np.ndarray,
                sample_rate: float,
                t_skip: float,
                t_read: float,
                min_filter_proportion: float = 0.5,
                threshold_voltage: Union[float, None] = None,
                filter_traces=True,
                plot: bool = False):
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
        threshold_voltage: threshold voltage for a ``high`` voltage (blip).
            If not specified, ``find_high_low`` is used to determine threshold.

    Returns:
        Dict[str, float]:
        * **fidelity_empty** (float): proportion of empty traces that end
          ionized (high voltage). Traces are filtered out that do not start
          neutral (low voltage).
        * **voltage_difference_empty** (float): voltage difference between high
          and low state in empty traces
        * **fidelity_load** (float): proportion of plunge traces that end
          neutral (low voltage). Traces are filtered out that do not start
          ionized (high voltage).
        * **voltage_difference_load** (float): voltage difference between high
          and low state in plunge traces
        * **up_proportion** (float): proportion of read traces that have blips
          For each trace, only up to t_read is considered.
        * **dark_counts** (float): proportion of read traces that have dark
          counts. For each trace, only the final t_read is considered.
        * **contrast** (float): =up_proportion - dark_counts
        * **voltage_difference_read** (float): voltage difference between high
          and low state in read traces
        * **blips** (float): average blips per trace in read traces.
        * **mean_low_blip_duration** (float): average duration in low state.
        * **mean_high_blip_duration** (float): average duration in high state.
    """
    if plot is True:
        plot = MatPlot(subplots=3)
        plot[0].set_title('Empty')
        plot[1].set_title('Plunge')
        plot[2].set_title('Read long')
    elif plot is False:
        plot = [False] * 3

    # Analyse empty stage
    results_empty = analyse_traces(traces=empty_traces,
                                   sample_rate=sample_rate,
                                   filter='low' if filter_traces else None,
                                   min_filter_proportion=min_filter_proportion,
                                   threshold_voltage=threshold_voltage,
                                   t_skip=t_skip,
                                   plot=plot[0])

    # Analyse plunge stage
    results_load = analyse_traces(traces=plunge_traces,
                                  sample_rate=sample_rate,
                                  filter='high' if filter_traces else None,
                                  min_filter_proportion=min_filter_proportion,
                                  threshold_voltage=threshold_voltage,
                                  t_skip=t_skip,
                                  plot=plot[1])

    # Analyse read stage
    results_read = analyse_traces(traces=read_traces,
                                  sample_rate=sample_rate,
                                  filter='low' if filter_traces else None,
                                  min_filter_proportion=min_filter_proportion,
                                  threshold_voltage=threshold_voltage,
                                  t_skip=t_skip)
    results_read_begin = analyse_traces(traces=read_traces,
                                        sample_rate=sample_rate,
                                        filter='low' if filter_traces else None,
                                        min_filter_proportion=min_filter_proportion,
                                        threshold_voltage=threshold_voltage,
                                        t_read=t_read,
                                        segment='begin',
                                        t_skip=t_skip,
                                        plot=plot[2])
    results_read_end = analyse_traces(traces=read_traces,
                                      sample_rate=sample_rate,
                                      t_read=t_read,
                                      threshold_voltage=threshold_voltage,
                                      segment='end',
                                      t_skip=t_skip,
                                      plot=plot[2])

    return {'fidelity_empty': results_empty['end_high'],
            'voltage_difference_empty': results_empty['voltage_difference'],

            'fidelity_load': results_load['end_low'],
            'voltage_difference_load': results_load['voltage_difference'],

            'up_proportion': results_read_begin['up_proportion'],
            'contrast': (results_read_begin['up_proportion'] -
                         results_read_end['up_proportion']),
            'dark_counts': results_read_end['up_proportion'],

            'voltage_difference_read': results_read['voltage_difference'],
            'voltage_average_read': results_read['average_voltage'],
            'num_traces': results_read['num_traces'],
            'filtered_traces_idx': results_read['filtered_traces_idx'],
            'blips': results_read['blips'],
            'mean_low_blip_duration': results_read['mean_low_blip_duration'],
            'mean_high_blip_duration': results_read['mean_high_blip_duration']}


def analyse_threshold_up_proportion(up_proportions_arrs: np.ndarray,
                                    shots_per_frequency: int):
    proportion_space = np.linspace(0, 1, num=shots_per_frequency + 1,
                                   endpoint=True)
    up_proportions_arrs = up_proportions_arrs.reshape(1, -1)
    kernel = gaussian_kde(up_proportions_arrs)
    gaussian_up_proportions = kernel(proportion_space)
    peak_idxs = peakutils.peak.indexes(gaussian_up_proportions,
                                       thres=0.5/up_proportions_arrs.shape[-1],
                                       min_dist=shots_per_frequency/5)
    if len(peak_idxs) == 0:
        logger.debug(f'Adaptive thresholding routine: 0 peaks were '
                     f'found, using threshold 0.5')
        true_idx = round((len(proportion_space) - 1) / 2)
    elif len(peak_idxs) == 1:
        if (len(proportion_space) - peak_idxs[0]) / len(proportion_space) > 0.5:
            true_slice = slice(*[peak_idxs[0], len(proportion_space)])
            true_idx = np.argmin(np.round(gaussian_up_proportions[true_slice], 4)) + \
                       peak_idxs[0]
        else:
            true_slice = slice(*[0, peak_idxs[0]])
            true_idx = np.argmin(np.round(gaussian_up_proportions[true_slice], 4))
    else:
        true_slice = slice(*[min(peak_idxs), max(peak_idxs)])
        true_idx = np.argmin(np.round(gaussian_up_proportions[true_slice], 4)) + \
                   peak_idxs[0]

    threshold_up_proportion = proportion_space[true_idx]

    return threshold_up_proportion


def analyse_flips(
        up_proportions_arrs: List[np.ndarray],
        threshold_up_proportion: Union[Sequence, float, None],
        shots_per_frequency: int,
        labels: List[str] = None,
        label_pairs: List[List[str]] = 'neighbouring'
):
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
        labels: Optional labels for the different up proportions.
            If not provided, zero-based indices are used
        label_pairs: Optional selection of up_proportion pairs to compare.
            These are used for the combined_flips/flip_probability and for
            filtered_combined_flips/flip_probability.
            Can be one of the following:

                List of string pairs: Each string in a pair must correspond to
                    a label
                'neighbouring': Only compare neighbouring pairs
                 'full': compare all possible pairs

    Returns:
        Dict[str, Any]:
        The following results are returned if a threshold_up_proportion is
        provided:

        * **flips(_{idx})** (int): Flips between high/low up_proportion.
          If more than one up_proportion_arr is provided, a zero-based index is
          added to specify the up_proportion_array. If an up_proportion is
          between the lower and upper threshold, it is not counted.
        * **flip_probability(_{idx})** (float): proportion of flips compared to
          maximum flips (samples - 1). If more than one up_proportion_arr is
          provided, a zero-based index is added to specify the
          up_proportion_array.

        The following results are between pairs of up_proportions, and can be
        specified with ``label_pairs``, and are only returned if more
        than one up_proportion_arr is given:

        * **possible_flips_{label1}_{label2}**: Number of possible flipping events,
          i.e. where successive pairs both satisfy one up_proportion being above
          and one below threshold.
        * **combined_flips_{label1}_{label2}**: combined flipping events between
          up_proportion_arrs label1 and label2, where one of the
          up_proportions switches from high to low, and the other from
          low to high.
        * **combined_flip_probability_{idx1}{idx2}**: Flipping probability
          of the combined flipping events (combined_flips / possible_flips).

        Additionally, each of the above results will have another result with
        the same name, but prepended with ``filtered_``. Here, all the values are
        filtered out where the corresponding pair of up_proportion samples do
        not have exactly one high and one low for each sample. The values that
        do not satisfy the filter are set to np.nan.
    """
    results = {}

    if labels is None:
        labels = [str(k) for k in range(len(up_proportions_arrs))]
        separator = ''  # No separator should be used for combined flips labels
    else:
        separator = '_'  # Use separator to distinguish labels in combined flips

    if not isinstance(up_proportions_arrs, np.ndarray):  # Convert to numpy array
        up_proportions_arrs = np.array(up_proportions_arrs)

    up_proportions_dict = {label: arr for label, arr in zip(labels, up_proportions_arrs)}

    max_flips = up_proportions_arrs.shape[-1] - 1  # number of samples - 1

    # Determine threshold_up_proportion_low/high
    if (len(up_proportions_arrs) == 1) and (threshold_up_proportion is None):
        threshold_up_proportion = analyse_threshold_up_proportion(
            up_proportions_arrs=up_proportions_arrs,
            shots_per_frequency=shots_per_frequency)
        threshold_low = threshold_up_proportion
        threshold_high = threshold_up_proportion
    elif isinstance(threshold_up_proportion, collections.Sequence):
        if len(threshold_up_proportion) != 2:
            raise SyntaxError(f'threshold_up_proportion must be either single '
                              'value, or two values (low and high threshold)')
        threshold_low = threshold_up_proportion[0]
        threshold_high = threshold_up_proportion[1]
    else:
        threshold_low = threshold_up_proportion
        threshold_high = threshold_up_proportion

    # First calculate flips/flip_probability for individual up_proportions
    # Note that we skip this step if threshold_up_proportion is None
    if threshold_up_proportion is not None:
        # State of up proportions by threshold (above: 1, below: -1, between: 0)
        with np.errstate(invalid='ignore'): # ignore errors (up_proportions may contain NaN)
            state_arrs = np.zeros(up_proportions_arrs.shape)
            state_arrs[up_proportions_arrs > threshold_high] = 1
            state_arrs[up_proportions_arrs < threshold_low] = -1

        for k, state_arr in enumerate(state_arrs):
            flips = np.sum(np.abs(np.diff(state_arr)) == 2, axis=-1, dtype=float)
            # Flip probability by dividing flips by number of samples - 1
            flip_probability = flips / max_flips

            # Add suffix if more than one up_proportion array is provided
            suffix = f'_{labels[k]}' if len(up_proportions_arrs) > 1 else ''
            results['flips' + suffix] = flips
            results['flip_probability' + suffix] = flip_probability

    # Determine combined flips/flip_probability
    if len(up_proportions_arrs) > 1:
        if label_pairs == 'neighbouring':
            # Only look at neighbouring combined flips
            label_pairs = list(zip(labels[:-1], labels[1:]))
        elif label_pairs == 'full':
            label_pairs = []
            for k, label1 in enumerate(labels):
                for label2 in labels[k+1:]:
                    label_pairs.append((label1, label2))

        for label1, label2 in label_pairs:
            up_proportions_1 = up_proportions_dict[label1]
            up_proportions_2 = up_proportions_dict[label2]
            suffix = f'_{label1}{separator}{label2}'

            # If no threshold is provided, it is dynamically chosen as the value
            # for which every pair of up proportions has exactly one value above
            # and one below this value. If no such value exists, all combined
            # results are NaN
            if threshold_up_proportion is None:
                up_max = np.max([up_proportions_1, up_proportions_2], axis=0)
                up_min = np.min([up_proportions_1, up_proportions_2], axis=0)
                if max(up_min) < min(up_max):
                    threshold_up_proportion = np.mean([min(up_max), max(up_min)])
                    threshold_high = threshold_up_proportion
                    threshold_low = threshold_up_proportion
                else:
                    # No threshold can be determined, do not proceed with this pair
                    results['combined_flips' + suffix] = np.nan
                    results['combined_flip_probability' + suffix] = np.nan
                    results['filtered_combined_flips' + suffix] = np.nan
                    results['filtered_combined_flip_probability' + suffix] = np.nan
                    results['possible_flips' + suffix] = np.nan
                    continue

            # Determine state that are above/below threshold.
            state_arrs = []
            for k, up_proportions in enumerate([up_proportions_1, up_proportions_2]):
                # Boolean arrs equal to True if up proportion is above/below threshold
                with np.errstate(invalid='ignore'): # ignore errors if up_proportions contains NaN
                    state_arr = np.zeros(up_proportions.shape)
                    state_arr[up_proportions > threshold_high] = 1
                    state_arr[up_proportions < threshold_low] = -1
                    above_low = threshold_low <= up_proportions
                    below_high = up_proportions <= threshold_high
                    state_arr[above_low & below_high] = np.nan
                    state_arrs.append(state_arr)

            # Calculate relative states, with possible values:
            # -2: state1 high,       state2 low
            # 2:  state1 low,        state2 high
            # NaN: at least one of the two states is undefined (between thresholds)
            relative_state_arr = state_arrs[1] - state_arrs[0]

            # Combined flips, happens if relative_state_arr changes by 4
            # (high, low) -> (low, high) and (low, high) -> (high, low)
            combined_flips = np.sum(
                np.abs(np.diff(relative_state_arr)) == 4,
                axis=-1, dtype=float
            )
            results['combined_flips' + suffix] = combined_flips

            # Number of possible flips, i.e. where successive up_proportion pairs
            # both satisfy one being above and one below threshold
            possible_flips = np.sum(
                ~np.isnan(np.diff(relative_state_arr)),
                axis=-1, dtype=float
            )
            results['possible_flips' + suffix] = possible_flips
            if possible_flips > 0:
                combined_flip_probability = combined_flips / possible_flips
            else:
                combined_flip_probability = np.nan
            results['combined_flip_probability' + suffix]  = combined_flip_probability

            # Check if all up_proportion pairs satisfy threshold condition
            if possible_flips == max_flips:
                results['filtered_combined_flips' + suffix] = combined_flips
                results['filtered_combined_flip_probability' + suffix] = combined_flip_probability
            else:
                results['filtered_combined_flips' + suffix] = np.nan
                results['filtered_combined_flip_probability' + suffix] = np.nan

            results['threshold_up_proportion_combined'] = (threshold_low, threshold_high)

    return results
