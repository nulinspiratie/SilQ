import numpy as np
import peakutils
import logging

__all__ = ['smooth', 'find_high_low', 'edge_voltage', 'find_up_proportion',
           'count_blips', 'analyse_traces', 'analyse_EPR', 'analyse_NMR']

logger = logging.getLogger(__name__)

from silq import config
if 'analysis' not in config:
    config['analysis'] = {}
analysis_config = config['analysis']


def smooth(x, window_len=11, window='hanning'):
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


def find_high_low(traces, plot=False, threshold_peak=0.02, attempts=8,
                  threshold_method='config', min_SNR=None):
    # Calculate DC (mean) voltage
    DC_voltage = np.mean(traces)

    # Determine threshold method
    if threshold_method == 'config':
        threshold_method = analysis_config.get('threshold_method', 'mean')

    hist, bin_edges = np.histogram(np.ravel(traces), bins=30)

    # Find two peaks
    for k in range(attempts):
        peaks_idx = np.sort(peakutils.indexes(hist, thres=threshold_peak, min_dist=5))
        if len(peaks_idx) == 2:
            break
        elif len(peaks_idx) == 1:
            # print('One peak found instead of two, lowering threshold')
            threshold_peak /= 1.5
        elif len(peaks_idx) > 2:
            # print('Found {} peaks instead of two, increasing threshold'.format(len(peaks_idx)))
            threshold_peak *= 1.5
        else:
            return {'low': None,
                    'high': None,
                    'threshold_voltage': None,
                    'voltage_difference': None,
                    'DC_voltage': DC_voltage}

    # Find mean voltage, used to determine which points are low/high
    mean_voltage_idx = int(round(np.mean(peaks_idx)))
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

def edge_voltage(traces, edge, state, threshold_voltage=None, points=5,
                 start_point=0):
    assert edge in ['begin', 'end'], \
        'Edge {} must be either "begin" or "end"'.format(edge)
    assert state in ['low', 'high'], \
        'State {} must be either "low" or "high"'.format(state)

    if edge == 'begin':
        if start_point > 0:
            idx_list = slice(start_point, start_point + points)
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

def find_up_proportion(traces, threshold_voltage=None, return_mean=True,
                       start_point=0, filter_window=0):
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
    traces_up_electron = [np.any(trace[start_point:] > threshold_voltage)
                          for trace in traces]

    if return_mean:
        return sum(traces_up_electron) / len(traces)
    else:
        return traces_up_electron


def count_blips(traces, threshold_voltage, sample_rate, t_skip):
    low_blip_pts, high_blip_pts = [], []
    start_idx = round(t_skip * 1e-3 * sample_rate)

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

    low_blip_durations = np.array(low_blip_pts) / sample_rate * 1e3
    high_blip_durations = np.array(high_blip_pts) / sample_rate * 1e3

    duration = len(traces[0]) / sample_rate
    return {'blips': len(low_blip_durations),
            'blips_per_second': len(low_blip_durations) / duration,
            'low_blip_durations': low_blip_durations,
            'high_blip_durations': high_blip_durations,
            'mean_low_blip_duration': np.mean(low_blip_durations),
            'mean_high_blip_duration': np.mean(high_blip_durations)}


def analyse_traces(traces, sample_rate, filter=None, min_trace_perc=0.5,
                   t_skip=0, t_read=None, segment='begin',
                   threshold_voltage=None, threshold_method='config'):
    if filter not in [None, 'low', 'high']:
        raise SyntaxError('filter must be either None, `low`, or `high`')

    if segment not in ['begin', 'end']:
        raise SyntaxError('segment must be either `begin` or `end`')

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
    start_idx = round(t_skip * 1e-3 * sample_rate)

    if filter == 'low':
        # Filter all traces that do not start with low voltage
        filtered_traces_idx = edge_voltage(traces, edge='begin', state='low',
                                           start_point=start_idx,
                                           threshold_voltage=threshold_voltage)
    elif filter == 'high':
        # Filter all traces that do not start with high voltage
        filtered_traces_idx = edge_voltage(traces, edge='begin', state='high',
                                           start_point=start_idx,
                                           threshold_voltage=threshold_voltage)
    else:
        filtered_traces_idx = np.ones(len(traces), dtype=bool)

    results['filtered_traces_idx'] = filtered_traces_idx
    filtered_traces = traces[filtered_traces_idx]
    results['num_traces'] = len(filtered_traces)

    if len(filtered_traces) / len(traces) < min_trace_perc:
        logger.debug(f'Not enough traces start {filter}')
        return results

    if t_read is not None:
        # Only use a time segment of each trace
        read_pts = int(round(t_read * 1e-3 * sample_rate))
        if segment == 'begin':
            segmented_filtered_traces = filtered_traces[:, :read_pts]
        else:
            segmented_filtered_traces = filtered_traces[:, -read_pts:]
    else:
        segmented_filtered_traces = filtered_traces

    # Calculate up proportion of traces
    up_proportion = find_up_proportion(segmented_filtered_traces,
                                       start_point=start_idx,
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


def analyse_EPR(empty_traces, plunge_traces, read_traces,
                sample_rate, t_read, t_skip,
                min_trace_perc=0.5):
    # Analyse empty stage
    results_empty = analyse_traces(traces=empty_traces,
                                   sample_rate=sample_rate,
                                   filter='low',
                                   min_trace_perc=min_trace_perc,
                                   t_skip=t_skip)

    # Analyse plunge stage
    results_load = analyse_traces(traces=plunge_traces,
                                  sample_rate=sample_rate,
                                  filter='high',
                                  min_trace_perc=min_trace_perc,
                                  t_skip=t_skip)

    # Analyse read stage
    results_read = analyse_traces(traces=read_traces,
                                  sample_rate=sample_rate,
                                  filter='low',
                                  min_trace_perc=min_trace_perc,
                                  t_skip=t_skip)
    results_read_begin = analyse_traces(traces=read_traces,
                                        sample_rate=sample_rate,
                                        filter='low',
                                        min_trace_perc=min_trace_perc,
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


def analyse_NMR(pulse_traces, threshold_up_proportion, sample_rate, t_skip=0,
                shots_per_read=1, min_trace_perc=0.5, t_read=None,
                threshold_voltage=None, threshold_method='config'):
    # Find names of all read segments unless they are read_long
    # (for dark counts)
    read_segment_names = [key for key in pulse_traces
                          if key != 'read_long' and 'read' in key]
    # Number of distinct reads in a trace (i.e. with different ESR frequency)
    distinct_reads_per_trace = len(read_segment_names) // shots_per_read

    results = {}

    # Get shape of single read segment (samples * points_per_segment
    single_read_segment = pulse_traces[read_segment_names[0]]['output']
    samples, read_pts = single_read_segment.shape

    if t_read is not None:
        read_pts = round(t_read * 1e-3 * sample_rate)

    # Create 4D array of all read segments
    read_traces = np.zeros((distinct_reads_per_trace, # Distinct ESR frequencies
                            shots_per_read, # Repetitions of each ESR frequency
                            samples, # Samples (= max_flips + 1)
                            read_pts # sampling points within segment
                            ))
    for k, read_segment_name in enumerate(read_segment_names):
        shot_traces = pulse_traces[read_segment_name]['output'][:, :read_pts]
        # For each shot, all frequencies are looped over.
        # Therefore read_idx is inner loop, and shot_idx outer loop
        read_idx = k % distinct_reads_per_trace
        shot_idx = k // distinct_reads_per_trace
        read_traces[read_idx, shot_idx] = shot_traces

    # Find threshold voltage if not provided
    if threshold_voltage is None:
        high_low = find_high_low(np.ravel(read_traces),
                                 threshold_method=threshold_method)
        threshold_voltage = high_low['threshold_voltage']

    # Populate the up proportions
    for read_idx in range(distinct_reads_per_trace):

        up_proportions = np.zeros(samples)
        for sample in range(samples):
            sample_traces = read_traces[read_idx, :, sample]
            results_read = analyse_traces(sample_traces,
                                          sample_rate=sample_rate,
                                          min_trace_perc=min_trace_perc,
                                          t_read=t_read,
                                          t_skip=t_skip,
                                          threshold_voltage=threshold_voltage)
            up_proportions[sample] = results_read['up_proportion']

        # Determine number of flips
        has_high_contrast = up_proportions > threshold_up_proportion

        flips = sum(abs(np.diff(has_high_contrast)))
        if distinct_reads_per_trace == 1:
            results['up_proportions'] = up_proportions
            results['flips'] = flips
            results['flip_probability'] = flips / (samples - 1)
        else:
            results[f'up_proportions_{read_idx}'] = up_proportions
            results[f'flips_{read_idx}'] = flips
            results[f'flip_probability_{read_idx}'] = flips / (samples - 1)

    return results
