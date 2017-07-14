import numpy as np
import peakutils
import logging

__all__ = ['find_high_low', 'edge_voltage', 'find_up_proportion',
           'count_blips', 'analyse_load', 'analyse_empty', 'analyse_read',
           'analyse_read_long', 'analyse_EPR', 'analyse_multi_read_EPR',
           'analyse_PR']

logger = logging.getLogger(__name__)

from silq import config
if 'analysis' not in config:
    config['analysis'] = {}
analysis_config = config['analysis']

def find_high_low(traces, plot=False, threshold_peak=0.02, attempts=8,
                  threshold_method='mean', min_SNR=None):
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

    low_blip_duration = np.array(low_blip_pts) / sample_rate * 1e3
    high_blip_duration = np.array(high_blip_pts) / sample_rate * 1e3

    return {'low_blip_duration': low_blip_duration,
            'low_blip_duration': high_blip_duration,
            'mean_low_blip_duration': np.mean(low_blip_duration),
            'mean_high_blip_duration': np.mean(high_blip_duration)}


def analyse_load(traces, filter_empty=True):
    high_low = find_high_low(traces)
    threshold_voltage = high_low['threshold_voltage']

    if threshold_voltage is None:
        # print('Could not find two peaks for empty and load state')
        return {'up_proportion': 0,
                'num_traces': 0,
                'voltage_difference': 0}

    if filter_empty:
        # Filter data that starts at high conductance (no electron)
        idx_begin_empty = edge_voltage(traces, edge='begin', state='high',
                                       threshold_voltage=threshold_voltage)
        traces = traces[idx_begin_empty]

    if not len(traces):
        # print('None of the load traces start with an empty state')
        return {'up_proportion': 0,
                'num_traces': 0,
                'voltage_difference': high_low['voltage_difference']}

    idx_end_load = edge_voltage(traces, edge='end', state='low',
                                threshold_voltage=threshold_voltage)

    return {'up_proportion': sum(idx_end_load) / len(traces),
            'num_traces': len(traces),
            'voltage_difference': high_low['voltage_difference']}


def analyse_empty(traces, filter_loaded=True):
    high_low = find_high_low(traces)
    threshold_voltage = high_low['threshold_voltage']

    if threshold_voltage is None:
        return {'up_proportion': 0,
                'num_traces': 0,
                'voltage_difference': 0}

    if filter_loaded:
        # Filter data that starts at high conductance (no electron)
        idx_begin_load = edge_voltage(traces, edge='begin', state='low',
                                      threshold_voltage=threshold_voltage)
        traces = traces[idx_begin_load]

    if not len(traces):
        # print('None of the empty traces start with a loaded state')
        return {'up_proportion': 0,
                'num_traces': 0,
                'voltage_difference': high_low['voltage_difference']}

    idx_end_empty = edge_voltage(traces, edge='end', state='high',
                                 threshold_voltage=threshold_voltage)

    return {'up_proportion': sum(idx_end_empty) / len(traces),
            'num_traces': len(traces),
            'voltage_difference': high_low['voltage_difference']}


def analyse_read(traces, start_idx=0, threshold_voltage=None,
                 filter_loaded=True):
    if threshold_voltage is None:
        threshold_voltage = find_high_low(traces)['threshold_voltage']

    if threshold_voltage is None:
        # print('Could not find two peaks for empty and load state')
        # Return the full trace length as mean if return_mean=True
        return {'up_proportion': 0, 'num_traces': 0,
                'idx': np.ones(len(traces), dtype=bool)}

    if filter_loaded:
        # Filter out the traces that start off loaded
        idx = edge_voltage(traces, edge='begin', state='low',
                                        start_point=start_idx,
                                        threshold_voltage=threshold_voltage)
        traces = traces[idx]
    else:
        idx = np.ones(len(traces), dtype=bool)

    if not len(traces):
        # print('None of the load traces start with a loaded state')
        return {'up_proportion': 0, 'num_traces': 0,
                'idx': idx}

    up_proportion = find_up_proportion(traces,
                                       start_point=start_idx,
                                       threshold_voltage=threshold_voltage)

    return {'up_proportion': up_proportion,
            'num_traces': len(traces),
            'idx': idx}


def analyse_read_long(t_read, sample_rate, traces=None,
                      read_segment_begin=None, read_segment_end=None,
                      min_trace_perc=0.5,
                      t_skip=0, threshold_method='config'):
    if traces is None and read_segment_begin is None \
            and read_segment_end is None:
        raise SyntaxError('Must provide traces (optionally read_segments), '
                          'or both read_segment_begin and read_segment_end')

    read_pts = round(t_read * 1e-3 * sample_rate)
    start_idx = round(t_skip * 1e-3 * sample_rate)

    high_low = find_high_low(traces, threshold_method=threshold_method)
    threshold_voltage = high_low['threshold_voltage']
    logger.debug(f'Read threshold voltage: {threshold_voltage}')

    if threshold_voltage is None:
        # Could not find threshold voltage, either too high SNR or no blips
        return {'up_proportion': 0,
                'contrast': 0,
                'dark_counts': 0,
                'threshold_voltage': 0,
                'voltage_difference': 0,
                'DC_voltage': 0,
                'low_blip_duration': 0,
                'high_blip_duration': 0}
    else:
        if read_segment_begin is None:
            # read_segment_begin is the start segment of read_long trace
            read_segment_begin = traces[:, :read_pts]
        if read_segment_end is None:
            # read_segment_end is the end segment of read_long trace
            read_segment_end = traces[:, -read_pts:]

        results_begin = analyse_read(read_segment_begin, start_idx=start_idx,
                                     threshold_voltage=threshold_voltage,
                                     filter_loaded=True)
        up_proportion = results_begin['up_proportion']
        dark_counts = analyse_read(read_segment_end, start_idx=start_idx,
                                   threshold_voltage=threshold_voltage,
                                   filter_loaded=False)['up_proportion']

        if results_begin['num_traces'] / len(traces) < min_trace_perc:
            # Not enough traces start loaded
            contrast = 0
        else:
            contrast = up_proportion - dark_counts

        blips = count_blips(traces, threshold_voltage=threshold_voltage,
                            sample_rate=sample_rate, t_skip=t_skip)

    return {'up_proportion': up_proportion,
            'contrast': contrast,
            'dark_counts': dark_counts,
            'threshold_voltage': threshold_voltage,
            'voltage_difference': high_low['voltage_difference'],
            'DC_voltage': high_low['DC_voltage'],
            'low_blip_duration': blips['mean_low_blip_duration'],
            'high_blip_duration': blips['mean_high_blip_duration']}


def analyse_EPR(pulse_traces, sample_rate, t_read, t_skip,
                min_trace_perc=0.5):
    # Analyse empty stage
    results_empty = analyse_empty(pulse_traces['empty']['output'])

    # Analyse plunge stage
    results_load = analyse_load(pulse_traces['plunge']['output'])

    # Analyse read stage
    results_read = analyse_read_long(traces=pulse_traces['read_long']['output'],
                                     t_read=t_read,
                                     sample_rate=sample_rate,
                                     t_skip=t_skip,
                                     min_trace_perc=min_trace_perc)

    return {'fidelity_empty': results_empty['up_proportion'],
            'voltage_difference_empty': results_empty['voltage_difference'],
            'fidelity_load': results_load['up_proportion'],
            'voltage_difference_load': results_load['voltage_difference'],
            'up_proportion': results_read['up_proportion'],
            'contrast': results_read['contrast'],
            'dark_counts': results_read['dark_counts'],
            'voltage_difference_read': results_read['voltage_difference'],
            'low_blip_duration': results_read['low_blip_duration'],
            'high_blip_duration': results_read['high_blip_duration']}


def analyse_multi_read_EPR(pulse_traces, sample_rate, t_read, t_skip,
                           min_trace_perc=0.5, read_segment_names=None):
    # Analyse empty stage
    results_empty = analyse_empty(pulse_traces['empty']['output'])

    # Analyse plunge stage, there are mulitple plunges in the PulseSequence
    plunge_trace = [val for key, val in pulse_traces.items()
                    if 'plunge' in key][0]
    results_load = analyse_load(plunge_trace['output'])

    # Analyse read traces
    # Analyse first read (corresponding to ESR pulse). The dark counts from
    # read_long must be subtracted to get the contrast. That's why we
    # override segment1
    if read_segment_names is None:
        # Get all read traces (except final read_long)
        read_segment_names = [key for key in pulse_traces
                              if key != 'read_long' and 'read' in key]

    results_read = {}
    for read_segment_name in read_segment_names:
        results_read[read_segment_name] = analyse_read_long(
            traces=pulse_traces['read_long']['output'],
            read_segment_begin=pulse_traces[read_segment_name]['output'],
            t_read=t_read, sample_rate=sample_rate, t_skip=t_skip,
            min_trace_perc=min_trace_perc)

    # Analyse read long (which belongs to the EPR part of pulse sequence)
    results_read_EPR = analyse_read_long(
        traces=pulse_traces['read_long']['output'],
        t_read=t_read,
        sample_rate=sample_rate,
        t_skip=t_skip,
        min_trace_perc=min_trace_perc)

    results =  {
        'fidelity_empty': results_empty['up_proportion'],
        'voltage_difference_empty': results_empty['voltage_difference'],
        'fidelity_load': results_load['up_proportion'],
        'voltage_difference_load': results_load['voltage_difference'],
        'contrast': results_read_EPR['contrast'],
        'dark_counts': results_read_EPR['dark_counts'],
        'voltage_difference_read': results_read_EPR['voltage_difference'],
        'low_blip_duration': results_read_EPR['low_blip_duration'],
        'high_blip_duration': results_read_EPR['high_blip_duration']}

    # Add read results
    if len(read_segment_names) == 1:
        segment_results = results_read[read_segment_names[0]]
        results[f'contrast_read'] = segment_results['contrast']
        results[f'up_proportion_read'] = segment_results['up_proportion']
    else:
        for read_segment_name in read_segment_names:
            segment_results = results_read[read_segment_name]

            read_segment_name = read_segment_name.replace('[', '')
            read_segment_name = read_segment_name.replace(']', '')

            read_contrast = segment_results['contrast']
            results[f'contrast_{read_segment_name}'] = read_contrast
            up_proportion = segment_results['up_proportion']
            results[f'up_proportion_{read_segment_name}'] = up_proportion

    return results

def analyse_PR(pulse_traces, sample_rate, t_skip=0, t_read=20,
               min_trace_perc=0.5):
    # Analyse read stage
    results_read = analyse_read_long(traces=pulse_traces['read_long']['output'],
                                     t_read=t_read,
                                     sample_rate=sample_rate,
                                     t_skip=t_skip,
                                     min_trace_perc=min_trace_perc)

    return {'contrast': results_read['contrast'],
            'dark_counts': results_read['dark_counts'],
            'voltage_difference_read': results_read['voltage_difference'],
            'low_blip_duration': results_read['low_blip_duration'],
            'high_blip_duration': results_read['high_blip_duration']}
