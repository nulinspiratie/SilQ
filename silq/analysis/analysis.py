import numpy as np
import peakutils

from qcodes import Instrument
import qcodes.instrument.parameter as parameter
from qcodes.utils import validators as vals

def find_high_low(traces, plot=False, threshold_peak=0.02, attempts=8):
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
            # print('No peaks found')
            return {'low': None, 'high': None, 'threshold_voltage': None,
                    'voltage_difference': None}

    # Find threshold, mean low, and mean high voltages
    threshold_idx = int(round(np.mean(peaks_idx)))
    threshold_voltage = bin_edges[threshold_idx]

    # Create dictionaries containing information about the low, high state
    low, high = {}, {}
    low['traces'] = traces[traces < threshold_voltage]
    high['traces'] = traces[traces > threshold_voltage]
    for signal in [low, high]:
        signal['mean'] = np.mean(signal['traces'])
        signal['std'] = np.std(signal['traces'])
    voltage_difference = (high['mean'] - low['mean'])
    SNR = voltage_difference / np.sqrt(high['std'] ** 2 + low['std'] ** 2)
    if SNR < 2:
        # print('Signal to noise ratio {} is too low'.format(SNR))
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
            'voltage_difference': voltage_difference}

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

def analyse_load(traces, filter_empty=True):
    threshold_voltage = find_high_low(traces)['threshold_voltage']

    if threshold_voltage is None:
        # print('Could not find two peaks for empty and load state')
        return {'up_proportion': 0, 'num_traces': 0}

    if filter_empty:
        # Filter data that starts at high conductance (no electron)
        idx_begin_empty = edge_voltage(traces, edge='begin', state='high',
                                       threshold_voltage=threshold_voltage)
        traces = traces[idx_begin_empty]


    if not len(traces):
        # print('None of the load traces start with an empty state')
        return {'up_proportion': 0, 'num_traces': 0}

    idx_end_load = edge_voltage(traces, edge='end', state='low',
                                threshold_voltage=threshold_voltage)

    return {'up_proportion': sum(idx_end_load) / len(traces),
            'num_traces': len(traces)}


def analyse_empty(traces, filter_loaded=True):
    threshold_voltage = find_high_low(traces)['threshold_voltage']

    if threshold_voltage is None:
        return {'up_proportion': 0, 'num_traces': 0}

    if filter_loaded:
        # Filter data that starts at high conductance (no electron)
        idx_begin_load = edge_voltage(traces, edge='begin', state='low',
                                      threshold_voltage=threshold_voltage)
        traces = traces[idx_begin_load]

    if not len(traces):
        # print('None of the empty traces start with a loaded state')
        return {'up_proportion': 0, 'num_traces': 0}

    idx_end_empty = edge_voltage(traces, edge='end', state='high',
                                 threshold_voltage=threshold_voltage)

    return {'up_proportion': sum(idx_end_empty) / len(traces),
            'num_traces': len(traces)}


def analyse_EPR(pulse_traces, sample_rate, t_skip=0, t_read=20,
                min_trace_perc=0.5):
    start_idx = round(t_skip * 1e-3 * sample_rate)
    read_pts = round(t_read * 1e-3 * sample_rate)

    results_empty = analyse_empty(pulse_traces['empty']['output'])
    fidelity_empty = results_empty['up_proportion']

    results_load = analyse_load(pulse_traces['plunge'])
    fidelity_load = results_load['up_proportion']


    read_high_low = find_high_low(pulse_traces['read']['output'])
    threshold_voltage = read_high_low['threshold_voltage']
    voltage_difference = read_high_low['voltage_difference']

    if threshold_voltage is None:
        return {'contrast': 0,
                'dark_counts': 0,
                'voltage_difference': 0,
                'fidelity_empty': fidelity_empty,
                'fidelity_load': fidelity_load}
    else:
        read_segment1 = pulse_traces['read']['output'][:, :read_pts]
        read_segment2 = pulse_traces['read']['output'][:, -read_pts:]

        results1 = analyse_read(read_segment1, start_idx=start_idx,
                                threshold_voltage=threshold_voltage,
                                filter_loaded=True)
        up_proportion = results1['up_proportion']
        dark_counts = analyse_read(read_segment2, start_idx=start_idx,
                                   threshold_voltage=threshold_voltage,
                                   filter_loaded=False)['up_proportion']

        if sum(results1['idx']) < min_trace_perc:
            # Not enough traces start loaded
            contrast = 0
        else:
            contrast = up_proportion - dark_counts

    return {'contrast': contrast,
            'dark_counts': dark_counts,
            'voltage_difference': voltage_difference,
            'fidelity_empty': fidelity_empty,
            'fidelity_load': fidelity_load}


def analyse_PR(pulse_traces, sample_rate, t_skip=0, t_read=20,
               min_trace_perc=0.5):
    start_idx = round(t_skip * 1e-3 * sample_rate)
    read_pts = round(t_read * 1e-3 * sample_rate)

    read_high_low = find_high_low(pulse_traces['read']['output'])
    threshold_voltage = read_high_low['threshold_voltage']
    voltage_difference = read_high_low['voltage_difference']

    if threshold_voltage is None:
        return {'contrast': 0,
                'dark_counts': 0,
                'voltage_difference': 0}
    else:
        read_segment1 = pulse_traces['read']['output'][:, :read_pts]
        read_segment2 = pulse_traces['read']['output'][:, -read_pts:]

        results1 = analyse_read(read_segment1, start_idx=start_idx,
                                threshold_voltage=threshold_voltage,
                                filter_loaded=True)
        up_proportion = results1['up_proportion']

        results2 = analyse_read(read_segment2, start_idx=start_idx,
                                threshold_voltage=threshold_voltage,
                                filter_loaded=False)
        dark_counts = results2['up_proportion']

        if sum(results1['idx']) < min_trace_perc:
            # Not enough traces start loaded
            contrast = 0
        else:
            contrast = up_proportion - dark_counts

    return {'contrast': contrast,
            'dark_counts': dark_counts,
            'voltage_difference': voltage_difference}
