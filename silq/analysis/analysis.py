import numpy as np
import peakutils
from matplotlib import pyplot as plt

from qcodes import Instrument
import qcodes.instrument.parameter as parameter
from qcodes.utils import validators as vals

def find_high_low(traces, plot=False, threshold_peak=0.02):
    hist, bin_edges = np.histogram(np.ravel(traces), bins=30)

    # Find two peaks
    for k in range(4):
        peaks_idx = np.sort(peakutils.indexes(hist, thres=threshold_peak, min_dist=5))
        if len(peaks_idx) == 2:
            break
        elif len(peaks_idx) == 1:
            # print('One peak found instead of two, lowering threshold')
            threshold_peak /= 1.5
        elif len(peaks_idx) > 2:
            print('Found {} peaks instead of two, increasing threshold'.format(len(peaks_idx)))
            threshold_peak *= 1.5
        else:
            print('No peaks found')
            return None, None, None

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
    SNR = (high['mean'] - low['mean']) / np.sqrt(high['std'] ** 2 + low['std'] ** 2)
    if SNR < 3:
        'Signal to noise ratio {} is too low'.format(SNR)
        threshold_voltage = None

    # Plotting
    if plot:
        plt.figure()
        for k, signal in enumerate([low, high]):
            sub_hist, sub_bin_edges = np.histogram(np.ravel(signal['traces']), bins=10)
            plt.bar(sub_bin_edges[:-1], sub_hist, width=0.05, color='bg'[k])
            plt.plot(signal['mean'], hist[peaks_idx[k]], 'or', ms=12)

    return low, high, threshold_voltage

def edge_voltage(traces, edge, state, threshold_voltage=None, points=6,
                 start_point=0, plot=False):
    assert edge in ['begin', 'end'], 'Edge {} must be either "begin" or "end"'.format(edge)
    assert state in ['low', 'high'], 'State {} must be either "low" or "high"'.format(state)

    if edge == 'begin':
        if start_point > 0:
            idx_list = slice(start_point, start_point + points)
        else:
            idx_list = slice(None, points)
    else:
        idx_list = slice(-points, None)

    # Determine threshold voltage if not provided
    if threshold_voltage is None:
        low, high, threshold_voltage = find_high_low(traces, plot=plot)

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
                       start_point=0, filter_window=0,
                       plot=False):
    # trace has to contain read stage only
    # TODO Change start point to start time (sampling rate independent)
    if threshold_voltage is None:
        _, _, threshold_voltage = find_high_low(traces, plot=plot)

    if threshold_voltage is None:
        traces_up_electron = [False] * len(traces)
    else:
        if filter_window > 0:
            traces = [np.convolve(trace, np.ones(filter_window) / filter_window, mode='valid') for trace in traces]

        # Filter out the traces that contain one or more peaks
        traces_up_electron = [np.any(trace[start_point:] > threshold_voltage) for trace in traces]

    if return_mean:
        return sum(traces_up_electron) / len(traces)
    else:
        return traces_up_electron

def analyse_read(traces, start_idx=0, threshold_voltage=None, plot=False,
                 return_fidelity=True):
    if threshold_voltage is None:
        low, high, threshold_voltage = find_high_low(traces, plot=plot)

    if threshold_voltage is None:
        # print('Could not find two peaks for empty and load state')
        # Return the full trace length as mean if return_mean=True
        return 0, 0, traces.shape[1] if return_fidelity else []

    # Filter out the traces that start off loaded
    idx_begin_loaded = edge_voltage(traces, edge='begin', state='low',
                                    start_point=start_idx,
                                    threshold_voltage=threshold_voltage)
    traces_loaded = traces[idx_begin_loaded]
    num_traces_loaded = sum(idx_begin_loaded)

    if not any(idx_begin_loaded):
        print('None of the load traces start with a loaded state')
        return (float('nan'), num_traces_loaded,
               traces.shape[1] if return_fidelity else [])

    up_proportion = find_up_proportion(traces_loaded,
                                       start_point=start_idx,
                                       threshold_voltage=threshold_voltage)

    # Filter out the traces that at some point have conductance
    # Assume that if there is current, the electron must have been up
    final_conductance_idx_list = [(k,max(np.where(trace > threshold_voltage)[0]))
                                  for k, trace in enumerate(traces_loaded)
                                  if np.max(trace) > threshold_voltage]
    if return_fidelity:
        return (up_proportion, num_traces_loaded,
                1 - np.mean(final_conductance_idx_list)/ traces.shape[1])
    else:
        return (up_proportion, num_traces_loaded, final_conductance_idx_list)

def analyse_load(traces, plot=False, return_idx=False):
    idx_list = np.arange(len(traces))
    low, high, threshold_voltage = find_high_low(traces, plot=plot)

    if threshold_voltage is None:
        # print('Could not find two peaks for empty and load state')
        if return_idx:
            return 0, []
        else:
            return 0

    # Filter data that starts at high conductance (no electron)
    idx_begin_empty = edge_voltage(traces, edge='begin', state='high',
                                        threshold_voltage=threshold_voltage)
    traces_begin_empty = traces[idx_begin_empty]
    idx_list = idx_list[idx_begin_empty]

    if not len(idx_begin_empty):
        print('None of the load traces start with an empty state')
        return 0 if not return_idx else 0, []

    idx_end_load = edge_voltage(traces_begin_empty, edge='end', state='low',
                                     threshold_voltage=threshold_voltage)

    idx_list = idx_list[idx_end_load] if len(idx_end_load) else []

    if return_idx:
        return sum(idx_end_load) / sum(idx_begin_empty), idx_list
    else:
        return sum(idx_end_load) / sum(idx_begin_empty)

def analyse_empty(traces, plot=False, return_idx=False):
    idx_list = np.arange(len(traces))
    low, high, threshold_voltage = find_high_low(traces, plot=plot)

    if threshold_voltage is None:
        # print('Could not find two peaks for empty and load state')
        if return_idx:
            return 0, []
        else:
            return 0

    # Filter data that starts at high conductance (no electron)
    idx_begin_load = edge_voltage(traces, edge='begin', state='low',
                                  threshold_voltage=threshold_voltage)
    traces_begin_load = traces[idx_begin_load]
    idx_list = idx_list[idx_begin_load]

    if not len(idx_begin_load):
        print('None of the empty traces start with a loaded state')
        return 0

    idx_end_empty = edge_voltage(traces_begin_load, edge='end', state='high',
                                      threshold_voltage=threshold_voltage)

    idx_list = idx_list[idx_end_empty] if len(idx_end_empty) else []

    if return_idx:
        return sum(idx_end_empty) / sum(idx_begin_load), idx_list
    else:
        return sum(idx_end_empty) / sum(idx_begin_load)

def analyse_ELR(trace_segments, sample_rate, t_skip=0, t_read=20, plot=False):
    start_idx = round(t_skip * 1e-3 * sample_rate)
    read_pts = round(t_read * 1e-3 * sample_rate)

    fidelity_empty = analyse_empty(trace_segments['empty'], plot=plot)
    fidelity_load = analyse_load(trace_segments['load'], plot=plot)

    _,_,threshold_voltage = find_high_low(trace_segments['read'], plot=plot)

    if threshold_voltage is None:
        return (fidelity_empty, fidelity_load, 0, 0, 0, 0)
    else:
        read_segment1 = trace_segments['read'][:,:read_pts]
        read_segment2 = trace_segments['read'][:,-read_pts:]

        up_proportion, _, fidelity_read = \
            analyse_read(read_segment1, start_idx=start_idx,
                         threshold_voltage=threshold_voltage)
        dark_counts, _, _ = analyse_read(read_segment2,
                                         start_idx=start_idx,
                                         threshold_voltage=threshold_voltage)
        contrast = up_proportion - dark_counts

    return (fidelity_empty, fidelity_load, fidelity_read,
            up_proportion, dark_counts, contrast)

def analyse_ELRLR(trace_segments, start_idx=0, plot=False):
    fidelity_empty = analyse_empty(trace_segments['empty'], plot=plot)
    fidelity_load = analyse_load(trace_segments['load1'], plot=plot)

    _,_,threshold_voltage = find_high_low(trace_segments['read1'], plot=plot)

    if threshold_voltage is None:
        up_proportion, fidelity_read, dark_counts, difference_up_dark = [0, 0, 0, 0]
    else:
        up_proportion, _, fidelity_read = \
            analyse_read(trace_segments['read1'], start_idx=start_idx,
                         threshold_voltage=threshold_voltage)
        dark_counts, _, _ = analyse_read(trace_segments['read2'],
                                         start_idx=start_idx,
                                         threshold_voltage=threshold_voltage)
        difference_up_dark = up_proportion - dark_counts

    return (fidelity_empty, fidelity_load, fidelity_read,
            up_proportion, dark_counts, difference_up_dark)