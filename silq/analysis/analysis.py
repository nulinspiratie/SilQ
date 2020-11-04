import numpy as np
import functools
import itertools
from matplotlib.axis import Axis
from matplotlib.colors import TwoSlopeNorm
import peakutils
import logging
from typing import Union, Dict, Any, List, Sequence, Iterable, Tuple
from copy import copy
import collections
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from silq.tools.general_tools import property_ignore_setter

from qcodes import MatPlot
from qcodes.instrument.parameter_node import ParameterNode, parameter
from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators as vals

__all__ = [
    "find_high_low",
    "edge_voltage",
    "find_up_proportion",
    "count_blips",
    "analyse_traces",
    "analyse_EPR",
    "determine_threshold_up_proportion",
    "determine_threshold_up_proportion_single_state",
    "analyse_flips",
    "Analysis",
    "AnalyseEPR",
    "AnalyseElectronReadout",
    "AnalyseMultiStateReadout",
]

logger = logging.getLogger(__name__)

from silq import config

if "analysis" not in config:
    config["analysis"] = {}
analysis_config = config["analysis"]


class Analysis(ParameterNode):
    delegate_attr_dicts = ['parameters', 'parameter_nodes', 'functions',
                           'submodules', 'settings']
    def __init__(self, name):
        super().__init__(name=name, use_as_attributes=True)
        self.settings = ParameterNode(use_as_attributes=True)
        self.outputs = ParameterNode(use_as_attributes=True)
        self.results = {}

        self.enabled = Parameter(set_cmd=None, initial_value=True)

    @property_ignore_setter
    def result_parameters(self):
        parameters = []
        for name, output in self.outputs.parameters.items():
            if not output():
                continue

            output_copy = copy(output)
            parameters.append(output_copy)
        return parameters

    @property_ignore_setter
    def names(self):
        return tuple([parameter.name for parameter in self.result_parameters])

    @property_ignore_setter
    def units(self):
        return tuple([parameter.unit for parameter in self.result_parameters])

    @property_ignore_setter
    def shapes(self):
        return tuple(
            [getattr(parameter, "shape", ()) for parameter in self.result_parameters]
        )

    def analyse(self, **kwargs):
        raise NotImplementedError("Analysis must be implemented in a subclass")


def find_high_low(
    traces: Union[np.ndarray, list, dict],
    plot: bool = False,
    threshold_peak: float = 0.02,
    attempts: int = 8,
    threshold_method: str = "config",
    min_voltage_difference: Union[float, str] = "config",
    threshold_requires_high_low: Union[bool, str] = "config",
    min_SNR: Union[float, None] = None,
    skip_pts=0
):
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
        skip_pts: Optional number of points to skip at the start of each trace

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
            f"Attempts {attempts} to find high and low voltage must be at least 1"
        )

    # Convert traces to list of traces, as traces may contain multiple 2D arrays
    if isinstance(traces, np.ndarray):
        traces = [traces]
    elif isinstance(traces, dict):
        traces = list(traces.values())
        assert isinstance(traces[0], np.ndarray)
    elif isinstance(traces, list):
        pass

    # Optionally remove the first points of each trace
    if skip_pts > 0:
        traces = [trace[:, skip_pts:] for trace in traces]

    # Turn list of 2D traces into a single 2D array
    traces = np.ravel(traces)

    # Retrieve properties from config.analysis
    analysis_config = config.get("analysis", {})
    if threshold_method == "config":
        threshold_method = analysis_config.get("threshold_method", "mean")
    if min_voltage_difference == "config":
        min_voltage_difference = analysis_config.get("min_voltage_difference", 0.3)
    if threshold_requires_high_low == "config":
        threshold_requires_high_low = analysis_config.get(
            "threshold_requires_high_low", True
        )
    if min_SNR is None:
        min_SNR = analysis_config.get("min_SNR", None)

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
        peaks_idx = np.sort(
            peakutils.indexes(hist, thres=threshold_peak, min_dist=min_dist)
        )
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
            "low": None,
            "high": None,
            "threshold_voltage": np.nan,
            "voltage_difference": np.nan,
            "DC_voltage": DC_voltage,
        }

        if not threshold_requires_high_low and threshold_method != "mean":
            # Still return threshold voltage even though no two peaks were observed
            low_or_high, equation = threshold_method.split(':')
            assert low_or_high in ['low', 'high']

            voltages = sorted(traces.flatten())
            if low_or_high == 'low':
                # Remove top 20 percent (high voltage)
                cutoff_slice = slice(None, int(0.8 * len(voltages)))
                voltages_cutoff = voltages[cutoff_slice]
                mean = results['low'] = np.mean(voltages_cutoff)
            else:
                # Remove bottom 20 percent of voltages (low voltage)
                cutoff_slice = slice(int(0.8 * len(voltages)), None)
                voltages_cutoff = voltages[cutoff_slice]
                mean = results['high'] = np.mean(voltages_cutoff)
            # Mean and std are used when evaluating the equation
            std = results['std'] = np.std(voltages_cutoff)

            threshold_voltage = eval(equation)
            results["threshold_voltage"] = threshold_voltage

        return results

    # Find mean voltage, used to determine which points are low/high
    # Note that this is slightly odd, since we might use another threshold_method
    # later on to distinguish between high and low voltage
    mean_voltage_idx = int(np.round(np.mean(peaks_idx)))
    mean_voltage = bin_edges[mean_voltage_idx]

    # Create dictionaries containing information about the low, high state
    low, high = {}, {}
    low["traces"] = traces[traces < mean_voltage]
    high["traces"] = traces[traces > mean_voltage]
    for signal in [low, high]:
        signal["mean"] = np.mean(signal["traces"])
        signal["std"] = np.std(signal["traces"])
    voltage_difference = high["mean"] - low["mean"]

    if threshold_method == "mean":
        # Threshold_method is midway between low and high mean
        threshold_voltage = (high["mean"] + low["mean"]) / 2
    elif ':' in threshold_method:
        low_or_high, equation = threshold_method.split(':')
        assert low_or_high in ['low', 'high']
        signal = {'low': low, 'high': high}[low_or_high]
        mean = signal["mean"]
        std = signal["std"]
        threshold_voltage = eval(equation)
    else:
        raise RuntimeError(f"Threshold method {threshold_method} not understood")

    SNR = voltage_difference / np.sqrt(high["std"] ** 2 + low["std"] ** 2)

    if min_SNR is not None and SNR < min_SNR:
        logger.info(f"Signal to noise ratio {SNR} is too low")
        threshold_voltage = np.nan

    # Plotting
    if plot is not False:
        if plot is True:
            plt.figure()
        else:
            plt.sca(plot)
        for k, signal in enumerate([low, high]):
            sub_hist, sub_bin_edges = np.histogram(np.ravel(signal["traces"]), bins=10)
            plt.bar(sub_bin_edges[:-1], sub_hist, width=0.05, color="bg"[k])
            if k < len(peaks_idx):
                plt.plot(signal["mean"], hist[peaks_idx[k]], "or", ms=12)

    return {
        "low": low,
        "high": high,
        "threshold_voltage": threshold_voltage,
        "voltage_difference": voltage_difference,
        "SNR": SNR,
        "DC_voltage": DC_voltage,
    }


def edge_voltage(
    traces: np.ndarray,
    edge: str,
    state: str,
    threshold_voltage: Union[float, None] = None,
    points: int = 5,
    start_idx: int = 0,
) -> np.ndarray:
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
    assert edge in ["begin", "end"], f"Edge {edge} must be `begin` or `end`"
    assert state in ["low", "high"], f"State {state} must be `low` or `high`"

    if edge == "begin":
        if start_idx > 0:
            idx_list = slice(start_idx, start_idx + points)
        else:
            idx_list = slice(None, points)
    else:
        idx_list = slice(-points, None)

    # Determine threshold voltage if not provided
    if threshold_voltage is None or np.isnan(threshold_voltage):
        threshold_voltage = find_high_low(traces)["threshold_voltage"]

    if threshold_voltage is None or np.isnan(threshold_voltage):
        # print('Could not find two peaks for empty and load state')
        success = np.array([False] * len(traces))
    elif state == "low":
        success = [np.mean(trace[idx_list]) < threshold_voltage for trace in traces]
    else:
        success = [np.mean(trace[idx_list]) > threshold_voltage for trace in traces]
    return np.array(success)


def find_up_proportion(
    traces: np.ndarray,
    threshold_voltage: Union[float, None] = None,
    return_array: bool = False,
    start_idx: int = 0,
    filter_window: int = 0,
) -> Union[float, np.ndarray]:
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
    if threshold_voltage is None or np.isnan(threshold_voltage):
        threshold_voltage = find_high_low(traces)["threshold_voltage"]

    if threshold_voltage is None or np.isnan(threshold_voltage):
        return 0

    if filter_window > 0:
        traces = [
            np.convolve(trace, np.ones(filter_window) / filter_window, mode="valid")
            for trace in traces
        ]

    # Filter out the traces that contain one or more peaks
    traces_up_electron = np.array(
        [np.any(trace[start_idx:] > threshold_voltage) for trace in traces]
    )

    if not return_array:
        return sum(traces_up_electron) / len(traces)
    else:
        return traces_up_electron


def count_blips(
    traces: np.ndarray,
    threshold_voltage: float,
    sample_rate: float,
    t_skip: float,
    ignore_final: bool = False,
):
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
    start_idx = int(round(t_skip * sample_rate))

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
                if not ignore_final:
                    next_idx = len(trace) - idx
                    blip_list.append(next_idx)
                    blip_events[k].append(
                        (int(trace[idx] >= threshold_voltage), next_idx)
                    )
                break
            else:
                blip_list.append(next_idx)
                blip_events[k].append((int(trace[idx] >= threshold_voltage), next_idx))
                idx += next_idx

    low_blip_durations = np.array(low_blip_pts) / sample_rate
    high_blip_durations = np.array(high_blip_pts) / sample_rate

    mean_low_blip_duration = (
        np.NaN if not len(low_blip_durations) else np.mean(low_blip_durations)
    )
    mean_high_blip_duration = (
        np.NaN if not len(high_blip_durations) else np.mean(high_blip_durations)
    )

    blips = len(low_blip_durations) / len(traces)

    duration = len(traces[0]) / sample_rate

    return {
        "blips": blips,
        "blip_events": blip_events,
        "blips_per_second": blips / duration,
        "low_blip_durations": low_blip_durations,
        "high_blip_durations": high_blip_durations,
        "mean_low_blip_duration": mean_low_blip_duration,
        "mean_high_blip_duration": mean_high_blip_duration,
    }


def analyse_traces(
    traces: np.ndarray,
    sample_rate: float,
    filtered_shots: np.ndarray = None,
    filter: Union[str, None] = None,
    min_filter_proportion: float = 0.5,
    t_skip: float = 0,
    t_read: Union[float, None] = None,
    t_read_vals: Union[int, None, Sequence] = None,
    segment: str = "begin",
    threshold_voltage: Union[float, None] = None,
    threshold_method: str = "config",
    plot: Union[bool, Axis] = False,
    plot_high_low: Union[bool, Axis] = False,
):
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
        t_read_vals: Optional range of t_read values for which to extract
            up proportion. Can be:
            - an int, indicating that t_read should be uniformly chosen across
              the trace duration.
            - a list of t_read values
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
        * **t_read_vals** (list(float)): t_read list if provided as kwarg.
          If t_read_vals was an int, this is converted to a list.
          Not returned if t_read_vals is not set.
        * **up_proportions** (list(float)): up_proportion values for each t_read
          if t_read_vals is provided. Not returned if t_read_vals is not set.

    Note:
        If no threshold voltage is provided, and no two peaks can be discerned,
            all results except average_voltage are set to an initial value
            (either 0 or undefined)
        If the filtered trace proportion is less than min_filter_proportion,
            the results ``up_proportion``, ``end_low``, ``end_high`` are set to an
            initial value
    """
    assert filter in [None, "low", "high"], "filter must be None, `low`, or `high`"

    assert segment in ["begin", "end"], "segment must be either `begin` or `end`"

    # Initialize all results to None
    results = {
        "up_proportion": 0,
        "up_proportion_idxs": np.nan * np.zeros(len(traces)),
        "end_high": 0,
        "end_low": 0,
        "num_traces": 0,
        "filtered_traces_idx": None,
        "voltage_difference": np.nan,
        "average_voltage": np.mean(traces),
        "threshold_voltage": np.nan,
        "blips": None,
        "mean_low_blip_duration": None,
        "mean_high_blip_duration": None,
    }

    # minimum trace idx to include (to discard initial capacitor spike)
    start_idx = int(round(t_skip * sample_rate))

    # Calculate threshold voltage if not provided
    if threshold_voltage is None or np.isnan(threshold_voltage):
        # Histogram trace voltages to find two peaks corresponding to high and low
        high_low_results = find_high_low(
            traces[:, start_idx:], threshold_method=threshold_method, plot=plot_high_low
        )
        results["high_low_results"] = high_low_results
        results["voltage_difference"] = high_low_results["voltage_difference"]
        # Use threshold voltage from high_low_results
        threshold_voltage = high_low_results["threshold_voltage"]

        results["threshold_voltage"] = threshold_voltage

    else:
        # We don't know voltage difference since we skip a high_low measure.
        results["voltage_difference"] = np.nan
        results["threshold_voltage"] = threshold_voltage

    if plot is not False:  # Create plot for traces
        ax = MatPlot()[0] if plot is True else plot
        t_list = np.linspace(0, len(traces[0]) / sample_rate, len(traces[0])) * 1e3

        # A
        if threshold_voltage:
            divnorm = TwoSlopeNorm(vmin=np.min(traces), vcenter=threshold_voltage, vmax=np.max(traces))
        else:
            divnorm = None

        ax.add(traces, x=t_list, y=np.arange(len(traces), dtype=float), cmap="seismic", norm=divnorm)
        # Modify x-limits to add blips information
        xlim = ax.get_xlim()
        xpadding = 0.05 * (xlim[1] - xlim[0])
        if segment == "begin":
            xpadding_range = [-xpadding + xlim[0], xlim[0]]
            ax.set_xlim(-xpadding + xlim[0], xlim[1])
        else:
            xpadding_range = [xlim[1], xlim[1] + xpadding]
            ax.set_xlim(xlim[0], xlim[1] + xpadding)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Sample')

    if threshold_voltage is None or np.isnan(threshold_voltage):
        logger.debug("Could not determine threshold voltage")
        if plot is not False:
            ax.text(
                np.mean(xlim),
                len(traces) + 0.5,
                "Unknown threshold voltage",
                horizontalalignment="center",
            )
        return results

    # Analyse blips (disabled because it's very slow)
    # blips_results = count_blips(traces=traces,
    #                             sample_rate=sample_rate,
    #                             threshold_voltage=threshold_voltage,
    #                             t_skip=t_skip)
    # results['blips'] = blips_results['blips']
    # results['mean_low_blip_duration'] = blips_results['mean_low_blip_duration']
    # results['mean_high_blip_duration'] = blips_results['mean_high_blip_duration']

    if filter == "low":  # Filter all traces that do not start with low voltage
        filtered_traces_idx = edge_voltage(
            traces,
            edge="begin",
            state="low",
            start_idx=start_idx,
            threshold_voltage=threshold_voltage,
        )
    elif filter == "high":  # Filter all traces that do not start with high voltage
        filtered_traces_idx = edge_voltage(
            traces,
            edge="begin",
            state="high",
            start_idx=start_idx,
            threshold_voltage=threshold_voltage,
        )
    else:  # Do not filter traces
        filtered_traces_idx = np.ones(len(traces), dtype=bool)

    if filtered_shots is not None:
        filtered_traces_idx = filtered_traces_idx & filtered_shots

    results["filtered_traces_idx"] = filtered_traces_idx
    filtered_traces = traces[filtered_traces_idx]
    results["num_traces"] = len(filtered_traces)

    if len(filtered_traces) / len(traces) < min_filter_proportion:
        logger.debug(f"Not enough traces start {filter}")

        if plot is not False:
            ax.pcolormesh(
                xpadding_range,
                np.arange(len(traces) + 1) - 0.5,
                filtered_traces.reshape(1, -1),
                cmap="RdYlGn",
            )
            ax.text(
                np.mean(xlim),
                len(traces) + 0.5,
                f"filtered traces: {len(filtered_traces)} / {len(traces)} = "
                f"{len(filtered_traces) / len(traces):.2f} < {min_filter_proportion}",
                horizontalalignment="center",
            )
        return results

    # Determine all the t_read's for which to determine up proportion
    total_duration = filtered_traces.shape[1] / sample_rate
    if t_read is None:  # Only use a time segment of each trace
        t_read = total_duration

    if isinstance(t_read_vals, int):
        # Choose equidistantly spaced t_read values
        t_read_vals = np.linspace(total_duration/t_read_vals, total_duration, num=t_read_vals)
    elif t_read_vals is None:
        t_read_vals = []
    elif not isinstance(t_read_vals, Sequence):
        raise ValueError('t_read_vals must be an int, Sequence, or None')

    # Determine up_proportion for each t_read
    up_proportions = []
    for k, t_read_val in enumerate(list(t_read_vals) + [t_read]):

        read_pts = int(round(t_read_val * sample_rate))
        if segment == "begin":
            segmented_filtered_traces = filtered_traces[:, :read_pts]
        else:
            segmented_filtered_traces = filtered_traces[:, -read_pts:]

        # Calculate up proportion of traces
        up_proportion_idxs = find_up_proportion(
            segmented_filtered_traces,
            start_idx=start_idx,
            threshold_voltage=threshold_voltage,
            return_array=True,
        )
        up_proportion = sum(up_proportion_idxs) / len(traces)
        if k == len(t_read_vals):
            results["up_proportion"] = up_proportion
            results["up_proportion_idxs"][filtered_traces_idx] = up_proportion_idxs
        else:
            up_proportions.append(up_proportion)

    if t_read_vals is not None:
        results['up_proportions'] = up_proportions
        results['t_read_vals'] = t_read_vals

    # Calculate ratio of traces that end up with low voltage
    idx_end_low = edge_voltage(
        segmented_filtered_traces,
        edge="end",
        state="low",
        threshold_voltage=threshold_voltage,
    )
    results["end_low"] = np.sum(idx_end_low) / len(segmented_filtered_traces)

    # Calculate ratio of traces that end up with high voltage
    idx_end_high = edge_voltage(
        segmented_filtered_traces,
        edge="end",
        state="high",
        threshold_voltage=threshold_voltage,
    )
    results["end_high"] = np.sum(idx_end_high) / len(segmented_filtered_traces)

    if plot is not False:
        # Plot information on up proportion
        up_proportion_arr = 2 * up_proportion_idxs - 1
        up_proportion_arr[~filtered_traces_idx] = 0
        up_proportion_arr = up_proportion_arr.reshape(-1, 1)  # Make array 2D

        mesh = ax.pcolormesh(
            xpadding_range,
            np.arange(len(traces) + 1) - 0.5,
            up_proportion_arr,
            cmap="RdYlGn",
        )
        mesh.set_clim(-1, 1)

        # Add vertical line for t_read
        if t_read is not None:
            ax.vlines(t_read*1e3, -0.5, len(traces + 0.5), lw=2, linestyle="--", color="orange")
            ax.text(
                t_read*1e3,
                len(traces) + 0.5,
                f"t_read={t_read*1e3} ms",
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            ax.text(
                t_skip*1e3,
                len(traces) + 0.5,
                f"t_skip={t_skip*1e6:.0f} us",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

    return results


def analyse_EPR(
    empty_traces: np.ndarray,
    plunge_traces: np.ndarray,
    read_traces: np.ndarray,
    sample_rate: float,
    t_skip: float,
    t_read: float,
    min_filter_proportion: float = 0.5,
    threshold_voltage: Union[float, None] = None,
    threshold_method='config',
    filter_traces=True,
    plot: bool = False,
):
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
        plot[0].set_title("Empty")
        plot[1].set_title("Plunge")
        plot[2].set_title("Read long")
    elif plot is False:
        plot = [False] * 3

    # Analyse empty stage
    results_empty = analyse_traces(
        traces=empty_traces,
        sample_rate=sample_rate,
        filter="low" if filter_traces else None,
        min_filter_proportion=min_filter_proportion,
        threshold_voltage=threshold_voltage,
        threshold_method=threshold_method,
        t_skip=t_skip,
        plot=plot[0],
    )

    # Analyse plunge stage
    results_load = analyse_traces(
        traces=plunge_traces,
        sample_rate=sample_rate,
        filter="high" if filter_traces else None,
        min_filter_proportion=min_filter_proportion,
        threshold_voltage=threshold_voltage,
        threshold_method=threshold_method,
        t_skip=t_skip,
        plot=plot[1],
    )

    # Analyse read stage
    results_read = analyse_traces(
        traces=read_traces,
        sample_rate=sample_rate,
        filter="low" if filter_traces else None,
        min_filter_proportion=min_filter_proportion,
        threshold_voltage=threshold_voltage,
        threshold_method=threshold_method,
        t_skip=t_skip,
    )
    results_read_begin = analyse_traces(
        traces=read_traces,
        sample_rate=sample_rate,
        filter="low" if filter_traces else None,
        min_filter_proportion=min_filter_proportion,
        threshold_voltage=threshold_voltage,
        threshold_method=threshold_method,
        t_read=t_read,
        segment="begin",
        t_skip=t_skip,
        plot=plot[2],
    )
    results_read_end = analyse_traces(
        traces=read_traces,
        sample_rate=sample_rate,
        t_read=t_read,
        threshold_voltage=threshold_voltage,
        threshold_method=threshold_method,
        segment="end",
        t_skip=t_skip,
        plot=plot[2],
    )

    return {
        "fidelity_empty": results_empty["end_high"],
        "voltage_difference_empty": results_empty["voltage_difference"],
        "fidelity_load": results_load["end_low"],
        "voltage_difference_load": results_load["voltage_difference"],
        "up_proportion": results_read_begin["up_proportion"],
        "contrast": (
            results_read_begin["up_proportion"] - results_read_end["up_proportion"]
        ),
        "dark_counts": results_read_end["up_proportion"],
        "voltage_difference_read": results_read["voltage_difference"],
        "voltage_average_read": results_read["average_voltage"],
        "num_traces": results_read["num_traces"],
        "filtered_traces_idx": results_read["filtered_traces_idx"],
        "blips": results_read["blips"],
        "mean_low_blip_duration": results_read["mean_low_blip_duration"],
        "mean_high_blip_duration": results_read["mean_high_blip_duration"],
        "threshold_voltage": results_read["threshold_voltage"]

    }


class AnalyseEPR(Analysis):
    def __init__(self, name):
        super().__init__(name=name)
        self.settings.sample_rate = Parameter(set_cmd=None)
        self.settings.t_skip = Parameter(
            initial_value=4e-5,
            set_cmd=None,
            config_link="properties.t_skip",
            update_from_config=True,
        )
        self.settings.t_read = Parameter(
            initial_value=None,
            set_cmd=None,
            config_link="properties.t_read",
            update_from_config=True,
        )
        self.settings.min_filter_proportion = Parameter(
            initial_value=0.5,
            set_cmd=None,
            config_link="analysis.min_filter_proportion",
            update_from_config=True,
        )
        self.settings.threshold_voltage = Parameter(
            initial_value=None,
            set_cmd=None,
            config_link="analysis.threshold_voltage",
            update_from_config=True,
        )
        self.settings.threshold_method = Parameter(
            initial_value=None,
            set_cmd=None,
            config_link="analysis.threshold_method",
            update_from_config=True,
        )
        self.settings.filter_traces = Parameter(
            initial_value=True,
            set_cmd=None,
            config_link="analysis.filter_traces",
            update_from_config=True,
        )

        self.outputs.fidelity_empty = Parameter(initial_value=False, set_cmd=None)
        self.outputs.voltage_difference_empty = Parameter(
            initial_value=False, unit="V", set_cmd=None
        )
        self.outputs.fidelity_load = Parameter(initial_value=False, set_cmd=None)
        self.outputs.voltage_difference_load = Parameter(
            initial_value=False, unit="V", set_cmd=None
        )
        self.outputs.up_proportion = Parameter(initial_value=True, set_cmd=None)
        self.outputs.contrast = Parameter(initial_value=True, set_cmd=None)
        self.outputs.dark_counts = Parameter(initial_value=True, set_cmd=None)
        self.outputs.voltage_difference_read = Parameter(
            initial_value=True, unit="V", set_cmd=None
        )
        self.outputs.voltage_average_read = Parameter(
            initial_value=False, unit="V", set_cmd=None
        )
        self.outputs.num_traces = Parameter(initial_value=False, set_cmd=None)
        self.outputs.blips = Parameter(initial_value=False, set_cmd=None)
        self.outputs.mean_low_blip_duration = Parameter(
            initial_value=False, unit="s", set_cmd=None
        )
        self.outputs.mean_high_blip_duration = Parameter(
            initial_value=False, unit="s", set_cmd=None
        )

    @functools.wraps(analyse_EPR)
    def analyse(self, **kwargs):
        settings = self.settings.to_dict(get_latest=False)
        settings.update(**kwargs)
        self.results = analyse_EPR(**settings)
        return self.results


def analyse_electron_readout(
    traces: dict,
    sample_rate: float,
    filtered_shots: np.ndarray = None,
    shots_per_frequency: int = 1,
    labels: List[str] = None,
    t_skip: float = 0,
    t_read: Union[float, None] = None,
    t_read_vals: Union[int, None, Sequence] = None,
    threshold_voltage: Union[float, None] = None,
    threshold_method: str = "config",
    min_filter_proportion: float = 0.5,
    dark_counts: float = None,
    plot: Union[bool, Axis] = False,
    plot_high_low: Union[bool, Axis] = False,
    threshold_up_proportion: float = None
):
    if len(traces) % shots_per_frequency:
        raise RuntimeError(
            "Number of electron readout traces is not a multiple of shots_per_frequency"
        )
    num_frequencies = int(len(traces) / shots_per_frequency)

    results = {}
    read_results = [[] for _ in range(num_frequencies)]

    # Extract samples and points per shot
    first_trace = next(iter(traces.values()))
    if isinstance(first_trace, dict):
        # All trace values are dictinoaries for each channel, select output channel
        traces = {key: trace_arr["output"] for key, trace_arr in traces.items()}
        first_trace = first_trace["output"]
    samples, points_per_shot = first_trace.shape

    # Calculate threshold voltages from combined read traces
    results['high_low'] = high_low = find_high_low(
        traces=traces,
        plot=plot_high_low,
        threshold_method=threshold_method,
        skip_pts = round(sample_rate * t_skip)
    )
    if threshold_voltage is None or np.isnan(threshold_voltage):
        threshold_voltage = high_low["threshold_voltage"]
    results["threshold_voltage"] = threshold_voltage
    results["voltage_difference"] = high_low["voltage_difference"]

    if shots_per_frequency == 1:
        read_traces = [[traces_arr] for traces_arr in traces.values()]
    else:
        # Shuffle traces if shots_per_frequency > 1
        read_traces = [
            [np.zeros((shots_per_frequency, points_per_shot)) for _ in range(samples)]
            for _ in range(num_frequencies)
        ]
        for k, read_traces_arr in enumerate(traces.values()):
            for sample_idx, read_trace in enumerate(read_traces_arr):
                f_idx = k % num_frequencies
                shot_idx = k // num_frequencies

                read_traces[f_idx][sample_idx][shot_idx] = read_trace

    # Iterate through traces and extract data
    if shots_per_frequency == 1:
        up_proportions = np.zeros((num_frequencies, shots_per_frequency))
        up_proportions_idxs = np.nan * np.zeros((num_frequencies, shots_per_frequency, samples))
    else:
        up_proportions = np.zeros((num_frequencies, samples))
        up_proportions_idxs = np.nan * np.zeros((num_frequencies, samples, shots_per_frequency))

    num_traces = np.zeros((num_frequencies, samples))
    for f_idx, read_traces_single_frequency in enumerate(read_traces):
        for sample_idx, read_traces_arr in enumerate(read_traces_single_frequency):
            # read_traces_arr is 2D (shots_per_frequency, points_per_shot)
            read_result = analyse_traces(
                traces=read_traces_arr,
                sample_rate=sample_rate,
                filtered_shots=filtered_shots,
                t_read=t_read,
                t_skip=t_skip,
                t_read_vals=t_read_vals,
                threshold_voltage=threshold_voltage,
                min_filter_proportion=min_filter_proportion,
                plot=plot,
            )
            read_results[f_idx].append(read_result)

            up_proportions[f_idx, sample_idx] = read_result["up_proportion"]
            up_proportions_idxs[f_idx, sample_idx] = read_result["up_proportion_idxs"]
            num_traces[f_idx, sample_idx] = read_result["num_traces"]

    # Add all up proportions as measurement results
    for k, (up_proportion_arr, up_proportion_idxs_arr) in enumerate(zip(up_proportions, up_proportions_idxs)):
        # Determine the right up_proportion label
        label = "up_proportion" if shots_per_frequency == 1 else "up_proportions"
        if labels is not None:
            suffix = f"_{labels[k]}"
        elif num_frequencies > 1:
            suffix = str(k)  # Note that we avoid an underscore
        else:
            suffix = ""

        if shots_per_frequency == 1:  # Remove outer dimension
            up_proportion_arr = up_proportion_arr[0]
            up_proportion_idxs_arr = up_proportion_idxs_arr[0]

        results[f"{label}{suffix}"] = up_proportion_arr
        results[f"{label}_idxs{suffix}"] = up_proportion_idxs_arr

        if threshold_up_proportion:
            results["filtered_shots" + suffix] = up_proportion_arr > threshold_up_proportion

        if dark_counts is not None:
            results["contrast" + suffix] = up_proportion_arr - dark_counts

        results["num_traces" + suffix] = sum(num_traces[k])

    results["read_results"] = read_results

    return results


class AnalyseElectronReadout(Analysis):
    def __init__(self, name):
        super().__init__(name=name)
        self.settings.sample_rate = Parameter(set_cmd=None)
        self.settings.num_frequencies = Parameter(initial_value=1, set_cmd=None,)
        self.settings.samples = Parameter(
            initial_value=1,
            set_cmd=None,
            docstring=(
                "Number of samples (pulse sequence repetitions). Only relevant "
                "when shots_per_frequency > 1, in which case the shape of each "
                "up_proportions is equal to the number of samples"
            ),  # TODO get rid of this parameter once the new Measurement is used
        )
        self.settings.shots_per_frequency = Parameter(initial_value=1, set_cmd=None,)
        self.settings.labels = Parameter(
            initial_value=None, set_cmd=None, vals=vals.Lists(allow_none=True)
        )
        self.settings.t_skip = Parameter(
            initial_value=4e-5,
            set_cmd=None,
            config_link="properties.t_skip",
            update_from_config=True,
        )
        self.settings.t_read = Parameter(
            initial_value=None,
            set_cmd=None,
            config_link="properties.t_read",
            update_from_config=True,
        )
        self.settings.t_read_vals = Parameter(
            initial_value=None,
            set_cmd=None,
        )
        self.settings.threshold_voltage = Parameter(
            initial_value=None,
            set_cmd=None,
            config_link="analysis.threshold_voltage",
            update_from_config=True,
        )
        self.settings.threshold_method = Parameter(
            initial_value="mean",
            set_cmd=None,
            config_link="analysis.threshold_method",
            update_from_config=True,
        )
        self.settings.min_filter_proportion = Parameter(
            initial_value=0.5,
            set_cmd=None,
            config_link="analysis.min_filter_proportion",
            update_from_config=True,
        )
        self.settings.threshold_up_proportion = Parameter(
            initial_value=None,
            set_cmd=None,
            docstring="Optional up proportion threshold, used to generate " \
                      "filtered_shots"
        )

        self.outputs.read_results = Parameter(initial_value=False, set_cmd=False)
        self.outputs.up_proportion = Parameter(initial_value=True, set_cmd=None)
        self.outputs.contrast = Parameter(initial_value=False, set_cmd=None)
        self.outputs.num_traces = Parameter(initial_value=True, set_cmd=None)
        self.outputs.voltage_difference = Parameter(
            initial_value=True, unit="V", set_cmd=None
        )
        self.outputs.threshold_voltage = Parameter(
            initial_value=True, unit="V", set_cmd=None
        )
        self.outputs.filtered_shots = Parameter(
            initial_value=False, set_cmd=None
        )
        self.outputs.up_proportion_idxs = Parameter(
            initial_value=False, set_cmd=None
        )

    @property
    def result_parameters(self):
        parameters = []

        if self.settings.labels is not None:
            suffixes = [f"_{label}" for label in self.settings.labels]
        elif self.settings.num_frequencies > 1:
            # Note that we avoid an underscore
            suffixes = [str(k) for k in range(self.settings.num_frequencies)]
        else:
            suffixes = [""] * self.settings.num_frequencies

        for name, output in self.outputs.parameters.items():
            if not output():
                continue

            if name in ["up_proportion", "contrast", "num_traces", "filtered_shots"]:
                # Add a label for each frequency
                for k in range(self.settings.num_frequencies):
                    output_copy = copy(output)
                    output_copy.name = name + suffixes[k]

                    if (
                        name == "up_proportion"
                        and self.settings.shots_per_frequency > 1
                    ):
                        output_copy.name = "up_proportions" + suffixes[k]
                        output_copy.shape = (self.settings.samples,)

                    parameters.append(output_copy)
            elif name == 'up_proportion_idxs':
                # Add a label for each frequency
                for k in range(self.settings.num_frequencies):
                    output_copy = copy(output)

                    if self.settings.shots_per_frequency > 1:
                        output_copy.name = "up_proportions_idxs" + suffixes[k]
                        output_copy.shape = (self.settings.samples, self.settings.shots_per_frequency)
                    else:
                        output_copy.name = "up_proportion_idxs" + suffixes[k]
                        output_copy.shape = (self.settings.samples, )

                    parameters.append(output_copy)
            else:
                output_copy = copy(output)
                parameters.append(output_copy)
        return parameters

    @functools.wraps(analyse_electron_readout)
    def analyse(self, **kwargs):
        settings = self.settings.to_dict(get_latest=False)
        settings.update(**kwargs)
        settings.pop("num_frequencies", None)  # Not needed
        settings.pop("samples", None)  # Not needed
        self.results = analyse_electron_readout(**settings)
        return self.results


def determine_threshold_up_proportion_single_state(up_proportions_arr: np.ndarray,
                                                   shots_per_frequency: int):
    """ Determine threshold up-proportion for single nuclear state readout using
        single up proportion array of samples

    Using kernel density estimation, 'determine_threshold_up_proportion_single_state'
    function determines the probability density distribution of measured electron spin-up
    proportions within the 0 to 1 up proportion space in the form of a Gaussian function.
    It is later analysed for the peaks in the density distribution and based on their
    relative position within 0 to 1 up proportion space, the density minimum is found,
    which is considered to be the threshold for determining the nuclear spin state.

    WARNING:
        If no peaks in the density distribution are detected (something failed), threshold
        is set to 0.5 by default.
        If only one peak is detected (e.g. nuclear spin did not flip or transition detuned),
        depending on whether it is in the lower (<0.5) or the upper (>=0.5) half of
        proportion space, the minimum of density distribution is found respectively above or
        below this peak.
        If more than one peak is detected, then minimum is found between the lowest and the
        highest peak.
        Since the density distribution values reach 1e-20 at the edges of the proportion space
        (i.e. at 0 or 1), to smooth a bit the threshold jumps between the measurement points,
        rounding to the 4th digit is implemented.

    NOTE: algorithm generally might fail if the number of shots is less than 25 (not exact number,
    can still be adjusted). Therefore, if less than 25 shots are given, function still uses 25 shots.
    This is due to the division of up proportion space by the number of shots and if there are too
    few points to perform kernel density estimation, it fails.

    Args:
        up_proportions_arr: 1D Array of up proportions, calculated from
            `analyse_traces`. Up proportion is the proportion of traces that
            contain blips. The length is number of samples.
        shots_per_frequency: Integer number of ESR shots/traces, which are
            used to determine the up proportion.

    Returns:
        (float) threshold up-proportion
    """
    assert isinstance(up_proportions_arr, np.ndarray), 'Up proportions must be a 1D array, not a list.'

    num_shots = max(shots_per_frequency, 25)
    if shots_per_frequency < 25:
        logger.warning(f'Readout shots {shots_per_frequency} < 25. This may cause issues when finding '
                       f'an optimal threshold up proportion, thus 25 shots are used instead.')
    # Algorithm fails if the number of shots is less than 25 (not exact number, can still be adjusted).
    # The reason for this is that in the following division of up_proportion space by the number of shots
    # there are too few points to perform kernel density estimation.

    proportion_space = np.linspace(0, 1, num=num_shots + 1, endpoint=True)
    up_proportions_arr = up_proportions_arr.reshape(1, -1)  # 'gaussian_kde' function requires at least 2D array
    kernel = gaussian_kde(up_proportions_arr)
    gaussian_up_proportions = kernel(proportion_space)
    # 'gaussian_up_proportions' is a 1D array of spin-up proportions' density distribution within
    # up proportion space, i.e. has length of proportion_space.

    samples = up_proportions_arr.shape[-1]  # other dimension is empty

    # Finding indexes of density distribution peaks. Might still require some fine adjustments in
    # 'thres' and 'min_dist' parameters. TODO: find better values for 'thres' and 'min_dist'.
    peak_idxs = peakutils.peak.indexes(gaussian_up_proportions,
                                       thres=0.5/samples,
                                       min_dist=num_shots/5)

    # If no peaks are detected (something failed), by default threshold is set to 0.5.
    middle_point = (num_shots + 1) // 2
    if len(peak_idxs) == 0:
        logger.debug('Adaptive thresholding routine: 0 peaks were found, using threshold 0.5')
        trough_idx = middle_point

    # If only one peak is detected (nuclear spin did not flip), depending on whether
    # it is in the lower (<0.5) or the upper (>=0.5) half of proportion space,
    # the minimum index of density distribution is found respectively above or below
    # this peak's index.
    # Since the density distribution values reach 1e-20 at the edges of the
    # proportion space (i.e. at 0 or 1), to smooth a bit the threshold jumps between
    # the measurement points, rounding to the 4th digit is implemented.
    # TODO: Find better way to determine threshold between density distribution peaks.
    elif len(peak_idxs) == 1:
        if peak_idxs[0] < middle_point:
            trough_slice = slice(peak_idxs[0], num_shots + 1)
            search_space = np.round(gaussian_up_proportions[trough_slice], 4)
            trough_idx = peak_idxs[0] + np.argmin(search_space)
        else:
            trough_slice = slice(0, peak_idxs[0])
            search_space = np.round(gaussian_up_proportions[trough_slice], 4)
            trough_idx = np.argmin(search_space)

    # If more than one peak is detected, then minimum is found between the lowest and the highest peak.
    else:
        trough_slice = slice(min(peak_idxs), max(peak_idxs))
        search_space = np.round(gaussian_up_proportions[trough_slice], 4)
        trough_idx = peak_idxs[0] + np.argmin(search_space)

    threshold_up_proportion = proportion_space[trough_idx]

    return threshold_up_proportion


def determine_threshold_up_proportion(
    up_proportions_arrs: np.ndarray,
    threshold_up_proportion: Union[float, Tuple[float, float]] = None,
    filtered_shots: np.ndarray = None,
) -> dict:
    """Extract the threshold_up_proportion from up proportion arrays

    For each column of up proportions, exactly one should have a high
    up proportion and the rest a low up proportion.

    The threshold is chosen that maximizes the number of columns that have
    exactly one up proportion above threshold

    Args:
        up_proportions_arrs: 2D array of up proportions.
            Each column corresponds to a single readout, and the rows correspond
            to the different possible states
        threshold_up_proportion: Optional preset threshold_up_proportion.
            Can be either a single value, or (threshold_low, threshold_high) pair.
            If provided, will simply return the value.
        filtered_shots: Optional 1D boolean array, where True indicates that the
            corresponding column in up_proportions_arrs should be used in this
            analysis to calculate the threshold.

    Returns:
        Dict containing

        - ``threshold_up_proportion``: up_proportion threshold that maximizes
            the number of up proportion pairs where one up proportion is above
            and one below the threshold.
        - ``threshold_high`` highest threshold value for which the above
            condition holds
        - ``threshold_low`` lowest threshold value for which the above
            condition holds
        - ``filtered_fraction`` fraction of up proportion pairs that have one
            up proportion above threshold_high and one below threshold_low
        - ``N_filtered_shots`` number of up proportion pairs that have one
            up proportion above threshold_high and one below threshold_low
        - ``filtered_shots`` numpy boolean 1D array, an element is True if one
            up proportion is above threshold_high and the rest below threshold_low.
            If filtered_shots is also provided as an arg, the two are combined
            using a logical ``and``.
        - ``all_filtered``: Bool whether all shots have one above threshold and
            the rest below threshold. This is used later to determine
             ``filtered_flip_probability``. Note that if filtered shots is also
             explicitly passed as a kwarg, those False values do not make
             ``all_filtered`` False. In other words, ``all_filtered`` is only
             False if there are shots that do not have exactly one above
             threshold high, the rest below threshold_low, and where the
             corresponding element in the filtered_shots kwarg is not False.

    Raises:
        AssertionError if up_proportion arrays are not 1D

    TODO:
        - Handle when up_proportions contain NaN
        - Think about greater than vs greater than or equal to w.r.t. thresholds
    """
    assert up_proportions_arrs.ndim == 2

    results = {
        "threshold_up_proportion": None,
        "threshold_high": None,
        "threshold_low": None,
        "filtered_fraction": 0,
        "N_filtered_shots": 0,
        "filtered_shots": np.zeros(up_proportions_arrs.shape[1], dtype=bool),
        "all_filtered": False
    }

    unique_up_proportions = sorted({*up_proportions_arrs.ravel()})
    if len(unique_up_proportions) == 1 and threshold_up_proportion is None:
        # No distinct up proportions provided, system is likely out of tune
        return results

    # Calculate minimum and maximum up proportions
    up_max = np.nanmax(up_proportions_arrs, axis=0)
    if np.any(np.isnan(up_proportions_arrs)):
        # Not sure what the best approach is here. up proportions can be NaN if
        # a voltage threshold cannot be determined.
        up_min = np.nanmin(up_proportions_arrs, axis=0)
    else:
        up_min = np.sort(up_proportions_arrs, axis=0)[:,-2]

    if threshold_up_proportion is not None:
        # Threshold already provided
        if isinstance(threshold_up_proportion, float):
            # Single threshold provided => equal low and high thresholds
            results["threshold_low"] = threshold_up_proportion
            results["threshold_high"] = threshold_up_proportion
            results["threshold_up_proportion"] = threshold_up_proportion
        elif isinstance(threshold_up_proportion, Sequence):
            # A separate low and high threshold is provided
            assert len(threshold_up_proportion) == 2

            results["threshold_low"] = threshold_up_proportion[0]
            results["threshold_high"] = threshold_up_proportion[1]
            results["threshold_up_proportion"] = np.mean(threshold_up_proportion)
        else:
            raise RuntimeError("Threshold_up_proportion must be float or float pair")
    elif max(up_min) < min(up_max):
        results["threshold_low"] = max(up_min)
        results["threshold_high"] = min(up_max)
        results["threshold_up_proportion"] = np.mean([min(up_max), max(up_min)])
    else:
        # No threshold can be determined for which one up proportion is above
        # and one below. We find the best threshold.
        # Possible thresholds are averages between successive up proportions
        up_proportion_differences = np.diff(unique_up_proportions)
        up_proportion_inter = unique_up_proportions[:-1] + up_proportion_differences / 2

        # Calculate threshold results for each possible threshold
        threshold_results = [
            determine_threshold_up_proportion(
                up_proportions_arrs=up_proportions_arrs,
                threshold_up_proportion=threshold,
            )
            for threshold in up_proportion_inter
        ]
        # The best threshold is the one with the highest filtered_fraction
        optimal_idx = np.argmax(
            [result["filtered_fraction"] for result in threshold_results]
        )
        optimal_threshold = up_proportion_inter[optimal_idx]
        threshold_difference = up_proportion_differences[optimal_idx]
        results["threshold_up_proportion"] = optimal_threshold
        results["threshold_high"] = optimal_threshold + threshold_difference / 2
        results["threshold_low"] = optimal_threshold - threshold_difference / 2

    # Determine the fraction of up proportion pairs that have one
    # up proportion above threshold_high and one below threshold_low
    # Note 1e-11 for machine precision errors
    above_threshold = np.nansum(up_proportions_arrs >= results["threshold_high"]-1e-11, axis=0)
    below_threshold = np.nansum(up_proportions_arrs <= results["threshold_low"]+1e-11, axis=0)

    # Each column should have one up proportion above threshold, and the
    # remaining should be below threshold
    results["filtered_shots"] = np.logical_and(
        above_threshold == 1, below_threshold == up_proportions_arrs.shape[0] - 1
    )
    if filtered_shots is not None:
        # An explicit filtered_shots is also provided. In this case, all_filtered
        # is False if there are elements that do not satisfy the filter (one
        # up proportion above threshold_high, the rest below threshold low) but
        # additionally the corresponding explicit filtered_shots element must be
        # True
        filtered_shots_copy = results["filtered_shots"].copy()
        filtered_shots_copy[~filtered_shots] = True
        results["all_filtered"] = np.all(filtered_shots_copy)

        results["filtered_shots"] = np.logical_and(
            filtered_shots, results["filtered_shots"]
        )
    else:
        results["all_filtered"] = np.all(results["filtered_shots"])

    results["N_filtered_shots"] = sum(results["filtered_shots"])
    results["filtered_fraction"] = results["N_filtered_shots"] / len(
        results["filtered_shots"]
    )

    return results


def parse_flip_pairs(
        flip_pairs: Union[str, List[Tuple[int, int]]],
        num_states: int = None,
        labels: List[str] = None
) -> Tuple[list, list]:
    """

    Args:
        flip_pairs: Pairs of states between which to compare flips.
            Can either be specific strings, or a list of pair tuples.
            The following strings are allowed:

            - ``all``: Compare flipping between all states
            - ``neighbouring``: Only compare flipping between neighbouring states

            Additionally, tuple pairs of states are allowed.
            The states can either be integers, or strings
        num_states: Total number of states. If not provided, ``labels`` must be
            provided, and num_states will equal ``len(labels)``.
        labels: Labels for each of the states. If not provided, num_states must
            be provided and the state labels are simply the state indices

    Returns:
        flip_pairs: Parsed pairs of states between which to consider flipping
            events. If ``flip_pairs`` is a str, it will be converted to the
            corresponding pairs of states. States are also converted to their
            corresponding labels.
        flip_pairs_indices: Indices of the flip pairs

    """
    if num_states is None and labels is None:
        raise SyntaxError('Either num_states or labels must be provided')
    elif num_states is None:
        num_states = len(labels)

    # Ensure each flip pair contains ints.
    if isinstance(flip_pairs, str):
        # Flip pairs is a string specifying the different combinations
        if flip_pairs == "neighbouring":
            flip_pair_indices = list(zip(range(num_states - 1), range(1, num_states)))
        elif flip_pairs == "all":
            flip_pair_indices = list(itertools.combinations(range(num_states), r=2))
        else:
            raise RuntimeError(f"Flip pairs {flip_pairs} not understood")

        if labels is None:
            flip_pairs = flip_pair_indices
        else:
            flip_pairs = [(labels[k1], labels[k2]) for (k1, k2) in flip_pair_indices]

    elif isinstance(flip_pairs[0][0], str):
        # Flip pairs use state labels, convert to state indices
        assert labels is not None
        flip_pair_indices = [map(labels.index, flip_pair) for flip_pair in flip_pairs]
    else:
        flip_pair_indices = flip_pairs
    # Ensure flip_pairs_int are tuples and sorted
    flip_pair_indices = [tuple(sorted(flip_pair)) for flip_pair in flip_pair_indices]
    return flip_pairs, flip_pair_indices


def analyse_flips(
        states: np.ndarray,
        flip_pairs: Union[str, List[Tuple[int, int]]] = None,
        num_states: int = None,
        all_filtered: bool = None
) -> Dict[str, Dict[Tuple, Union[int, float]]]:
    """Analyse flipping between pairs of states. Used for nuclear spin readout

    Args:
        states: 1D array where each element is the index (int) of a measured state.
            If successive elements are x1 and x2, a flip has occurred between
            x1 and x2.
            When an element is NaN, the state could not be determined and is
            therefore not considered for flipping
        flip_pairs: Optionally provided pairs of states to measure flipping
            events for. If not provided, flipping events are measured between
            all possible pairs of states.
            Note that if num_states is not provided and a state with high index
            is not in the sequence of states, it will not be included in the
            possible pairs of states between which flipping can occur
        num_states: Optionally provided number of states. Only necessary
            if flip_pairs is not provided. See comment in flip_pairs.
        all_filtered: Whether all up_proportion tuples have satisfied the filter
            criteria (see `determine_threshold_up_proportion` for details).
            If set to None (default), the number of possible flips must equal the
            length of the states array minus one.
            If False, filtered_flip_probability will be NaN.


    Returns:
        Dict where each item is a dict with state pair tuples as keys.
        Dict items are:

        - ``possible_flips``: Number of possible flips for each pair of states.
        - ``flips``: Number of flips between each pair of states.
        - ``flip_probability``: flips / possible_flips for each state pair.
            If possible_flips == 0, flip_probability is NaN
        - ``filtered_flip_probability`` Flip probability if all states are valid,
            else NaN. See ``all_filtered`` for details
    """
    results = {
        "possible_flips": {},
        "flips": {},
        "flip_probabilities": {},
        "filtered_flip_probabilities": {}
    }

    if flip_pairs is not None:
        # Ensure flip_pairs are valid
        for flip_pair in flip_pairs:
            assert len(flip_pair) == 2
            assert isinstance(flip_pair[0], int)
            assert isinstance(flip_pair[1], int)
    else:
        if num_states is None:
            if np.all(np.isnan(states)):
                logger.warning("All states in analyse_flips are NaN")
                return results

            # Set the number of states based on the maximum state index
            num_states = np.nanmax(states) + 1

        # Choose all possible pairs of states.
        # Note that each element is a list, not a tuple
        flip_pairs = itertools.combinations(range(num_states), r=2)

    # Sort each flip pair in ascending order
    flip_pairs = [tuple(sorted(flip_pair)) for flip_pair in flip_pairs]

    # Iterate through successive states and record the flips
    possible_flips = {flip_pair: 0 for flip_pair in flip_pairs}
    flips = {flip_pair: 0 for flip_pair in flip_pairs}
    for k, (state1, state2) in enumerate(zip(states[:-1], states[1:])):
        if np.isnan(state1) or np.isnan(state2):
            # State could not be determined or is filtered out for another reason
            continue

        for flip_pair in flip_pairs:
            if state1 not in flip_pair:
                # first state does not belong to the flip pair
                continue

            if state2 != state1 and state2 in flip_pair:
                # Flip occurred between the two states
                flips[flip_pair] += 1

            possible_flips[flip_pair] += 1

    flip_probabilities = {
        flip_pair: flips[flip_pair] / possible_flips[flip_pair]
        if possible_flips[flip_pair]
        else np.nan
        for flip_pair in flip_pairs
    }

    filtered_flip_probabilities = {
        flip_pair: flip_probability if all_filtered else np.nan
        for flip_pair, flip_probability in flip_probabilities.items()
    }

    return {
        "possible_flips": possible_flips,
        "flips": flips,
        "flip_probabilities": flip_probabilities,
        "filtered_flip_probabilities": filtered_flip_probabilities
    }


def analyse_multi_state_readout(
    up_proportions_arrs: np.ndarray,
    threshold_up_proportion: Union[float, Tuple[float, float]] = None,
    filtered_shots: np.ndarray = None,
    labels: Tuple[str, str] = None,
    flip_pairs: List[Tuple[Union[int, str], Union[int, str, None]]] = None,
):
    """

    Args:
        up_proportions_arrs:
        threshold_up_proportion:
        filtered_shots:
        labels:
        flip_pairs:

    Returns:

    TODO:
        determine flip probability if only one array is provided
    """
    results = {}
    num_states = len(up_proportions_arrs)

    # Determine threshold_up_proportion. If already provided, it will simply
    # return the threshold, while otherwise it will determine the best threshold
    threshold_results = determine_threshold_up_proportion(
        up_proportions_arrs=up_proportions_arrs,
        threshold_up_proportion=threshold_up_proportion,
        filtered_shots=filtered_shots,
    )
    results.update(**threshold_results)
    filtered_shots = threshold_results["filtered_shots"]
    N_filtered_shots = threshold_results["N_filtered_shots"]

    if threshold_results["threshold_up_proportion"] is None:
        logger.warning(
            'No threshold up proportion could be determined. This probably means '
            'the system is out of tune (all up proportions are 1 or all are zero)'
        )
        results['state_sequence'] = np.nan * np.ones(up_proportions_arrs.shape[1])
        results['states'] = np.nan * np.ones(up_proportions_arrs.shape[0])
        results['state_probabilities'] = np.nan * np.ones(up_proportions_arrs.shape[0])
        if flip_pairs is not None:
            flip_pairs, flip_pairs_indices = parse_flip_pairs(
                flip_pairs, labels=labels, num_states=num_states
            )
            for label1, label2 in flip_pairs:
                results[f"possible_flips_{label1}_{label2}"] = 0
                results[f"flips_{label1}_{label2}"] = 0
                results[f"flip_probability_{label1}_{label2}"] = np.nan
                results[f"filtered_flip_probability_{label1}_{label2}"] = np.nan

        return results

    if N_filtered_shots == 0:
        results['state_sequence'] = np.nan * np.ones(up_proportions_arrs.shape[1])
        results['states'] = np.nan * np.ones(up_proportions_arrs.shape[0])
        results['state_probabilities'] = np.nan * np.ones(up_proportions_arrs.shape[0])
    else:
        # Determine the state for each column
        # Each index is the corresponding state index for the given column
        state_sequence = np.nanargmax(up_proportions_arrs, axis=0).astype(float)
        # state is NaN if it cannot be uniquely determined, i.e. is filtered out
        state_sequence[~filtered_shots] = np.nan
        results['state_sequence'] = state_sequence
        results['states'] = np.array([sum((state_sequence == k)) for k in range(num_states)])
        results["state_probabilities"] = results['states'] / N_filtered_shots

    # Determine flip probabilities
    if flip_pairs is not None:
        flip_pairs, flip_pairs_indices = parse_flip_pairs(
            flip_pairs, labels=labels, num_states=num_states
        )
        flip_results = analyse_flips(
            states=results["state_sequence"],
            flip_pairs=flip_pairs_indices,
            all_filtered=threshold_results["all_filtered"]
        )

        for (label1, label2), flip_pair_int in zip(flip_pairs, flip_pairs_indices):
            for key, val in flip_results.items():
                key = key.replace("probabilities", "probability")

                results[f"{key}_{label1}_{label2}"] = val[flip_pair_int]

    return results


class AnalyseMultiStateReadout(Analysis):
    def __init__(self, name):
        super().__init__(name=name)
        self.settings.threshold_up_proportion = Parameter(
            initial_value=None,
            set_cmd=None,
            config_link="analysis.threshold_up_proportion",
            update_from_config=True,
        )
        self.settings.num_frequencies = Parameter(
            initial_value=None, set_cmd=None, vals=vals.Ints()
        )
        self.settings.labels = Parameter(
            initial_value=None, set_cmd=None, vals=vals.Lists(allow_none=True)
        )
        self.settings.flip_pairs = Parameter(
            initial_value="neighbouring",
            set_cmd=None,
            vals=vals.MultiType(vals.Enum("neighbouring", "all"), vals.Lists(), allow_none=True),
        )

        self.outputs.threshold_up_proportion = Parameter(
            initial_value=False, set_cmd=None
        )
        self.outputs.threshold_low = Parameter(initial_value=False, set_cmd=None)
        self.outputs.threshold_high = Parameter(initial_value=False, set_cmd=None)

        self.outputs.filtered_fraction = Parameter(initial_value=False, set_cmd=None)
        self.outputs.N_filtered_shots = Parameter(initial_value=False, set_cmd=None)
        self.outputs.filtered_shots = Parameter(initial_value=False, set_cmd=None)
        self.outputs.all_filtered = Parameter(initial_value=True, set_cmd=None)
        self.outputs.state_sequence = Parameter(initial_value=True, set_cmd=None)
        self.outputs.states = Parameter(initial_value=False, set_cmd=None)
        self.outputs.state_probabilities = Parameter(initial_value=False, set_cmd=None)

        self.outputs.flips = Parameter(initial_value=True, set_cmd=None)
        self.outputs.flip_probability = Parameter(initial_value=True, set_cmd=None)
        self.outputs.filtered_flip_probability = Parameter(initial_value=True, set_cmd=None)
        self.outputs.possible_flips = Parameter(initial_value=True, set_cmd=None)

    @property
    def labels(self):
        if self.settings.labels is not None:
            return self.settings.labels
        elif self.settings.num_frequencies is not None:
            return [str(k) for k in range(self.settings.num_frequencies)]
        else:
            return []

    @property
    def flip_pairs(self):
        if self.settings.flip_pairs == "neighbouring":
            flip_pairs = list(zip(self.labels[:-1], self.labels[1:]))
        elif self.settings.flip_pairs == "all":
            flip_pairs = []
            for k, label1 in enumerate(self.labels):
                for label2 in self.labels[k + 1:]:
                    flip_pairs.append((label1, label2))
        else:
            flip_pairs = self.settings.flip_pairs

        return flip_pairs

    @property
    def flip_pairs_str(self):
        if self.flip_pairs is not None:
            return [f"{label1}_{label2}" for (label1, label2) in self.flip_pairs]
        else:
            return []

    @property
    def result_parameters(self):
        parameters = []
        for name, output in self.outputs.parameters.items():
            if not output():
                continue
            elif "flip" in name:
                for label_pair_str in self.flip_pairs_str:
                    output_copy = copy(output)
                    output_copy.name = f"{name}_{label_pair_str}"
                    parameters.append(output_copy)
            else:
                parameters.append(copy(output))
        return parameters

    @functools.wraps(analyse_multi_state_readout)
    def analyse(self, **kwargs):
        settings = self.settings.to_dict(get_latest=False)
        settings.update(**kwargs)
        settings.pop("num_frequencies", None)  # Not needed
        self.results = analyse_multi_state_readout(**settings)
        return self.results


def analyse_flips_old(
    up_proportions_arrs: List[np.ndarray],
    shots_per_frequency: int,
    threshold_up_proportion: Union[Sequence, float] = None,
    labels: List[str] = None,
    label_pairs: List[List[str]] = "neighbouring",
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
        separator = ""  # No separator should be used for combined flips labels
    else:
        separator = "_"  # Use separator to distinguish labels in combined flips

    if not isinstance(up_proportions_arrs, np.ndarray):  # Convert to numpy array
        up_proportions_arrs = np.array(up_proportions_arrs)

    up_proportions_dict = {
        label: arr for label, arr in zip(labels, up_proportions_arrs)
    }

    max_flips = up_proportions_arrs.shape[-1] - 1  # number of samples - 1

    # Determine threshold_up_proportion_low/high
    if (len(up_proportions_arrs) == 1) and (threshold_up_proportion is None):
        threshold_up_proportion = determine_threshold_up_proportion_single_state(
            up_proportions_arr=up_proportions_arrs,
            shots_per_frequency=shots_per_frequency)
        threshold_low = threshold_up_proportion
        threshold_high = threshold_up_proportion
    elif isinstance(threshold_up_proportion, collections.Sequence):
        if len(threshold_up_proportion) != 2:
            raise SyntaxError(
                f"threshold_up_proportion must be either single "
                "value, or two values (low and high threshold)"
            )
        threshold_low = threshold_up_proportion[0]
        threshold_high = threshold_up_proportion[1]
    else:
        threshold_low = threshold_up_proportion
        threshold_high = threshold_up_proportion

    # First calculate flips/flip_probability for individual up_proportions
    # Note that we skip this step if threshold_up_proportion is None
    if threshold_up_proportion is not None:
        # State of up proportions by threshold (above: 1, below: -1, between: 0)
        with np.errstate(
            invalid="ignore"
        ):  # ignore errors (up_proportions may contain NaN)
            state_arrs = np.zeros(up_proportions_arrs.shape)
            state_arrs[up_proportions_arrs > threshold_high] = 1
            state_arrs[up_proportions_arrs < threshold_low] = -1

        for k, state_arr in enumerate(state_arrs):
            flips = np.sum(np.abs(np.diff(state_arr)) == 2, axis=-1, dtype=float)
            # Flip probability by dividing flips by number of samples - 1
            flip_probability = flips / max_flips

            # Add suffix if more than one up_proportion array is provided
            suffix = f"_{labels[k]}" if len(up_proportions_arrs) > 1 else ""
            results["flips" + suffix] = flips
            results["flip_probability" + suffix] = flip_probability

    # Determine combined flips/flip_probability
    if len(up_proportions_arrs) > 1:
        if label_pairs == "neighbouring":
            # Only look at neighbouring combined flips
            label_pairs = list(zip(labels[:-1], labels[1:]))
        elif label_pairs == "full":
            label_pairs = []
            for k, label1 in enumerate(labels):
                for label2 in labels[k + 1:]:
                    label_pairs.append((label1, label2))

        for label1, label2 in label_pairs:
            up_proportions_1 = up_proportions_dict[label1]
            up_proportions_2 = up_proportions_dict[label2]
            suffix = f"_{label1}{separator}{label2}"

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
                    results["combined_flips" + suffix] = np.nan
                    results["combined_flip_probability" + suffix] = np.nan
                    results["filtered_combined_flips" + suffix] = np.nan
                    results["filtered_combined_flip_probability" + suffix] = np.nan
                    results["possible_flips" + suffix] = np.nan
                    continue

            # Determine state that are above/below threshold.
            state_arrs = []
            for k, up_proportions in enumerate([up_proportions_1, up_proportions_2]):
                # Boolean arrs equal to True if up proportion is above/below threshold
                with np.errstate(
                    invalid="ignore"
                ):  # ignore errors if up_proportions contains NaN
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
                np.abs(np.diff(relative_state_arr)) == 4, axis=-1, dtype=float
            )
            results["combined_flips" + suffix] = combined_flips

            # Number of possible flips, i.e. where successive up_proportion pairs
            # both satisfy one being above and one below threshold
            possible_flips = np.sum(
                ~np.isnan(np.diff(relative_state_arr)), axis=-1, dtype=float
            )
            results["possible_flips" + suffix] = possible_flips
            if possible_flips > 0:
                combined_flip_probability = combined_flips / possible_flips
            else:
                combined_flip_probability = np.nan
            results["combined_flip_probability" + suffix] = combined_flip_probability

            # Check if all up_proportion pairs satisfy threshold condition
            if possible_flips == max_flips:
                results["filtered_combined_flips" + suffix] = combined_flips
                results[
                    "filtered_combined_flip_probability" + suffix
                ] = combined_flip_probability
            else:
                results["filtered_combined_flips" + suffix] = np.nan
                results["filtered_combined_flip_probability" + suffix] = np.nan

            results["threshold_up_proportion"] = (threshold_low, threshold_high)

    return results
