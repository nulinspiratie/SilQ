from __future__ import division

import numpy as np
import warnings
from matplotlib import pyplot as plt
from typing import Union
import numpy as np
import logging
from collections import Counter
from copy import copy
import h5py
from statsmodels.base.model import GenericLikelihoodModel
import itertools


from .analysis import find_high_low, count_blips
from .fit_toolbox import DoubleExponentialFit, ExponentialFit
from silq.tools.trace_tools import extract_pulse_slices_from_trace_file

from qcodes import DataSet

logger = logging.getLogger(__name__)


def extract_blip_statistics(
    traces: np.ndarray,
    sample_rate: float,
    include_first: bool = False,
    existing_results: dict = None,
    threshold_voltage: float = None,
    min_SNR: float = 2.5,
    threshold_method: str = "3.5*std_low",
    **kwargs
):
    """Extract blip statistics from SET current traces

    Args:
        traces:
        sample_rate:
        include_first:
        existing_results:
        threshold_voltage:
        min_SNR:
        threshold_method:
        **kwargs:

    Returns:

    """
    first_tunnel_times = []
    tunnel_times = [[], []]
    secondary_blip_traces = 0

    # Ensure traces is 2D
    if traces.ndim == 1:  # Transform single trace into 2D array
        traces = np.expand_dims(traces, axis=0)
    elif traces.ndim > 2:  # Merge all outer dimensions
        traces = np.reshape(traces, (-1, traces.shape[-1]))

    if threshold_voltage is None:  # Determine threshold voltage
        high_low = find_high_low(
            traces, min_SNR=min_SNR, threshold_method=threshold_method
        )
        threshold_voltage = high_low["threshold_voltage"]
        if threshold_voltage is None:
            threshold_voltage = high_low["DC_voltage"] + 0.5

    blip_data = count_blips(
        traces, sample_rate=sample_rate, threshold_voltage=threshold_voltage, **kwargs
    )

    for blip_event in blip_data["blip_events"]:
        if not blip_event:
            continue
        elif blip_event[0][0] != 0:  # Does not start at low voltage
            continue
        else:
            has_secondary_blip = False

            first_tunnel_times.append(blip_event[0][1] / sample_rate)

            if include_first:
                tunnel_times[0].append(blip_event[0][1] / sample_rate)

            for (above_threshold, pts) in blip_event[1:]:
                # either add to tunnel out times (0) or tunnel in times (1)
                tunnel_times[above_threshold].append(pts / sample_rate)

                if not above_threshold:
                    has_secondary_blip = True

            secondary_blip_traces += has_secondary_blip

    results = {
        "blip_data": blip_data,
        "first_tunnel_times": first_tunnel_times,
        "tunnel_out_times": tunnel_times[0],
        "tunnel_in_times": tunnel_times[1],
        "secondary_blip_traces": secondary_blip_traces,
    }

    if existing_results is None:
        return results
    else:
        for key, val in existing_results.items():
            new_result = results[key]
            if not isinstance(val, list):
                existing_results[key] = [val]
            if not isinstance(new_result, list):
                new_result = [new_result]
            existing_results[key] += new_result
        return existing_results


def analyse_tunnel_times(
    tunnel_times_arr,
    bins=50,
    range=None,
    silent=True,
    double_exponential=True,
    plot=False,
    fixed_parameters={},
    initial_parameters={},
    plot_fit_kwargs={},
    **kwargs,
):
    hist, bin_edges = np.histogram(tunnel_times_arr, bins=bins, range=range)
    bin_averages = (bin_edges[:-1] + bin_edges[1:]) / 2
    dt = bin_averages[1] - bin_averages[0]

    if double_exponential:
        fit = DoubleExponentialFit(
            xvals=bin_averages,
            ydata=hist,
            fixed_parameters={"offset": 0, **fixed_parameters},
            initial_parameters=initial_parameters,
        )

        tau_1 = fit.fit_result.best_values["tau_1"]
        tau_2 = fit.fit_result.best_values["tau_2"]
        A_1 = fit.fit_result.best_values["A_1"]
        A_2 = fit.fit_result.best_values["A_2"]

        if tau_1 > tau_2:
            tau_1, tau_2 = tau_2, tau_1
            A_1, A_2 = A_2, A_1

        N_1 = A_1 * tau_1 / dt
        N_2 = A_2 * tau_2 / dt

        if not silent:
            print(
                f"tau_up ={tau_1*1e3:>7.3g} ms\t"
                f"tau_down ={tau_2*1e3:>7.3g} ms\t"
                f"tau_up/tau_down ={tau_2/tau_1:>7.3g}\t"
                f"N_up ={N_1:>7.0f}\t"
                f"N_down ={N_2:>7.0f}\t"
                f"N_down / N_up ={N_1 / N_2:.3g}"
            )

        result = {
            "hist": hist,
            "bin_edges": bin_edges,
            "bin_averages": bin_averages,
            "fit": fit,
            "tau_up": tau_1,
            "tau_down": tau_2,
            "A_up": A_1,
            "A_down": A_2,
            "N_up": N_1,
            "N_down": N_2,
        }
    else:
        fit = ExponentialFit(
            xvals=bin_averages,
            ydata=hist,
            fixed_parameters={"offset": 0, **fixed_parameters},
            initial_parameters=initial_parameters,
        )
        tau = fit.fit_result.best_values["tau"]
        A = fit.fit_result.best_values["amplitude"]
        N = A * tau / dt
        if not silent:
            print(f"tau ={tau*1e3:>7.3g} ms\t" f"N ={N:>7.0f}")

        result = {
            "hist": hist,
            "bin_edges": bin_edges,
            "bin_averages": bin_averages,
            "fit": fit,
            "tau": tau,
            "A": A,
            "N": N,
        }

    if plot:
        ax = plt.subplots()[1] if plot is True else plot

        ax.plot(bin_averages * 1e3, hist, **kwargs)
        fit.add_to_plot(ax, xscale=1e3, **plot_fit_kwargs)

        ax.set_yscale("log")
        ax.set_ylabel("Counts")
        ax.set_xlabel("Tunnel time (ms)")

    return result


def t_read_optimal(tau_up, tau_down):
    """Optimal read duration"""
    return 1/(1/tau_down - 1/tau_up) * np.log(tau_up / tau_down)


def contrast_optimal(t, tau_up, tau_down):
    return np.exp(-t/tau_down) - np.exp(-t/tau_up)


def get_blips(
        traces: Union[h5py.File, np.ndarray],
        threshold_voltage: float,
        sample_rate: float,
        pulse_slice=None,
        silent=False,
):
    """Get first blips from an array"""
    if isinstance(traces, h5py.File):
        pulse_slices = extract_pulse_slices_from_trace_file(
            traces_file=traces, sample_rate=sample_rate
        )
        results = {}
        for pulse_name, pulse_slice in pulse_slices.items():
            result = results[pulse_name] = get_blips(
                traces=traces['traces']['output'],
                threshold_voltage=threshold_voltage,
                sample_rate=sample_rate,
                pulse_slice=pulse_slice,
                silent=True,
            )
            result['pulse_slice'] = pulse_slice

            if not silent:
                print(
                    f"Traces without blips: {result['N_traces_no_blips']}/{result['N_traces']} "
                    f"({result['N_traces_no_blips'] / result['N_traces']*100:.1f}%)"
                )

        if len(pulse_slices) == 1:
            # Only one pulse, return results of first pulse
            return next(iter(results.values()))
        else:
            return results
    else:
        assert isinstance(traces, (h5py.Dataset, np.ndarray))

    if traces.ndim > 2:
        results = {}
        for trace_arr in traces:
            result = get_blips(
                trace_arr,
                threshold_voltage=threshold_voltage,
                sample_rate=sample_rate,
                pulse_slice=pulse_slice,
                silent=True,
            )
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    results.setdefault(key, 0)
                else:
                    results.setdefault(key, [])

                results[key] += val
    else:
        if pulse_slice is not None:
            traces = traces[:, pulse_slice]

        samples, trace_pts = traces.shape

        blip_results = count_blips(
            traces,
            threshold_voltage=threshold_voltage,
            sample_rate=sample_rate,
            t_skip=0,
            ignore_final=True,
        )
        first_blip_durations = [
            elem[0][1] / sample_rate for elem in blip_results['blip_events'] if len(elem)
        ]

        N_traces_blips = len(first_blip_durations)
        N_traces_no_blips = samples - N_traces_blips
        N_traces = N_traces_blips + N_traces_no_blips

        results = {
            'N_traces_blips': N_traces_blips,
            'N_traces_no_blips': N_traces_no_blips,
            'N_traces': N_traces,
            'first_blip_durations': first_blip_durations,
            'low_blip_durations': list(blip_results['low_blip_durations']),
            'high_blip_durations': list(blip_results['high_blip_durations']),
        }

    if not silent:
        print(
            f"Traces without blips: {results['N_traces_no_blips']}/{results['N_traces']} "
            f"({results['N_traces_no_blips'] / results['N_traces']*100:.1f}%)"
        )

    return results


class ExponentialModel(GenericLikelihoodModel):
    parameter_names = ['tau']

    def __init__(self, endog, exog=None, **kwargs):
        if exog is None:
            exog = np.zeros_like(endog)

        self.results = None
        self.initial_parameters = None
        self.fixed_parameters = None
        self.parameters = None

        super(ExponentialModel, self).__init__(endog, exog, **kwargs)

    def find_initial_parameters(self):
        # Choose the 20th percentile for tunnel time
        return dict(tau=np.percentile(self.endog, 20))

    def fit_function(self, t, tau, **kwargs):
        if tau <= 0:
            return 1e-15
        result = np.exp(-t / tau) / tau
        return result

    def nloglikeobs(self, params):
        params = dict(zip(self.parameter_names, params))
        return -np.log(self.fit_function(self.endog, **params))

    def fit(
            self,
            initial_parameters=None,
            fixed_parameters=None,
            maxiter=10000,
            maxfun=5000,
            silent=False,
            **kwargs
    ):
        if initial_parameters is None:
            initial_parameters = self.find_initial_parameters()
        self.initial_parameters = initial_parameters
        self.fixed_parameters = fixed_parameters or {}

        if 'tau' not in self.fixed_parameters:
            self.results = super(ExponentialModel, self).fit(
                start_params=list(self.initial_parameters.values()),
                maxiter=maxiter,
                maxfun=maxfun,
                disp=not silent,
                **kwargs
            )
            self.parameters = dict(zip(self.parameter_names, self.results.params))
        else:
            # Only free parameter `tau` is already provided, no fitting needed
            self.parameters = self.fixed_parameters

        return self.results


class DoubleExponentialModel(GenericLikelihoodModel):
    parameter_names = ['A_1', 'tau_1', 'tau_2']

    def __init__(self, endog, exog=None, **kwargs):
        if exog is None:
            exog = np.zeros_like(endog)

        self.results = None
        self.initial_parameters = None
        self.fixed_parameters = None
        self.parameters = None

        super(DoubleExponentialModel, self).__init__(endog, exog, **kwargs)

    @staticmethod
    def calculate_A_2(A_1, tau_1, tau_2):
        return (1 - A_1 * tau_1) / tau_2

    def find_initial_parameters(self, fixed_parameters={}):
        results = dict(
            tau_1 = fixed_parameters.get('tau_1', 100e-6),
            tau_2 = fixed_parameters.get('tau_2', 1e-3)
        )
        results['A_1'] = 1/results['tau_1']/2
        # A_2 is a dependent parameter
        return results

    def fit_function(self, t, A_1, tau_1, tau_2, **kwargs):
        A_2 = self.calculate_A_2(A_1, tau_1, tau_2)
        if min(A_1, A_2) <= 0 or min(tau_1, tau_2) <= 0:
            return 1e-15
        result = A_1*np.exp(-t/tau_1) + A_2*np.exp(-t/tau_2)
        return result

    def nloglikeobs(self, params):
        params = dict(zip(self.initial_parameters, params))
        params.update(**self.fixed_parameters)
        return -np.log(self.fit_function(self.endog, **params))

    def fit(
            self,
            initial_parameters=None,
            fixed_parameters=None,
            maxiter=10000,
            maxfun=5000,
            silent=False,
            **kwargs
    ):
        self.fixed_parameters = fixed_parameters or {}
        if initial_parameters is None:
            initial_parameters = self.find_initial_parameters(self.fixed_parameters)
        # Remove all parameters that are already in fixed_parameters
        self.initial_parameters = {
            key: val for key, val in initial_parameters.items()
            if key not in self.fixed_parameters
        }

        self.results = super(DoubleExponentialModel, self).fit(
            start_params=list(initial_parameters.values()),
            maxiter=maxiter,
            maxfun=maxfun,
            disp=not silent,
            **kwargs
        )
        self.parameters = dict(zip(self.initial_parameters, self.results.params))
        self.parameters.update(**self.fixed_parameters)

        # # Swap parameters if tau_1 > tau_2
        # if self.parameters['tau_1'] > self.parameters['tau_2']:
        #     params = self.parameters
        #     for key in ['tau', 'A', 'p']:
        #         key_1, key_2 = params[f'{key}_1'], params[f'{key}_2']
        #         params[f'{key}_1'], params[f'{key}_2'] = key_2, key_1

        return self.results


class TunnelTimesAnalysis:
    """

    Args:
        tunnel_times: List
        traces:
        results:
        sample_rate:
        threshold_voltage:
        silent:
        t_skip:
    """
    def __init__(
            self,
            tunnel_times=None,
            traces=None,
            results=None,
            sample_rate=None,
            threshold_voltage=None,
            silent=False,
            t_skip=0,
            ignore_pre_t_skip=False,
            num_exponentials=(2, 1),
            fixed_parameters=None
    ):
        results = self.parse_tunnel_times(
            tunnel_times=tunnel_times, traces=traces, results=results,
            threshold_voltage=threshold_voltage, sample_rate=sample_rate
        )
        self.tunnel_times = np.array(results['tunnel_times'])
        self.N_traces = results['N_traces']
        self.N_traces_blips = results['N_traces_blips']
        self.N_traces_no_blips = results['N_traces_no_blips']

        self.original_tunnel_times = copy(self.tunnel_times)

        self.ignore_pre_t_skip = ignore_pre_t_skip
        self.t_skip = t_skip

        self.num_exponentials = num_exponentials

        self.fixed_parameters = fixed_parameters

        self.fig = None
        self.axes = None

        # Perform fitting
        self.model = None
        self.fit_result = None
        self.results = self.fit(silent=silent)


        if not silent:
            self.print_results()

    @staticmethod
    def parse_tunnel_times(
            tunnel_times=None, traces=None, results=None,
            threshold_voltage=None, sample_rate=None,
            silent=True
    ):
        if tunnel_times is not None:
            return {
                'tunnel_times': tunnel_times,
                'N_traces_blips': len(tunnel_times),
                # Cannot determine how many traces have no blips
                'N_traces_no_blips': None,
                'N_traces': len(tunnel_times)
            }
        elif traces is not None:
            results = get_blips(
                traces,
                threshold_voltage=threshold_voltage,
                sample_rate=sample_rate,
                silent=silent
            )
            return {
                'tunnel_times': results['first_blip_durations'],
                'N_traces_blips': results['N_traces_blips'],
                'N_traces_no_blips': results['N_traces_no_blips'],
                'N_traces': results['N_traces']
            }
        elif results is not None:
            return {
                'tunnel_times': results['first_blip_durations'],
                'N_traces_blips': results['N_traces_blips'],
                'N_traces_no_blips': results['N_traces_no_blips'],
                'N_traces': results['N_traces']
            }
        else:
            raise SyntaxError('Must provide either tunnel_times, traces, or results')

    @property
    def t_skip(self):
        return self._t_skip

    @t_skip.setter
    def t_skip(self, t_skip):
        self._t_skip = t_skip

        if t_skip is not None:
            self.tunnel_times = copy(self.original_tunnel_times)
            self.tunnel_times -= t_skip
            if self.ignore_pre_t_skip:
                self.tunnel_times = np.array([t for t in self.tunnel_times if t > 0])
            else:
                self.tunnel_times[self.tunnel_times < 0] = 0

            optimal_t_skip = self.optimal_t_skip()
            if optimal_t_skip is not None and t_skip < optimal_t_skip:
                logger.warning(
                    f't_skip {t_skip*1e6} us below optimum {optimal_t_skip*1e6:.0f} us'
                )
        else:
            self.tunnel_times = self.original_tunnel_times

    def optimal_t_skip(self):
        min_events = 30  # Minimum number of events at single tunnel time
        threshold_factor = 0.4  # Minimum factor of maximum tunnel time events
        max_tunnel_time = 100e-6

        tunnel_times_count = Counter(sorted(self.original_tunnel_times))
        max_tunnel_times = max(tunnel_times_count.values())

        if max_tunnel_times < min_events:
            logger.debug('Could not determine optimal t_skip, too few data points')
            return None

        for tunnel_time, counts in tunnel_times_count.items():
            if counts < threshold_factor * max_tunnel_times:
                continue
            elif tunnel_time > max_tunnel_time:
                logger.debug(
                    f'Could not determine optimal t_skip, exceeded {max_tunnel_time}'
                )
                return None
            else:
                return tunnel_time
        else:
            logger.debug(f'Could not determine optimal t_skip, unknown reason')
            return None

    def get_ancillary_parameters(self, A_1, tau_1, tau_2):
        A_2 = self.model.calculate_A_2(A_1=A_1, tau_1=tau_1, tau_2=tau_2)

        # Get initial probabilities (does not include traces without blips)
        p_1 = A_1 * tau_1
        p_2 = A_2 * tau_2

        # Calculate number of events with tau_1 or tau_2
        N_1 = int(p_1 * self.N_traces_blips)
        N_2 = int(p_2 * self.N_traces_blips)

        # Modify probabilities and events if some traces have no blips
        if self.N_traces_no_blips is not None:
            N_2 += self.N_traces_no_blips
            p_1 = N_1 / self.N_traces
            p_2 = N_2 / self.N_traces

        t_read_optimum = (tau_1*tau_2) / (tau_1 - tau_2) * np.log(tau_1 / tau_2)
        visibility_optimum = -np.exp(-t_read_optimum/tau_1) + (np.exp(-t_read_optimum / tau_2))
        contrast_optimum = (
            (1 - np.exp(-t_read_optimum / tau_1)) * p_1
            - (1 - np.exp(-t_read_optimum / tau_2)) * p_2
        )

        return dict(
            A_1=A_1, A_2=A_2, tau_1=tau_1, tau_2=tau_2,
            p_1=p_1, p_2=p_2, N_1=N_1, N_2=N_2,
            t_read_optimum=t_read_optimum,
            visibility_optimum=visibility_optimum,
            contrast_optimum=contrast_optimum
        )

    def fit(self, tunnel_times=None, silent=False):
        if tunnel_times is None:
            tunnel_times = self.tunnel_times

        num_exponentials = self.num_exponentials
        if isinstance(num_exponentials, int):
            num_exponentials = num_exponentials,

        models = {2: DoubleExponentialModel, 1: ExponentialModel}

        for num_exponential in num_exponentials:
            self.model = models[num_exponential](tunnel_times)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Inverting hessian failed")
                self.fit_result = self.model.fit(silent=silent, fixed_parameters=self.fixed_parameters)

            results = copy(self.model.parameters)

            if num_exponential == 2:
                # Add ancillary results
                results = self.get_ancillary_parameters(**results)

                tau_vals = (results['tau_1'], results['tau_2'])

                if tau_vals[0] < 1e-8 or (tau_vals[1] - tau_vals[0]) < 1e-6:
                    logger.warning('Could not fit a double exponential, using single exponential')
                    continue

            self.num_exponentials = num_exponential
            break

        return results

    def print_results(self):
        for key, value in self.results.items():
            if 'tau' in key:
                print(f"{key}: {value*1e6:.0f} us")
            else:
                print(f"{key}: {value:.3g}")

    def plot_tunnel_times(self, tunnel_times=None, t_cutoff=None, bins=41,
                          plot_fast=True, fig=None, axes=None, title=''):
        if tunnel_times is None:
            tunnel_times = self.tunnel_times

        if isinstance(self.model, ExponentialModel):
            plot_fast = False

        if plot_fast:
            if t_cutoff is None:
                if self.results is None:
                    t_cutoff = 5e-3
                else:
                    t_cutoff = 15 * self.results['tau_1']

            tunnel_times_fast = np.array([t for t in tunnel_times if t <= t_cutoff])
            tunnel_times_arrays = [tunnel_times_fast, tunnel_times]
            if axes is None:
                fig, axes = plt.subplots(2, 1, figsize=(6, 4))

            titles = [f'{title} - Zoom-in', title]
        else:
            tunnel_times_arrays = [tunnel_times]
            if axes is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
                axes = [ax]

            titles = [title]

        for k, (ax, tunnel_times_arr) in enumerate(zip(axes, tunnel_times_arrays)):
            # Normalize probabilities to tunnel times
            y, x, _ = ax.hist(tunnel_times_arr*1e3, bins=bins)
            x /= 1e3

            dx = np.diff(x)[0]
            x = (x[:-1] + x[1:]) / 2
            probabilities = self.model.fit_function(x, **self.model.parameters) * dx
            probabilities *= len(tunnel_times)
            ax.plot(x*1e3, probabilities, lw=2)
            ax.set_yscale('log')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Counts / bin')
            if isinstance(self.model, ExponentialModel):
                ax.legend(['Exponential fit', 'Measured bins'])
            else:
                ax.legend(['Double exponential fit', 'Measured bins'])

            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_title(titles[k])

        if fig is not None:
            fig.tight_layout()

        self.fig = fig
        self.axes = axes

        return fig, axes


def analyse_tunnel_times_measurement(
    data: Union[DataSet, h5py.File, dict, list],
    threshold_voltage=150e-3,
    t_skip=0,
    sample_rate=200e3,
    detailed=True,
    silent=False,
    first_blips_only=True
):
    if silent:
        detailed = False

    if isinstance(data, DataSet):
        data = data.load_traces()

    ### Extract blips
    if isinstance(data, h5py.File):
        blip_results = get_blips(
            data,
            threshold_voltage=threshold_voltage,
            sample_rate=sample_rate,
            silent=silent
        )
    elif isinstance(data, (dict, list)):
        # Ensure data is a list
        if isinstance(data, list):
            data = {f"readout_{k}": trace_arr for k, trace_arr in enumerate(data)}

        blip_results = {}
        for key, trace_arr in data.items():
            blip_results[key] = get_blips(
                trace_arr,
                threshold_voltage=threshold_voltage,
                sample_rate=sample_rate,
                silent=silent
            )
    else:
        raise SyntaxError('Must provide either data, or traces')

    assert len(blip_results) <= 2, "Cannot handle more than two blip results"

    ### Determine tunnel time analyses
    tunnel_times_analyses = {'tunnel_out': None, 'tunnel_in': None, 'individual': []}

    # Determine combined tunnel in/out times
    for in_out in ['out', 'in']:
        if in_out == 'in':
            key = 'high_blip_durations'
        elif first_blips_only:
            key = 'first_blip_durations'
        else:
            key = 'low_blip_durations'

        tunnel_times = [result[key] for result in blip_results.values()]
        # Flatten list of tunnel times lists
        tunnel_times = list(itertools.chain(*tunnel_times))
        tunnel_times_analysis = TunnelTimesAnalysis(
            tunnel_times=tunnel_times,
            t_skip=t_skip,
            ignore_pre_t_skip=True,
            sample_rate=sample_rate,
            threshold_voltage=threshold_voltage,
            num_exponentials=(2, 1) if in_out == 'out' else (1, ),
            silent=True
        )
        tunnel_times_analyses[f'tunnel_{in_out}'] = tunnel_times_analysis

    ### Analyse individual trace arrays using existing tunnel out times
    tunnel_out_analysis = tunnel_times_analyses['tunnel_out']
    # Tunnel times and number of exponentials are fixed by tunnel_out_analysis
    fixed_parameters = {
        key: val for key, val in tunnel_out_analysis.results.items()
        if key.startswith('tau')
    }
    for k, (key, result) in enumerate(blip_results.items()):
        tunnel_times_analysis = TunnelTimesAnalysis(
            results=result,
            t_skip=t_skip,
            sample_rate=sample_rate,
            threshold_voltage=threshold_voltage,
            num_exponentials=tunnel_out_analysis.num_exponentials,
            fixed_parameters=fixed_parameters,
            silent=True
        )
        tunnel_times_analyses['individual'].append(tunnel_times_analysis)

    if tunnel_out_analysis.num_exponentials == 1:
        tau_up = tau_down = tunnel_out_analysis.results['tau']
    else:
        tau_up = tunnel_out_analysis.results['tau_1']
        tau_down = tunnel_out_analysis.results['tau_2']

    results = {
        'tau_up': tau_up,
        'tau_down': tau_down,
        'tau_ratio': tau_down / tau_up,
        'tau_in': tunnel_times_analyses['tunnel_in'].results['tau'],
        'probability_spin_up': np.nan,
        'probability_spin_down': np.nan,
        'probability_initialize_spin_up': np.nan,
        't_read_optimum': tunnel_out_analysis.results.get('t_read_optimum', np.nan),
        'contrast_optimum': tunnel_out_analysis.results.get('contrast_optimum', np.nan),
        'analyses': tunnel_times_analyses
    }

    # Determine properties dependent on number of trace arrays
    high_analysis = low_analysis = None
    if tunnel_times_analysis.num_exponentials == 2:
        if len(blip_results) == 1:
            # Single array provided, can be either high up proportion or not
            tunnel_times_analysis = tunnel_times_analyses['individual'][0]
            results['probability_spin_up'] = tunnel_times_analysis.results['p_1']
            results['probability_spin_down'] = tunnel_times_analysis.results['p_2']
        elif len(blip_results) == 2:
            # We assume that one trace array has high up proportion and the other
            # is used to determine dark counts
            analyses = tunnel_times_analyses['individual']
            # Find analysis with high up proportion
            high_idx = analyses[1].results['p_1'] > analyses[0].results['p_1']
            high_analysis = analyses[high_idx]
            low_analysis = analyses[(1+high_idx) % 2]

            results['probability_spin_up'] = high_analysis.results['p_1']
            results['probability_spin_down'] = high_analysis.results['p_2']
            results['probability_initialize_spin_up'] = low_analysis.results['p_1']

    # Print results
    if not silent:
        print(
            f"tau_up = {results['tau_up']*1e6:.0f} us\n"
            f"tau_down = {results['tau_down']*1e6:.0f} us\n"
            f"tau_down / tau_up = {tau_down / tau_up:.0f}\n"
            f"tau_in = {results['tau_in']*1e6:.0f} us\n"
            f"P(spin-up) = {results['probability_spin_up']:.2f}\n"
            f"P(spin-down) = {results['probability_spin_down']:.2f}\n"
            f"P(initialize_spin_up) = {results['probability_initialize_spin_up']:.3f}\n"
            f"t_read_optimum = {results['t_read_optimum']*1e6:.0f} us\n"
            f"contrast_optimum = {results['contrast_optimum']:.2f}\n"
        )

    if detailed:
        print('\n***Detailed results***')
        for in_out in ['in', 'out']:
            title = f'Tunnel {in_out} statistics'
            print(f"\n{title}")
            tunnel_times_analyses[f'tunnel_{in_out}'].print_results()
            tunnel_times_analyses[f'tunnel_{in_out}'].plot_tunnel_times(title=title)

        if high_analysis is not None:
            fig, axes = plt.subplots(3, 1, figsize=(6,5.5))
            title = f'High spin-up - tunnel statistics'
            print(f"\n{title}")
            high_analysis.print_results()
            high_analysis.plot_tunnel_times(axes=axes[:2], plot_fast=True, title=title)

            title = f'Low spin-up - tunnel statistics'
            print(f"\n{title}")
            low_analysis.print_results()
            low_analysis.plot_tunnel_times(axes=axes[2:], plot_fast=False, title=title)

        fig.tight_layout()

    return results
