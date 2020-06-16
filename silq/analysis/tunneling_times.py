from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from typing import Union
import numpy as np
import logging
from collections import Counter
from copy import copy
import h5py
from statsmodels.base.model import GenericLikelihoodModel

from .analysis import find_high_low, count_blips
from .fit_toolbox import DoubleExponentialFit, ExponentialFit
from silq.tools.trace_tools import extract_pulse_slices_from_trace_file


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


def get_first_blips(
        traces: Union[h5py.File, np.ndarray],
        threshold_voltage: float,
        sample_rate: float,
        pulse_slice=None,
        silent=False
):
    """Get first blips from an array"""
    if isinstance(traces, h5py.File):
        pulse_slices = extract_pulse_slices_from_trace_file(
            traces_file=traces, sample_rate=sample_rate
        )
        results = {}
        for pulse_name, pulse_slice in pulse_slices.items():
            result = results[pulse_name] = get_first_blips(
                traces=traces['traces']['output'],
                threshold_voltage=threshold_voltage,
                sample_rate=sample_rate,
                pulse_slice=pulse_slice,
                silent=True
            )
            result['pulse_slice'] = pulse_slice

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
            result = get_first_blips(
                trace_arr,
                threshold_voltage=threshold_voltage,
                sample_rate=sample_rate,
                pulse_slice=pulse_slice,
                silent=True
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
            ignore_final=True
        )
        first_blip_events = [
            elem[0][1] / sample_rate for elem in blip_results['blip_events'] if len(elem)
        ]

        N_traces_blips = len(first_blip_events)
        N_traces_no_blips = samples - N_traces_blips
        N_traces = N_traces_blips + N_traces_no_blips

        results = {
            'N_traces_blips': N_traces_blips,
            'N_traces_no_blips': N_traces_no_blips,
            'N_traces': N_traces,
            'first_blip_events': first_blip_events,
        }

    if not silent:
        print(
            f"Traces without blips: {results['N_traces_no_blips']}/{results['N_traces']} "
            f"({results['N_traces_no_blips'] / results['N_traces']*100:.1f}%)"
        )

    return results



class DoubleExponentialModel(GenericLikelihoodModel):
    parameter_names = ['A_1', 'tau_1', 'tau_2']

    def __init__(self, endog, exog=None, **kwargs):
        if exog is None:
            exog = np.zeros_like(endog)

        self.results = None
        self.initial_parameters = None
        self.parameters = None

        super(DoubleExponentialModel, self).__init__(endog, exog, **kwargs)

    @staticmethod
    def calculate_A_2(A_1, tau_1, tau_2):
        return (1 - A_1 * tau_1) / tau_2

    def find_initial_parameters(self):
        results = dict(
            tau_1 = 100e-6,
            tau_2 = 1e-3
        )
        results['A_1'] = 1/results['tau_1']/2
        return results

    def fit_function(self, t, A_1, tau_1, tau_2, **kwargs):
        A_2 = self.calculate_A_2(A_1, tau_1, tau_2)
        if min(A_1, A_2) <= 0 or min(tau_1, tau_2) <= 0:
            return 1e-15
        result = A_1*np.exp(-t/tau_1) + A_2*np.exp(-t/tau_2)
        return result

    def nloglikeobs(self, params):
        params = dict(zip(self.parameter_names, params))
        return -np.log(self.fit_function(self.endog, **params))

    def fit(self, initial_parameters=None, maxiter=10000, maxfun=5000, silent=False, **kwargs):
        if initial_parameters is None:
            initial_parameters = self.find_initial_parameters()
        self.initial_parameters = initial_parameters

        start_params = [self.initial_parameters[name] for name in self.parameter_names]
        self.results = super(DoubleExponentialModel, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, disp=not silent, **kwargs
        )
        self.parameters = dict(zip(self.parameter_names, self.results.params))

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

        self.t_skip = t_skip

        # Perform fitting
        self.model = None
        self.fit_results = None
        self.results = self.fit(silent=silent)

        if not silent:
            self.print_results()

    @staticmethod
    def parse_tunnel_times(
            tunnel_times=None, traces=None, results=None,
            threshold_voltage=None, sample_rate=None
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
            results = get_first_blips(
                traces, threshold_voltage=threshold_voltage, sample_rate=sample_rate
            )
            return {
                'tunnel_times': results['first_blip_events'],
                'N_traces_blips': results['N_traces_blips'],
                'N_traces_no_blips': results['N_traces_no_blips'],
                'N_traces': results['N_traces']
            }
        elif results is not None:
            return {
                'tunnel_times': results['first_blip_events'],
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
        self.tunnel_times = copy(self.original_tunnel_times)
        self.tunnel_times -= t_skip
        self.tunnel_times[self.tunnel_times < 0] = 0

        optimal_t_skip = self.optimal_t_skip()
        if optimal_t_skip is not None and t_skip < optimal_t_skip:
            logger.warning(
                f't_skip {t_skip*1e6} us below optimum {optimal_t_skip*1e6:.0f} us'
            )

    def optimal_t_skip(self):
        min_events = 30  # Minimum number of events at single tunnel time
        threshold_factor = 0.4  # Minimum factor of maximum tunnel time events
        max_tunnel_time = 100e-6

        tunnel_times_count = Counter(sorted(self.original_tunnel_times))
        max_tunnel_times = max(tunnel_times_count.values())

        if max_tunnel_times < min_events:
            logger.warning('Could not determine optimal t_skip, too few data points')
            return None

        for tunnel_time, counts in tunnel_times_count.items():
            if counts < threshold_factor * max_tunnel_times:
                continue
            elif tunnel_time > max_tunnel_time:
                logger.warning(
                    f'Could not determine optimal t_skip, exceeded {max_tunnel_time}'
                )
                return None
            else:
                return tunnel_time
        else:
            logger.warning(f'Could not determine optimal t_skip, unknown reason')
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

        self.model = DoubleExponentialModel(tunnel_times)
        self.fit_results = self.model.fit(silent=silent)

        results = copy(self.model.parameters)

        results = self.get_ancillary_parameters(**results)
        return results

    def print_results(self):
        for key, value in self.results.items():
            if 'tau' in key:
                print(f"{key}: {value*1e6:.0f} us")
            else:
                print(f"{key}: {value:.3g}")

    def plot_tunnel_times(self, tunnel_times=None, t_cutoff=None, bins=41):
        if tunnel_times is None:
            tunnel_times = self.tunnel_times

        if t_cutoff is None:
            if self.results is None:
                t_cutoff = 5e-3
            else:
                t_cutoff = 15 * self.results['tau_1']

        tunnel_times_fast = np.array([t for t in tunnel_times if t <= t_cutoff])

        fig, axes = plt.subplots(2, 1, figsize=(6, 4))

        for ax, tunnel_times_arr in zip(axes, [tunnel_times_fast, tunnel_times]):
            # Normalize probabilities to tunnel times
            y, x, _ = ax.hist(tunnel_times_arr*1e3, bins=bins)
            x /= 1e3

            dx = np.diff(x)[0]
            x = (x[:-1] + x[1:]) / 2
            probabilities = self.model.fit_function(x, **self.model.parameters) * dx
            probabilities *= len(tunnel_times)
            ax.plot(x*1e3, probabilities)
            ax.set_yscale('log')

        fig.tight_layout()

        return fig, axes


def analyse_tunnel_times_measurement(
    data=None,
    tunnel_times_analyses=None,
    threshold_voltage=150e-3,
    t_skip=0,
    sample_rate=200e3,
    detailed=True
):
    if data is not None:
        trace_file = data.load_traces('ESR_adiabatic')
        results = get_first_blips(trace_file, threshold_voltage=threshold_voltage, sample_rate=sample_rate)

        tunnel_times_analyses = {'high': None, 'low': None}
        for k, (key, result) in enumerate(results.items()):
            tunnel_times_analysis = TunnelTimesAnalysis(
                results=result, t_skip=t_skip,
                sample_rate=sample_rate, threshold_voltage=threshold_voltage,
                silent=True
            )

            probability_spin_up = tunnel_times_analysis.results["p_1"]
            if probability_spin_up > 0.5:
                tunnel_times_analyses['high'] = tunnel_times_analysis
                print(f"Target nucleus: {['D', 'U'][k]}")
            else:
                tunnel_times_analyses['low'] = tunnel_times_analysis

    assert tunnel_times_analyses['low'] is not None, "Both readouts have high spin-up fraction"
    assert tunnel_times_analyses['high'] is not None, "Both readouts have low spin-up fraction"

    # Print results
    print(
        f"tau_up = {tunnel_times_analyses['high'].results['tau_1']*1e6:.0f} us\n"
        f"tau_down = {tunnel_times_analyses['high'].results['tau_2']*1e6:.0f} us\n"
        f"P(spin-up) = {tunnel_times_analyses['high'].results['p_1']:.2f}\n"
        f"P(spin-down) = {tunnel_times_analyses['high'].results['p_2']:.2f}\n"
        f"P(initialize_spin_up) = {tunnel_times_analyses['low'].results['p_1']:.3f}\n"
        f"t_read_optimum = {tunnel_times_analyses['high'].results['t_read_optimum']*1e6:.0f} us\n"
        f"contrast_optimum = {tunnel_times_analyses['high'].results['contrast_optimum']:.2f}\n"
    )

    if detailed:
        print("\n\n\nDetailed results:")
        for key, val in tunnel_times_analyses.items():
            print(f"\n{key} tunnel statistics")
            val.print_results()
            val.plot_tunnel_times()

    return tunnel_times_analyses
