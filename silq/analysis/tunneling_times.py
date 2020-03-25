import numpy as np
from matplotlib import pyplot as plt

from .analysis import find_high_low, count_blips
from .fit_toolbox import DoubleExponentialFit, ExponentialFit


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