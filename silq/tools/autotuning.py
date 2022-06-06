"""Functions for auto-tuning using the Nelder Mead optimization algorithm"""

from scipy import optimize
from qcodes import Sweep


def _create_initial_simplex(initial_values: dict, delta: float = None, **kwargs):
    if delta is None:
        assert len(kwargs) == len(initial_values)

    x0 = list(initial_values.values())
    initial_simplex = [x0]
    for k, name in enumerate(initial_values):
        x0_copy = x0[:]
        x0_copy[k] += kwargs.get(name, delta)
        initial_simplex.append(x0_copy)
    initial_simplex = np.array(initial_simplex)
    return initial_simplex


def optimize_Nelder_mead(
    target_function,
    initial_values: dict,
    delta: dict,
    maximum_evaluations: 100,
    tolerance: float = None,
    msmt=None,
    kwargs={},
):
    if msmt is not None:
        sweep_vals = iter(Sweep(range(maximum_evaluations + 30), "repetitions"))
        kwargs["msmt"] = msmt
    else:
        sweep_vals = None

    individual_results = []
    def evaluate_args_to_kwargs(args):
        arg_kwargs = dict(zip(initial_values, args))

        if sweep_vals is not None:
            next(sweep_vals)
        if msmt is not None:
            msmt.measure(arg_kwargs, "set_values")

        result = target_function(**arg_kwargs, **kwargs)

        # Result is either an int/float score to be minimized,
        # or a dict containing outputs and a key 'score'
        if isinstance(result, dict):
            assert 'score' in result
            score = result.pop('score')
            outputs = result
        elif isinstance(result, (int, float)):
            outputs = None
            score = result
        else:
            raise ValueError(f'Output must be dict, int or float: {result}')

        individual_results.append({
            'parameters': arg_kwargs,
            'score': score,
            'outputs': outputs
        })

        return score

    initial_simplex = _create_initial_simplex(initial_values, **delta)

    optimize.minimize(
        fun=evaluate_args_to_kwargs,
        x0=initial_simplex[0],
        method="Nelder-Mead",
        options=dict(initial_simplex=initial_simplex, maxfev=maximum_evaluations),
        tol=tolerance,
    )

    if msmt:
        msmt.step_out()

    parameters = {}
    for parameter in initial_values:
        parameters[parameter] = [
            result['parameters'][parameter] for result in individual_results
        ]

    outputs = {}
    if individual_results[0]['outputs'] is not None:
        for output in individual_results[0]['outputs']:
            outputs[output] = np.array([
                result['outputs'][output] for result in individual_results
            ])


    scores = [result['score'] for result in individual_results]
    optimal_idx = np.argmin(scores)

    results = {
        'optimum': individual_results[optimal_idx],
        'final': individual_results[-1],
        'parameters': parameters,
        'score': scores,
        'outputs': outputs,
        'individual_results': individual_results,
    }
    return results


# ignore_mode: analysis
from qcodes import MatPlot, Measurement, station
import numpy as np

# from scripts.autotune.nelder_mead import optimize_Nelder_mead

delta_retune = {gate: 10e-3 for gate in station.DC_gates.parameters}
delta_retune['SRC'] = 150e-6
delta_retune['TG'] = 3e-3


def plot_retune(results):
        plot = MatPlot(
            [*results['outputs']['up_proportions'].T, results['outputs']['contrast']],
            figsize=(9, 5), sharex=True, subplots=(2,1)
        )
        for key, val in results['parameters'].items():
            plot[1].add((val - val[0])*1e3, label=key)

        plot[0].set_ylabel('Fraction')
        plot[0].set_ylim(0, 1)
        plot[0].set_xlim(0, len(results['score']))
        plot[0].legend(['Up_proportion_D', 'Up_proportion_U', 'Contrast'])
        plot[0].grid('on')

        plot[1].set_xlabel('Iteration')
        plot[1].set_ylabel('Voltage difference (mV)')
        plot[1].legend()
        plot[1].grid('on')
        plot.tight_layout()


def retune_nelder_mead(
        target_function,
        initial_values=None,
        iterations=None,
        samples=100,
        plot=False,
        gates=('DBL', 'TG'),
        virtual=True
):
    if initial_values is None and not virtual:
        raise ValueError("Initial gate voltages must be provided if not using "
                         "virtual gates.")

    # Choose a ridiculously high contrast
    kwargs = dict(target_contrast=0.92, samples=samples, virtual=virtual)

    gate_names = [gate if isinstance(gate, str) else gate.name for gate in gates]

    if virtual:
        initial_values = {key: 0 for key in gate_names}
        station.virtual_gates.zero_offset()

    initial_values = {gate: initial_values[gate] for gate in gate_names}
    delta = {gate: delta_retune[gate] for gate in gate_names}

    # Choose iterations dependent on number of gates
    if iterations is None:
        iterations = 10 + 10 * len(gates)

    with Measurement('retune', notify=False) as msmt:
        results = optimize_Nelder_mead(
            target_function=target_function,
            initial_values=initial_values,
            delta=delta,
            kwargs=kwargs,
            maximum_evaluations=iterations,
            tolerance=0.001,  # Use too low number so it doesn't stop prematurely
            msmt=msmt,
        )

        contrasts = [result['outputs']['contrast'] for result in results['individual_results']]
        final_contrast = np.mean(contrasts[-4:])
        msmt.measure(final_contrast, 'final_contrast')

    msmt.log(f"Initial contrast: {results['outputs']['contrast'][0]:.2f}")
    msmt.log(f"optimum contrast: {results['optimum']['outputs']['contrast']:.2f}")
    msmt.log(f"Final contrast: {final_contrast:.2f}")

    if plot:
        plot_retune(results)

    return results


def measure_ESR_contrast(esr_parameter, msmt=None, samples=100, filter=False,
                         setup=False, silent=True, **kwargs):
    """Measure contrast using an ESR parameter.

        Requires that the ESR parameter measures at least two sets of up
        proportions. Normally this would be done by measuring the response to
        two different frequencies, or by enabling the EPR pulse sequence to
        determine dark counts.

    """
    if setup:
        esr_parameter.setup()

    if msmt:
        msmt.measure(esr_parameter, samples=samples, **kwargs)
    else:
        esr_parameter(save_traces=False, samples=samples, **kwargs)


    if esr_parameter.EPR.enabled:
        contrast = esr_parameter['contrast']
    else:
        up_proportions = np.array([
            val for key, val in esr_parameter.results.ESR.items()
            if 'up_proportion' in key and 'idx' not in key
        ])
        assert len(up_proportions) >= 2, "Need at least two sets of " \
                                         "up proportions to calculate contrast."
        contrast = abs(up_proportions[0] - up_proportions[1])

        if msmt:
            msmt.measure(contrast, 'contrast')

        if filter and (esr_parameter.results['ESR.num_traces0'] != samples or
                       esr_parameter.results['ESR.num_traces1'] != samples):
            contrast = 0

    if not silent:
        esr_parameter.print_results()

        max_voltage = max(
            np.max(arr_dict['output']) for arr_dict in esr_parameter.traces.values()
        )
        esr_parameter.plot_traces(clim=(0, max_voltage))

    return {
        'contrast': contrast,
        'up_proportions': up_proportions
    }
