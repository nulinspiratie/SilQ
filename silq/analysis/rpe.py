try:
    from quapack.pyRPE import RobustPhaseEstimation
    from quapack.pyRPE.quantum import Q
except ImportError:
    raise ImportError('RPE tools need to be installed before use, '
                      'see https://gitlab.com/quapack/pyrpe')

import numpy as np
from ..tools.general_tools import count_num_decimal_places

from matplotlib import pyplot as plt

__all__ = ['analyse_RPE', 'rpe_get_best_estimate', 'plot_rpe_results']

def analyse_RPE(dataset, gate_name, max_length: int = None):
    """Finds the estimated rotation angle of an operation from an RPE experiment.

    Args:
        dataset: A pygsti dataset
        gate_name: The name of the operation being tested.
        max_length: The maximum circuit depth to perform the analysis up to.
                    The analysis automatically calculates an estimated angle for
                    all powers of 2 up to max_length.

    Returns:

    """
    rpe_container = Q()

    circuit_list = dataset.keys()
    length = 1
    while True:
        if max_length is not None and length > max_length:
            break
        try:
            cos_circ = next(circuit_list)
            sin_circ = next(circuit_list)
        except StopIteration:
            break

        rpe_container.process_cos(length, (
        int(dataset[cos_circ]['0']), int(dataset[cos_circ]['1'])))
        rpe_container.process_sin(length, (
        int(dataset[sin_circ]['1']), int(dataset[sin_circ]['0'])))

        length *= 2

    rpe_analyzer = RobustPhaseEstimation(rpe_container)
    return rpe_container, rpe_analyzer

def rpe_get_best_estimate(rpe_analyzer:RobustPhaseEstimation, include_precision=False, **kwargs):
    """Gives the best estimate

    Args:
        rpe_analyzer: A RobustPhaseEstimation instance
        include_precision: If True, returns a tuple of the estimated rotation angle
         and the approximate precision to which it is known.

        **kwargs:

    Returns:

    """
    best_idx = rpe_analyzer.check_unif_local(**kwargs)

    if include_precision:
        precision = np.pi/(2 ** (best_idx + 1))
        return rpe_analyzer.angle_estimates[best_idx], precision

    return rpe_analyzer.angle_estimates[best_idx]

def plot_rpe_results(rpe_analyzer, target_angle = 0, plot_delta=False,
                     unit = 'deg', ax=None, **kwargs):
    """

    Args:
        rpe_analyzer:
        ax:

    Returns:

    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    max_lengths = np.array([2 ** i for i in range(len(rpe_analyzer.angle_estimates))])
    yerrs = np.pi/(2*max_lengths) # PRL 118, 190502 (2017)
    best_idx = rpe_analyzer.check_unif_local(**kwargs)
    (best_angle, precision) = rpe_get_best_estimate(rpe_analyzer, include_precision=True)
    n_digits = count_num_decimal_places(precision)

    best_estimate = '{0:.{1}f}'.format(round(best_angle, n_digits), n_digits) + \
                    '({0:.1g})'.format(round(precision, n_digits) * (10 ** n_digits))

    best_estimate = '{0:.{1}f}'.format(round(best_angle, n_digits)/np.pi, n_digits)\
                    + '({0:.1g})'.format(round(precision, n_digits)/np.pi * (10 ** n_digits))

    if unit == 'deg':
        scale = 180 / np.pi
        yunit = r'($^\circ\!$)'
    elif unit == 'rad':
        scale = 1
        yunit = r'(rad)'
    elif unit == 'pi':
        scale = 1 / np.pi
        yunit = r'($\times \pi$)'
    elif unit == '2pi':
        scale = 1 / (2 * np.pi)
        yunit = r'($\times 2\pi$)'



    if plot_delta:
        ax.plot(max_lengths, (rpe_analyzer.angle_estimates - target_angle) * scale,
                marker='.', label=r'$\langle \theta \rangle = \pi\cdot$' + best_estimate)

        ax.fill_between(max_lengths,
                        y1=(rpe_analyzer.angle_estimates - target_angle - yerrs) * scale,
                        y2=(rpe_analyzer.angle_estimates - target_angle + yerrs) * scale,
                        alpha=0.5, zorder=-1,
                        )

        ax.axhline(0, c='k', zorder=-1, label=r'$\Delta \theta = 0$')

        ylabel = r'$\Delta \theta$ ' + yunit
    else:
        ax.plot(max_lengths, rpe_analyzer.angle_estimates * scale,
                marker='.', label=r'$\langle \theta \rangle = \pi\cdot$' + best_estimate)

        ax.fill_between(max_lengths,
                        y1=(rpe_analyzer.angle_estimates - yerrs) * scale,
                        y2=(rpe_analyzer.angle_estimates + yerrs) * scale,
                        alpha=0.5, zorder=-1,
                        )

        ax.axhline(target_angle * scale, c='k', zorder=-1,
                   label=r'$\theta = \pi\cdot$' + f'{target_angle / np.pi}')

        ylabel = r'$\theta$ ' + yunit


    ax.set_xscale('log', base=2)
    ax.set_xticks(max_lengths)
    ax.set_xlabel('Number of gates in generation $N_k$')
    ax.set_ylabel(ylabel)
    ax.axvline(max_lengths[best_idx], lw=2, c='r', label='best estimate length')
    ax.legend(loc='upper center')

    plt.tight_layout()

    return fig, ax

    # plt.axhline(/np.pi, c='k', zorder=-1)
    # plt.axhline((np.pi/2 + delta_y)/np.pi, c='k', zorder=-1)