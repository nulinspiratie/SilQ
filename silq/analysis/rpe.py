try:
    from quapack.pyRPE import RobustPhaseEstimation
    from quapack.pyRPE.quantum import Q
except ImportError:
    raise ImportError('RPE tools need to be installed before use, '
                      'see https://gitlab.com/quapack/pyrpe')

import numpy as np
from ..tools.general_tools import count_num_decimal_places

from matplotlib import pyplot as plt

__all__ = ['analyze_rpe', 'get_rotation_angle', 'get_axis_angle', 'plot_rpe_results']


def analyze_rpe(dataset, max_length: int = None):
    """Finds the estimated rotation angle of a unitary used in an RPE experiment.

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


def get_epsilon(target_angle, measured_angle):
    """Get the epsilon parameter which measured the deviation between a target
        rotation angle and the actual rotation angle.

    Returns:
        epsilon = (measured_angle / target_angle) - 1
    """
    return (measured_angle / target_angle) - 1


def get_rotation_angle(rpe_analyzer: RobustPhaseEstimation,
                       include_precision=False, **kwargs):
    """Gives the best estimate for the rotation angle from an RPE experiment.

    Note: This should only be used for an RPE experiment designed to probe the
          rotation angle phi.

    Args:
        rpe_analyzer: A RobustPhaseEstimation instance
        include_precision: If True, returns a tuple of the estimated rotation angle
         and the approximate precision to which it is known.

        **kwargs: Passed to RobustPhaseEstimation methods.

    Returns:

    """
    best_idx = rpe_analyzer.check_unif_local(**kwargs)

    if include_precision:
        precision = np.pi/(2 ** (best_idx + 1))
        return rpe_analyzer.angle_estimates[best_idx], precision

    return rpe_analyzer.angle_estimates[best_idx]


def _extract_theta_from_Phi(Phi, epsilon=0, include_precision=False,
                            Phi_precision=None):
    theta = np.pi / 2 - np.sin(Phi / 2) / (2 * np.cos(np.pi * epsilon / 2))

    if include_precision:
        if Phi_precision is None:
            raise ValueError("Precision of angle Phi must be provided when "
                             "calculating the precision of theta.")
        theta_precision = Phi_precision / (4 * np.cos(np.pi * epsilon / 2))
        return theta, theta_precision

    return theta


def get_axis_angle(rpe_analyzer: RobustPhaseEstimation,
                   include_precision=False, epsilon=0, **kwargs):
    """Gives the best estimate for the angle theta between the X/Y axes.

    Note: This should only be used for an RPE experiment designed to probe the
    axis angle theta.

    Args:
        rpe_analyzer: A RobustPhaseEstimation instance
        include_precision: If True, returns a tuple of the estimated rotation
                    angle and the approximate precision to which it is known.
        epsilon: The parameter that quantifies the amount of over/under-rotation
                from a target rotation angle. Typically this is measured and
                then calibrated with a separate RPE experiment.

        **kwargs: Passed to RobustPhaseEstimation methods.

    Returns:

    """
    best_idx = rpe_analyzer.check_unif_local(**kwargs)
    Phi = rpe_analyzer.angle_estimates[best_idx]
    Phi_precision = np.pi / (2 ** (best_idx + 1))

    return _extract_theta_from_Phi(Phi, epsilon, include_precision, Phi_precision)


def plot_rpe_results(rpe_analyzer, target_parameter='phi', target_angle=0,
                     plot_delta=False, unit='deg', ax=None, **kwargs):
    """Plots the historical estimates of the target parameter with estimate
        precision shown as a shaded region.

    Args:
        rpe_analyzer: A RobustPhaseEstimation instance
        target_parameter: 'phi' or 'theta'
        target_angle: The desired angle for the target parameter.
        plot_delta: If True, plots the difference between the estimated angles
                    and the target angle.
        unit: If 'deg' plot angle in degrees.
              If 'rad' plot angle in radians.
              If 'pi' plot angle as multiple of pi.
              If '2pi' plot angle as multiple of 2*pi.
        ax: If provided, plot on an existing set of axes.
        **kwargs: Plot kwargs.

    Returns: Tuple (fig, ax) for figure and axis instances.


    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    max_lengths = np.array([2 ** i for i in range(len(rpe_analyzer.angle_estimates))])
    best_idx = rpe_analyzer.check_unif_local(**kwargs)

    if target_parameter == 'phi':
        (best_angle, precision) = get_rotation_angle(rpe_analyzer, include_precision=True)
        yerrs = np.pi/(2*max_lengths) # PRL 118, 190502 (2017)
        angle_estimates = rpe_analyzer.angle_estimates
    elif target_parameter == 'theta':
        (best_angle, precision) = get_axis_angle(rpe_analyzer, include_precision=True)
        angle_estimates = [_extract_theta_from_Phi(Phi, include_precision=True,
                            Phi_precision=np.pi/(2 ** (k + 1)))
                           for k, Phi in enumerate(rpe_analyzer.angle_estimates)]

        angle_estimates, yerrs = zip(*angle_estimates)
        angle_estimates = np.array(angle_estimates)
        yerrs = np.array(yerrs)
    else:
        raise ValueError(f'Target parameter "{target_parameter}" not understood, must be one of: "phi" or "theta"')

    n_digits = count_num_decimal_places(precision)

    best_estimate = '{0:.{1}f}'.format(round(best_angle, n_digits)/np.pi, n_digits)\
                    + '({0:.1g})'.format(round(precision, n_digits)/np.pi * (10 ** n_digits))

    if unit == 'deg':
        scale = 180.0 / np.pi
        yunit = r'($^\circ\!$)'
    elif unit == 'rad':
        scale = 1
        yunit = r'(rad)'
    elif unit == 'pi':
        scale = 1 / np.pi
        yunit = r'($\times \pi$ rad)'
    elif unit == '2pi':
        scale = 1 / (2 * np.pi)
        yunit = r'($\times 2\pi$ rad)'

    plot_label = f'$\\langle \\{target_parameter} \\rangle = \\pi\\cdot$' \
                 + best_estimate


    if plot_delta:
        target_label = f'$\\Delta \\{target_parameter} = 0'
        ax.plot(max_lengths, (angle_estimates - target_angle) * scale,
                marker='.', label=plot_label)

        ax.fill_between(max_lengths,
                        y1=(angle_estimates - target_angle - yerrs) * scale,
                        y2=(angle_estimates - target_angle + yerrs) * scale,
                        alpha=0.5, zorder=-1,
                        )

        ax.axhline(0, c='k', zorder=-1, label=target_label)

        ylabel = f'$\\Delta \\{target_parameter}$ ' + yunit
    else:
        target_label = f'$\\{target_parameter} = \\pi\\cdot$' + f'{target_angle / np.pi}'
        ax.plot(max_lengths, angle_estimates * scale,
                marker='.', label=plot_label)

        ax.fill_between(max_lengths,
                        y1=(angle_estimates - yerrs) * scale,
                        y2=(angle_estimates + yerrs) * scale,
                        alpha=0.5, zorder=-1,
                        )

        ax.axhline(target_angle * scale, c='k', zorder=-1, label=target_label)

        ylabel = f'$\\{target_parameter}$ ' + yunit


    ax.set_xscale('log', base=2)
    ax.set_xticks(max_lengths)
    ax.set_xlabel('Number of gates in generation $N_k$')
    ax.set_ylabel(ylabel)
    ax.axvline(max_lengths[best_idx], lw=2, c='r', label='best estimate length')
    ax.legend(loc='upper center')

    plt.tight_layout()

    return fig, ax
