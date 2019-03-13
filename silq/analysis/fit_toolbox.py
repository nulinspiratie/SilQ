from typing import Union, Tuple
import numpy as np
from lmfit import Parameters, Model
from lmfit.model import ModelResult
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
import logging
from qcodes.data.data_array import DataArray

__all__ = ['Fit',
           'LinearFit',
           'ExponentialFit',
           'SineFit',
           'ExponentialSineFit',
           'DoubleExponentialFit',
           'RabiFrequencyFit',
           'AMSineFit']

logger = logging.getLogger(__name__)


class Fit():
    """Base fitting class.

    To fit a specific function, this class should be subclassed.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    plot_kwargs = {'linestyle': '--', 'color': 'cyan', 'lw': 3}
    sweep_parameter = None

    def __init__(self, fit=True, print=False, plot=None, **kwargs):
        self.model = Model(self.fit_function)
        self.fit_result = None

        self.plot_handle = None

        self.xvals = None
        self.ydata = None
        self.weights = None
        self.parameters = None

        if kwargs:
            self.get_parameters(**kwargs)
            if fit:
                self.perform_fit(print=print, plot=plot, **kwargs)

    def fit_function(self, *args, **kwargs):
        raise NotImplementedError('This should be implemented in a subclass')

    def find_initial_parameters(self,
                                xvals: np.ndarray,
                                ydata: np.ndarray,
                                initial_parameters: dict) -> Parameters:
        """Estimate initial parameters from data.

        This is needed to ensure that the fitting will converge.

        Args:
            xvals: x-coordinates of data points
            ydata: data points
            initial_parameters: Fixed initial parameters to be skipped.

        Returns:
            Parameters object containing initial parameters.
        """
        raise NotImplementedError('This should be implemented in a subclass')

    def find_nearest_index(self, array, value):
        """Find index in array that is closest to a value

        Args:
            array: array from which to find the nearest index
            value: target value for which we want the index of the nearest
                array element

        Returns:
            Index of element in array that is closest to target value
        """
        idx = np.abs(array - value).argmin()
        return array[idx]

    def find_nearest_value(self,
                           array: np.ndarray,
                           value: float) -> float:
        """Find value in array that is closest to target value.

        Args:
            array: array from which to find the nearest value.
            value: target value for which we want the nearest array element.

        Returns:
            element in array that is nearest to value.
        """
        return array[self.find_nearest_index(array, value)]

    def get_parameters(self, xvals, ydata, initial_parameters={},
                       fixed_parameters={}, weights=None):
        """Get parameters for fitting
        Args:
            xvals: x-coordinates of data points
            ydata: Data points
            initial_parameters: {parameter: initial_value} combination,
                to specify the initial value of certain parameters. The initial
                values of other parameters are found using
                `Fit.find_initial_parameters`.
            fixed_parameters: {parameter: fixed_value} combination,
                to specify parameters whose values should not be varied.
            weights: Weights for data points, must have same shape as ydata
        """
        if isinstance(xvals, DataArray):
            xvals = xvals.ndarray
        elif not isinstance(xvals, np.ndarray):
            xvals = np.array(xvals)
        if isinstance(ydata, DataArray):
            ydata = ydata.ndarray
        elif not isinstance(ydata, np.ndarray):
            ydata = np.array(ydata)
        # Filter out all NaNs
        non_nan_indices = ~(np.isnan(xvals) | np.isnan(ydata))
        xvals = xvals[non_nan_indices]
        ydata = ydata[non_nan_indices]
        if weights is not None:
            weights = weights[non_nan_indices]
        self.xvals = xvals
        self.ydata = ydata
        self.weights = weights
        # Find initial parameters, also pass fixed parameters along so they are
        # not modified
        self.parameters = self.find_initial_parameters(
            xvals, ydata, initial_parameters={**fixed_parameters,
                                              **initial_parameters})

        # Ensure that fixed parameters do not vary
        for key, value in fixed_parameters.items():
            self.parameters[key].vary = False
        return self.parameters

    def perform_fit(self,
                    parameters=None,
                    print=False,
                    plot=None,
                    **kwargs) -> ModelResult:
        """Perform fitting routine

        Returns:
            ModelResult object containing fitting results
        """

        if parameters is None:
            if kwargs:
                self.get_parameters(**kwargs)
            parameters = self.parameters

        self.fit_result = self.model.fit(self.ydata, params=parameters,
                                         weights=self.weights,
                                         **{self.sweep_parameter: self.xvals})

        if print:
            self.print_results()

        if plot is not None:
            self.add_to_plot(plot)

        return self.fit_result

    def print_results(self):
        if self.fit_result is None:
            logger.warning('No fit results')
            return

        fit_report = self.fit_result.fit_report(show_correl=False)
        lines = fit_report.splitlines()
        variables_idx = lines.index('[[Variables]]') + 1
        lines = lines[variables_idx:]
        lines = '\n'.join(lines)
        print(lines)

    def add_to_plot(self,
                    ax: Axis,
                    N: int = 201,
                    xrange: Tuple[float] = None,
                    x_range: Tuple[float] = None,
                    xscale: float = 1,
                    yscale: float = 1,
                    **kwargs):
        """Add fit to existing plot axis

        Args:
            ax: Axis to add plot to
            N: number of points to use as x values (to smoothe fit curve)
            xrange: Optional range for x values (min, max)
            x_range: Same as xrange (deprecated)
            xscale: value to multiple x values by to rescale axis
            yscale: value to multiple y values by to rescale axis
            kwargs: Additional plot kwargs. By default Fit.plot_kwargs are used

        Returns:
            plot_handle of fit curve
        """
        if x_range is not None:
            DeprecationWarning('Please use xrange instead of x_range')
            xrange = x_range

        if xrange is None:
            x_vals = self.xvals
            x_vals_full = np.linspace(min(x_vals), max(x_vals), N)
        else:
            x_vals_full = np.linspace(*xrange, N)

        y_vals_full = self.fit_result.eval(
            **{self.sweep_parameter: x_vals_full})
        x_vals_full *= xscale
        y_vals_full *= yscale
        plot_kwargs = {**self.plot_kwargs, **kwargs}
        self.plot_handle, = ax.plot(x_vals_full,
                                   y_vals_full,
                                  **plot_kwargs)
        return self.plot_handle


class LinearFit(Fit):
    """Fitting class for an exponential function.

        To fit data to a function, use the method `Fit.perform_fit`.
        This will find its initial parameters via `Fit.find_initial_parameters`,
        after which it will fit the data to `Fit.fit_function`.

        Note:
            The fitting routine uses lmfit, a wrapper package around scipy.optimize.
        """
    sweep_parameter = 't'

    @staticmethod
    def fit_function(t: Union[float, np.ndarray],
                     gradient: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            gradient:
            offset:

        Returns:
            linear data points
        """
        return gradient * t + offset

    def find_initial_parameters(self,
                                xvals: np.ndarray,
                                ydata: np.ndarray,
                                initial_parameters: dict) -> Parameters:
        """Estimate initial parameters from data.

        This is needed to ensure that the fitting will converge.

        Args:
            xvals: x-coordinates of data points
            ydata: data points
            initial_parameters: Fixed initial parameters to be skipped.

        Returns:
            Parameters object containing initial parameters.
        """
        if initial_parameters is None:
            initial_parameters = {}

        parameters = Parameters()
        if not 'gradient' in initial_parameters:
            initial_parameters['gradient'] = (ydata[-1] - ydata[0]) / \
                                            (xvals[-1] - xvals[0])
        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[0]

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        return parameters


class ExponentialFit(Fit):
    """Fitting class for an exponential function.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 't'

    @staticmethod
    def fit_function(t: Union[float, np.ndarray],
                     tau: float,
                     amplitude: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            tau: Decay constant.
            amplitude:
            offset:

        Returns:
            exponential data points
        """
        return amplitude * np.exp(-t / tau) + offset

    def find_initial_parameters(self,
                                xvals: np.ndarray,
                                ydata: np.ndarray,
                                initial_parameters: dict) -> Parameters:
        """Estimate initial parameters from data.

        This is needed to ensure that the fitting will converge.

        Args:
            xvals: x-coordinates of data points
            ydata: data points
            initial_parameters: Fixed initial parameters to be skipped.

        Returns:
            Parameters object containing initial parameters.
        """
        if initial_parameters is None:
            initial_parameters = {}

        parameters = Parameters()
        if not 'amplitude' in initial_parameters:
            initial_parameters['amplitude'] = ydata[1] - ydata[-1]
        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[-1]

        if not 'tau' in initial_parameters:
            exponent_val = (initial_parameters['offset']
                            + initial_parameters['amplitude'] / np.exp(1))
            nearest_idx = np.abs(ydata - exponent_val).argmin()
            initial_parameters['tau'] = -(xvals[1] - xvals[nearest_idx])

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['tau'].min = 0

        return parameters


class DoubleExponentialFit(Fit):
    """Fitting class for a double exponential function.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 't'

    @staticmethod
    def fit_function(t: Union[float, np.ndarray],
                     tau_1: float,
                     A_1: float,
                     A_2: float,
                     tau_2: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            tau_1: First decay constant.
            tau_2: Second decay constant.
            A_1: Amplitude of first decay
            A_2: Amplitude of second decay
            offset: Offset of double exponential (t -> infinity)

        Returns:
            exponential data points
        """
        return A_1 * np.exp(-t / tau_1) + A_2 * np.exp(-t / tau_2) + offset

    def find_initial_parameters(self,
                                xvals: np.ndarray,
                                ydata: np.ndarray,
                                initial_parameters: dict) -> Parameters:
        """Estimate initial parameters from data.

        This is needed to ensure that the fitting will converge.

        Args:
            xvals: x-coordinates of data points
            ydata: data points
            initial_parameters: Fixed initial parameters to be skipped.

        Returns:
            Parameters object containing initial parameters.
        """
        if initial_parameters is None:
            initial_parameters = {}

        parameters = Parameters()
        if not 'A_1' in initial_parameters:
            initial_parameters['A_1'] = (ydata[1] - ydata[-1]) / 2
        if not 'A_2' in initial_parameters:
            initial_parameters['A_2'] = (ydata[1] - ydata[-1]) / 2

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[-1]

        if not 'tau_1' in initial_parameters:
            exponent_val = (initial_parameters['offset']
                            + initial_parameters['A_1'] / np.exp(1)
                            + initial_parameters['A_2'] / np.exp(1))
            nearest_idx = np.abs(ydata - exponent_val).argmin()
            initial_parameters['tau_1'] = -(xvals[1] - xvals[nearest_idx])

        if not 'tau_2' in initial_parameters:
            initial_parameters['tau_2'] = initial_parameters['tau_1']

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['tau_1'].min = 0
        parameters['tau_2'].min = 0

        parameters['A_1'].min = 0
        parameters['A_2'].min = 0

        return parameters


class SineFit(Fit):
    sweep_parameter = 't'

    @staticmethod
    def fit_function(t: Union[float, np.ndarray],
                     amplitude: float,
                     frequency: float,
                     phase: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Sinusoidal fit using time as x-coordinates

        Args:
            t: Time
            amplitude:
            frequency:
            phase:
            offset:

        Returns:
            Sinusoidal data points
        """
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        """Estimate initial parameters from data.

        This is needed to ensure that the fitting will converge.

        Args:
            xvals: x-coordinates of data points
            ydata: data points
            initial_parameters: Fixed initial parameters to be skipped.

        Returns:
            Parameters object containing initial parameters.
        """
        parameters = Parameters()
        if 'amplitude' not in initial_parameters:
            initial_parameters['amplitude'] = (max(ydata) - min(ydata)) / 2

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips) / 2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips), dt)[:int(len(
            fft_flips) / 2)]
        frequency_idx = np.argmax(fft_flips_abs[1:]) + 1

        if 'frequency' not in initial_parameters:
            frequency = fft_freqs[frequency_idx]
            initial_parameters['frequency'] = frequency

            if plot:
                plt.figure()
                plt.plot(fft_freqs, fft_flips_abs, 'o')
                plt.plot(frequency, fft_flips_abs[frequency_idx], 'o', ms=8)
        if 'phase' not in initial_parameters:
            phase = np.pi / 2 + np.angle(fft_flips[frequency_idx])
            initial_parameters['phase'] = phase
        if 'offset' not in initial_parameters:
            initial_parameters['offset'] = (max(ydata) + min(ydata)) / 2

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        # Amplitude is a positive real.
        parameters['amplitude'].set(min=0)

        return parameters


class AMSineFit(Fit):
    sweep_parameter = 't'

    @staticmethod
    def fit_function(t: Union[float, np.ndarray],
                     amplitude: float,
                     frequency: float,
                     phase: float,
                     offset: float,
                     amplitude_AM: float,
                     frequency_AM: float,
                     phase_AM: float,
                     ) -> Union[float, np.ndarray]:
        """ Amplitude-Modulated Sinusoidal fit using time as x-coordinates

        Args:
            t: Time
            amplitude:
            frequency:
            phase:
            offset:
            amplitude_AM:
            frequency_AM:
            phase_AM:

        Returns:
            Amplitude-Modulated Sinusoidal data points
        """
        return (1 + amplitude_AM) * np.sin(
            2 * np.pi * frequency_AM * t + phase_AM) \
               * amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        """Estimate initial parameters from data.

        This is needed to ensure that the fitting will converge.

        Args:
            xvals: x-coordinates of data points
            ydata: data points
            initial_parameters: Fixed initial parameters to be skipped.

        Returns:
            Parameters object containing initial parameters.
        """
        parameters = Parameters()
        if 'amplitude' not in initial_parameters:
            initial_parameters['amplitude'] = (max(ydata) - min(ydata)) / 2

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips) / 2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips), dt)[:int(len(
            fft_flips) / 2)]
        frequency_idx = np.argmax(fft_flips_abs[1:]) + 1

        if 'frequency' not in initial_parameters:
            frequency = fft_freqs[frequency_idx]
            initial_parameters['frequency'] = frequency

            if plot:
                plt.figure()
                plt.plot(fft_freqs, fft_flips_abs, 'o')
                plt.plot(frequency, fft_flips_abs[frequency_idx], 'o', ms=8)
        if 'phase' not in initial_parameters:
            phase = np.pi / 2 + np.angle(fft_flips[frequency_idx])
            initial_parameters['phase'] = phase
        if 'offset' not in initial_parameters:
            initial_parameters['offset'] = (max(ydata) + min(ydata)) / 2

        if 'amplitude_AM' not in initial_parameters:
            initial_parameters['amplitude_AM'] = 0.1

        if 'frequency_AM' not in initial_parameters:
            frequency_AM = fft_freqs[frequency_idx]
            initial_parameters['frequency_AM'] = frequency_AM

            if plot:
                plt.figure()
                plt.plot(fft_freqs, fft_flips_abs, 'o')
                plt.plot(frequency_AM, fft_flips_abs[frequency_idx], 'o', ms=8)

        if 'phase_AM' not in initial_parameters:
            phase_AM = 0
            initial_parameters['phase_AM'] = phase_AM

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['amplitude'].set(expr='(1-offset)/(1+amplitude_AM)')
        parameters['amplitude_AM'].set(min=0, max=1)

        parameters['frequency'].set(min=0, max=np.Inf)
        parameters['frequency_AM'].set(min=0, max=np.Inf)

        parameters['phase'].set(min=0, max=2 * np.pi)
        parameters['phase_AM'].set(min=0, max=2 * np.pi)

        parameters['offset'].set(min=0, max=1)

        return parameters


class ExponentialSineFit(Fit):
    sweep_parameter = 't'

    @staticmethod
    def fit_function(t, amplitude, tau, frequency, phase, offset,
                     exponent_factor):
        exponential = np.exp(-np.power(t / tau, exponent_factor))
        sine = np.sin(2 * np.pi * frequency * t + phase)
        return amplitude * exponential * sine + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        parameters = Parameters()
        if 'amplitude' not in initial_parameters:
            initial_parameters['amplitude'] = (max(ydata) - min(ydata)) / 2

        if 'tau' not in initial_parameters:
            initial_parameters['tau'] = xvals[-1] / 2

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips) / 2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips), dt)[:int(len(fft_flips) / 2)]
        frequency_idx = np.argmax(fft_flips_abs[1:]) + 1

        if 'frequency' not in initial_parameters:
            frequency = fft_freqs[frequency_idx]
            initial_parameters['frequency'] = frequency

            if plot:
                plt.figure()
                plt.plot(fft_freqs, fft_flips_abs)
                plt.plot(frequency, fft_flips_abs[frequency_idx], 'o', ms=8)
        if 'phase' not in initial_parameters:
            phase = np.pi / 2 + np.angle(fft_flips[frequency_idx])
            initial_parameters['phase'] = phase
        if 'offset' not in initial_parameters:
            initial_parameters['offset'] = (max(ydata) + min(ydata)) / 2

        if not 'exponent_factor' in initial_parameters:
            initial_parameters['exponent_factor'] = 1

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['tau'].min = 0
        parameters['exponent_factor'].min = 0

        return parameters


class RabiFrequencyFit(Fit):
    sweep_parameter = 'f'

    @staticmethod
    def fit_function(f, f0, gamma, t):
        Omega = np.sqrt(gamma ** 2 + (2 * np.pi * (f - f0)) ** 2 / 4)
        return gamma ** 2 / Omega ** 2 * np.sin(Omega * t) ** 2

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        parameters = Parameters()

        max_idx = np.argmax(ydata)
        max_frequency = xvals[max_idx]

        if 'f0' not in initial_parameters:
            initial_parameters['f0'] = max_frequency

        if 'gamma' not in initial_parameters:
            if 't' in initial_parameters:
                initial_parameters['gamma'] = np.pi / initial_parameters[
                    't'] / 2
            else:
                FWHM_min_idx = np.argmax(xvals > max_frequency / 2)
                FWHM_max_idx = len(xvals) - np.argmax(
                    (xvals > max_frequency / 2)[::-1]) - 1
                initial_parameters['gamma'] = 2 * np.pi * (
                xvals[FWHM_max_idx] - xvals[FWHM_min_idx]) / 2

        if 't' not in initial_parameters:
            initial_parameters['t'] = np.pi / initial_parameters['gamma'] / 2

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])
        parameters['gamma'].min = 0

        parameters.add('f_Rabi', expr='gamma/2/pi')
        # parameters.add('amplitude', expr='gamma^2/ Omega^2')

        return parameters
