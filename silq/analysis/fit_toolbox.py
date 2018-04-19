from typing import Union
import numpy as np
from lmfit import Parameters, Model
from lmfit.model import ModelResult
from matplotlib import pyplot as plt
import logging
from qcodes.data.data_array import DataArray


__all__ = ['Fit', 'ExponentialFit', 'SineFit', 'ExponentialSineFit',
           'RabiFrequencyFit']

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

    def __init__(self, **kwargs):
        self.model = Model(self.fit_function)
        self.fit_result = None

        self.plot_handle = None

        if kwargs:
            self.perform_fit(**kwargs)

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

    def perform_fit(self,
                    xvals:np.ndarray,
                    ydata: np.ndarray,
                    initial_parameters: dict = {},
                    fixed_parameters: dict = {},
                    weights: np.ndarray = None,
                    print: bool = False,
                    plot: bool = False) -> ModelResult:
        """Perform fitting routine

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

        Returns:
            ModelResult object containing fitting results
        """
        if isinstance(xvals, DataArray):
            xvals = xvals.ndarray
        if isinstance(ydata, DataArray):
            ydata = ydata.ndarray

        # Filter out all NaNs
        non_nan_indices = ~(np.isnan(xvals) | np.isnan(ydata))
        xvals = xvals[non_nan_indices]
        ydata = ydata[non_nan_indices]

        # Find initial parameters, also pass fixed parameters along so they are
        # not modified
        parameters = self.find_initial_parameters(
            xvals, ydata, initial_parameters={**fixed_parameters,
                                              **initial_parameters})

        # Ensure that fixed parameters do not vary
        for key, value in fixed_parameters.items():
            parameters[key].vary = False

        self.fit_result = self.model.fit(ydata, params=parameters,
                                         weights=weights,
                                         **{self.sweep_parameter: xvals})

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

    def add_to_plot(self, ax):
        self.plot_handle = ax.add(self.fit_result.best_fit,
                                  x=next(iter(self.fit_result.userkws.values())),
                                  **self.plot_kwargs)
        return self.plot_handle


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
        return amplitude * np.exp(-t/tau) + offset

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
        super().__init__()

        if initial_parameters is None:
            initial_parameters={}

        parameters=Parameters()
        if not 'tau' in initial_parameters:
            nearest_idx = np.abs(ydata - ydata[0] / np.exp(1)).argmin()
            initial_parameters['tau'] = -(xvals[1] - xvals[np.where(
                nearest_idx == ydata)[0][0]])
        if not 'amplitude' in initial_parameters:
            initial_parameters['amplitude'] = ydata[1]
        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[-1]

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

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
        super().__init__()

        parameters = Parameters()
        if 'amplitude' not in initial_parameters:
            initial_parameters['amplitude'] = (max(ydata) - min(ydata)) / 2

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips)/2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips), dt)[:int(len(
            fft_flips)/2)]
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
        super().__init__()

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

        parameters['exponent_factor'].min = 0

        return parameters


class RabiFrequencyFit(Fit):
    sweep_parameter = 'f'
    @staticmethod
    def fit_function(f, f0, gamma, t):
        Omega = np.sqrt(gamma**2 + (2*np.pi*(f - f0))**2 / 4)
        return gamma**2 / Omega**2 * np.sin(Omega*t)**2

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        super().__init__()

        parameters = Parameters()

        max_idx = np.argmax(ydata)
        max_frequency = xvals[max_idx]

        if 'f0' not in initial_parameters:
            initial_parameters['f0'] = max_frequency

        if 'gamma' not in initial_parameters:
            if 't' in initial_parameters:
                initial_parameters['gamma'] = np.pi / initial_parameters['t'] / 2
            else:
                FWHM_min_idx = np.argmax(xvals > max_frequency / 2)
                FWHM_max_idx = len(xvals) - np.argmax((xvals > max_frequency / 2)[::-1]) - 1
                initial_parameters['gamma'] = 2*np.pi * (xvals[FWHM_max_idx] - xvals[FWHM_min_idx]) / 2

        if 't' not in initial_parameters:
            initial_parameters['t'] = np.pi / initial_parameters['gamma'] / 2

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        return parameter
