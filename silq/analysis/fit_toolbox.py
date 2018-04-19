import numpy as np
from lmfit import Parameters, Model
from matplotlib import pyplot as plt
import logging
from qcodes.data.data_array import DataArray

__all__ = ['Fit', 'ExponentialFit', 'SineFit', 'ExponentialSineFit',
           'RabiFrequencyFit']

logger = logging.getLogger(__name__)


class Fit():
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

    def find_initial_parameters(self, xvals, ydata, initial_parameters):
        pass

    def perform_fit(self):
        pass

    def find_nearest_index(self, array, value):
        idx = np.abs(array - value).argmin()
        return array[idx]

    def get_parameters(self, xvals, ydata, initial_parameters={},
                       fixed_parameters={}, weights=None):
        if isinstance(xvals, DataArray):
            xvals = xvals.ndarray
        if isinstance(ydata, DataArray):
            ydata = ydata.ndarray
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

    def find_nearest_value(self, array, value):
        return array[self.find_nearest_index(array, value)]

    def perform_fit(self, parameters=None, print=False, plot=None, **kwargs):
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

    def add_to_plot(self, ax, **kwargs):
        x_vals = next(iter(self.fit_result.userkws.values()))
        x_vals_full = np.linspace(min(x_vals), max(x_vals), 201)
        y_vals_full = self.fit_result.eval(
            **{self.sweep_parameter: x_vals_full})
        plot_kwargs = {**self.plot_kwargs, **kwargs}
        self.plot_handle = ax.add(y_vals_full,
                                  x=x_vals_full,
                                  **plot_kwargs)
        return self.plot_handle


class ExponentialFit(Fit):
    sweep_parameter = 't'

    @staticmethod
    def fit_function(t, tau, amplitude, offset):
        return amplitude * np.exp(-t / tau) + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters):
        super().find_initial_parameters(xvals, ydata, initial_parameters)

        if initial_parameters is None:
            initial_parameters = {}

        parameters = Parameters()
        if not 'tau' in initial_parameters:
            initial_parameters['tau'] = -(xvals[1] - xvals[np.where(
                self.find_nearest_index(ydata, ydata[0] / np.exp(1)) == ydata)[
                0][0]])
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
    def fit_function(t, amplitude, frequency, phase, offset):
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        super().find_initial_parameters(xvals, ydata, initial_parameters)

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
        super().find_initial_parameters(xvals, ydata, initial_parameters)

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
        Omega = np.sqrt(gamma ** 2 + (2 * np.pi * (f - f0)) ** 2 / 4)
        return gamma ** 2 / Omega ** 2 * np.sin(Omega * t) ** 2

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        super().find_initial_parameters(xvals, ydata, initial_parameters)

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

        return parameters