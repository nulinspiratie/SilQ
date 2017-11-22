import numpy as np
from lmfit import Parameters, Model
from matplotlib import pyplot as plt

__all__ = ['Fit', 'ExponentialFit', 'SineFit']

class Fit():
    def __init__(self):
        self.model = Model(self.fit_function)

    def find_initial_parameters(self):
        pass

    def perform_fit (self):
        pass

    def find_nearest_index(self, array, value):
        idx = np.abs(array - value).argmin()
        return array[idx]

    def find_nearest_value(self, array, value):
        return array[self.find_nearest_index(array, value)]

    def perform_fit(self, xvals, ydata, initial_parameters=None, weights=None):

        parameters=self.find_initial_parameters(xvals, ydata, initial_parameters)

        return self.model.fit(ydata, t=xvals, params=parameters, weights=weights)


class ExponentialFit(Fit):
    def __init__(self):
        super().__init__()

    @staticmethod
    def fit_function(t,  tau, amplitude, offset):
        return amplitude * np.exp(-t/tau) + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters):
        super().__init__()

        if initial_parameters is None:
            initial_parameters={}

        parameters=Parameters()
        if not 'tau' in initial_parameters:
            initial_parameters['tau'] = -(xvals[1] - xvals[np.where(
                self.find_nearest_index(ydata, ydata[0] / np.exp(1)) == ydata)[0][0]])
        if not 'amplitude' in initial_parameters:
            initial_parameters['amplitude'] = ydata[1]
        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[-1]

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        return parameters


class SineFit(Fit):
    @staticmethod
    def fit_function(t, amplitude, frequency, phase, offset):
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters=None,
                                plot=False):
        super().__init__()
        if initial_parameters is None:
            initial_parameters = {}

        parameters = Parameters()
        if 'amplitude' not in initial_parameters:
            initial_parameters['amplitude'] = (max(ydata) - min(ydata)) / 2

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips)/2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips_abs), dt)[:int(len(
            fft_flips)/2)]
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

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        return parameters
