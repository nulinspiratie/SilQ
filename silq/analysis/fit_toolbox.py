import numpy as np
from lmfit import Parameters, report_fit, Model

__all__ = ['Fit', 'ExponentialFit']

class Fit(Model):
    def __init__(self):
        super().__init__(self.fit_function)
        self.parameters = self.make_params()

    def find_nearest_index(self, array, value):
        return np.abs(array - value).argmin()

    def find_nearest_value(self, array, value):
        return array[self.find_nearest_index(array, value)]

class ExponentialFit(Fit):
    def __init__(self, sweep_vals, data):
        super().__init__()
        self.sweep_vals = sweep_vals
        self.data = data

    @staticmethod
    def fit_function(t,  tau, amplitude, offset):
        return amplitude * np.exp(-t/tau) + offset

    def find_initial_parameters(self):
        # Tau is approximated as first value reaching 1/e of original value
        decay_idx = self.find_nearest_index(self.data, self.data[0] / np.exp(1))
        self.parameters['tau'].value = self.sweep_vals[decay_idx]

        # Amplitude is approximately difference between max and min
        self.parameters['amplitude'].value = max(self.data) - min(self.data)

        # Offset is approximated by minimum value
        self.parameters['offset'].value = min(self.data)

        return self.parameters
