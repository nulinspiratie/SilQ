import numpy as np
from lmfit import Parameters, report_fit, Model

class Fit():
    def __init__(self):
        self.model = Model(self.fit_function)

    def fit_function(self, ydata, xvals):
        pass

    def find_initial_parameters(self):
        pass

    def perform_fit (self):
        pass

    def find_nearest_index(self, array, value):
        idx = np.abs(array - value).argmin()
        return array[idx]

    def find_nearest_value(self, array, value):
        return array[self.find_nearest_index(array, value)]

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

    def perform_fit(self, xvals, ydata, initial_parameters=None, weights=None, options=None):
        super().__init__()

        parameters=self.find_initial_parameters(xvals, ydata, initial_parameters)

        return self.model.fit(ydata, t=xvals, params=parameters, weights=weights)
