from typing import Union
import numpy as np
from lmfit import Parameters, Model
from lmfit.model import ModelResult
from matplotlib import pyplot as plt

__all__ = ['Fit', 'ExponentialFit', 'SineFit']

class Fit():
    """Base fitting class.
    
    To fit a specific function, this class should be subclassed.
    
    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.
    
    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    def __init__(self):
        self.model = Model(self.fit_function)

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
        nearest_idx = np.abs(array - value).argmin()
        return array[nearest_idx]

    def perform_fit(self,
                    xvals:np.ndarray,
                    ydata: np.ndarray,
                    initial_parameters: dict = None,
                    weights: np.ndarray = None) -> ModelResult:
        """
        
        Args:
            xvals: x-coordinates of data points
            ydata: Data points
            initial_parameters: fixed initial parameters to be skipped
            weights: Weights for data points, must have same shape as ydata

        Returns:
            ModelResult object containing fitting results
        """

        parameters=self.find_initial_parameters(xvals, ydata, initial_parameters)

        return self.model.fit(ydata, t=xvals, params=parameters, weights=weights)


class ExponentialFit(Fit):
    """Fitting class for an exponential function.
    
    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.
    
    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    def __init__(self):
        super().__init__()

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

    def find_initial_parameters(self, xvals, ydata, initial_parameters=None,
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
