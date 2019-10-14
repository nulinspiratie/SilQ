from typing import Union, Tuple, List
import numpy as np
from lmfit import Parameters, Model
from lmfit.model import ModelResult
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter1d
import logging
from qcodes.data.data_array import DataArray

__all__ = ['Fit',
           'LinearFit',
           'MultiLinearFit',
           'LorentzianFit',
           'GaussianFit',
           'SumGaussianFit',
           'ExponentialFit',
           'SineFit',
           'VoigtFit',
           'ExponentialSineFit',
           'DoubleExponentialFit',
           'SumExponentialFit',
           'RabiFrequencyFit',
           'AMSineFit',
           'BayesianUpdateFit',
           'DoubleFermiFit']

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
        self.model = Model(self.fit_function, **kwargs)
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
                       fixed_parameters={}, parameter_constraints={}, weights=None, **kwargs):
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
            parameter_constraints: {parameter: {constraint : value, ...},  ...}
                combination to further constrain existing parameters. e.g.
                {'frequency' : {'min' : 0}} ensures only positive frequencies
                can be fit.
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

        # Apply all constraints to parameters
        for key, constraints_dict in parameter_constraints.items():
            for opt, val in constraints_dict.items():
                setattr(self.parameters[key], opt, val)

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
    """Fitting class for a linear function.

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

class MultiLinearFit(Fit):
    """Fitting class for a sum of linear functions.

        To fit data to a function, use the method `Fit.perform_fit`.
        This will find its initial parameters via `Fit.find_initial_parameters`,
        after which it will fit the data to `Fit.fit_function`.

        Note:
            The fitting routine uses lmfit, a wrapper package around scipy.optimize.
        """
    sweep_parameter = 't'

    def __init__(self,  max_lines=1, **kwargs):
        self.max_lines = max_lines
        super().__init__(self, **kwargs)

    @staticmethod
    def fit_function(t: Union[float, np.ndarray],
                     *args, **kwargs) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            gradient:
            offset:

        Returns:
            linear data points
        """

        gradients = [val for key, val in kwargs.items() if key.startswith('gradient')]
        offset = kwargs['offset']

        func = offset
        for gradient in gradients:
            func += gradient * t
        return func

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

        max_err = 0
        for param_name, param in self.fit_result.params.items():
            if not np.isfinite(param.stderr):
                max_err = param.stderr
            elif param.value == 0:
                continue
            elif abs(param.stderr / param.value) > max_err:
                max_err = abs(param.stderr / param.value)

        if max_err > 2 or not np.isfinite(max_err): # 200% std.dev in parameter
            idx = param_name[-1]
            logger.warning(f'Parameter {param_name} has std. dev. of {max_err*100:.0f}%. '
                           f'Removing final gradient and redoing fit.')
            parameters.pop(f'gradient_{idx}')
            self.perform_fit(parameters, print=False, plot=None, **kwargs)

        if print:
            self.print_results()

        if plot is not None:
            self.add_to_plot(plot)

        return self.fit_result

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

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[0]

        for k in range(self.max_lines):
            if not f'gradient_{k}' in initial_parameters:
                initial_parameters[f'gradient_{k}'] = (ydata[-1] - ydata[0]) / \
                                            (xvals[-1] - xvals[0])/self.max_lines

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        return parameters

class LorentzianFit(Fit):
    """Fitting class for a Lorentzian function.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 'x'

    @staticmethod
    def fit_function(x: Union[float, np.ndarray],
                     amplitude: float,
                     mean: float,
                     gamma: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Gaussian function using x as x-coordinate

        Args:
            x: independent variable
            mean: mean
            amplitude:
            sigma: standard deviation

        Returns:
            exponential data points
        """
        return amplitude * (gamma**2 /((x - mean)**2 + gamma**2)) + offset

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
            initial_parameters['amplitude'] = max(ydata)
        if not 'gamma' in initial_parameters:
            # Attempt to find the FWHM of the Gaussian to lessening degrees of accuracy
            try:
                A = initial_parameters['amplitude']
                for ratio in [1/100, 1/33, 1/10, 1/3]:
                    idxs, = np.where(abs(ydata - A/2) <= A*ratio)
                    if len(idxs) >= 2:
                        initial_parameters['gamma'] = \
                            1 / (2 * np.sqrt(2 * np.log(2))) * (xvals[idxs[-1]] - xvals[idxs[0]])
                        break
            finally:
                if not 'gamma' in initial_parameters:
                    # 5% of the x-axis
                    initial_parameters['gamma'] = (max(xvals)-min(xvals))/20

        if not 'mean' in initial_parameters:
            max_idx = np.argmax(ydata)
            initial_parameters['mean'] = xvals[max_idx]

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = np.mean(ydata)

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

#         parameters['tau'].min = 0

        return parameters

class VoigtFit(Fit):
    """Fitting class for a Voigt function.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 'x'

    @staticmethod
    def fit_function(x: Union[float, np.ndarray],
                     amplitude: float,
                     mean: float,
                     gamma: float,
                     sigma : float,
                     offset: float) -> Union[float, np.ndarray]:
        """Gaussian function using x as x-coordinate

        Args:
            x: independent variable
            mean: mean
            amplitude:
            sigma: standard deviation

        Returns:
            exponential data points
        """
        return amplitude * (
                    gamma ** 2 / ((x - mean) ** 2 + gamma ** 2)) * \
               np.exp(- (x - mean) ** 2 / (2 * sigma) ** 2) + offset

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
            initial_parameters['amplitude'] = max(ydata)
        if not 'gamma' in initial_parameters:
            # Attempt to find the FWHM of the Gaussian to lessening degrees of accuracy
            try:
                A = initial_parameters['amplitude']
                for ratio in [1 / 100, 1 / 33, 1 / 10, 1 / 3]:
                    idxs, = np.where(abs(ydata - A / 2) <= A * ratio)
                    if len(idxs) >= 2:
                        initial_parameters['gamma'] = \
                            1 / (2 * np.sqrt(2 * np.log(2))) * (
                                        xvals[idxs[-1]] - xvals[idxs[0]])
                        break
            finally:
                if not 'gamma' in initial_parameters:
                    # 5% of the x-axis
                    initial_parameters['gamma'] = (max(xvals) - min(
                        xvals)) / 20
        if not 'sigma' in initial_parameters:
            # Attempt to find the FWHM of the Gaussian to lessening degrees of accuracy
            try:
                A = initial_parameters['amplitude']
                for ratio in [1 / 100, 1 / 33, 1 / 10, 1 / 3]:
                    idxs, = np.where(abs(ydata - A / 2) <= A * ratio)
                    if len(idxs) >= 2:
                        initial_parameters['sigma'] = \
                            1 / (2 * np.sqrt(2 * np.log(2))) * (
                                        xvals[idxs[-1]] - xvals[idxs[0]])
                        break
            finally:
                if not 'sigma' in initial_parameters:
                    # 5% of the x-axis
                    initial_parameters['sigma'] = (max(xvals) - min(
                        xvals)) / 20

        if not 'mean' in initial_parameters:
            max_idx = np.argmax(ydata)
            initial_parameters['mean'] = xvals[max_idx]

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = np.mean(ydata)

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        #         parameters['tau'].min = 0

        return parameters

class GaussianFit(Fit):
    """Fitting class for a Gaussian function.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 'x'

    @staticmethod
    def fit_function(x: Union[float, np.ndarray],
                     amplitude: float,
                     mean: float,
                     sigma: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Gaussian function using x as x-coordinate

        Args:
            x: independent variable
            mean: mean
            amplitude:
            sigma: standard deviation

        Returns:
            exponential data points
        """
        return amplitude * np.exp(- (x - mean)**2 / (2*sigma)**2) + offset

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
            initial_parameters['amplitude'] = max(ydata)
        if not 'sigma' in initial_parameters:
            # Attempt to find the FWHM of the Gaussian to lessening degrees of accuracy
            try:
                A = initial_parameters['amplitude']
                for ratio in [1/100, 1/33, 1/10, 1/3]:
                    idxs, = np.where(abs(ydata - A/2) <= A*ratio)
                    if len(idxs) >= 2:
                        initial_parameters['sigma'] = \
                            1 / (2 * np.sqrt(2 * np.log(2))) * (xvals[idxs[-1]] - xvals[idxs[0]])
                        break
            finally:
                if not 'sigma' in initial_parameters:
                    # 5% of the x-axis
                    initial_parameters['sigma'] = (max(xvals)-min(xvals))/20

        if not 'mean' in initial_parameters:
            max_idx = np.argmax(ydata)
            initial_parameters['mean'] = xvals[max_idx]

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = np.mean(ydata)

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

#         parameters['tau'].min = 0

        return parameters

class SumGaussianFit(Fit):
    """Fitting class for a sum of Gaussian functions.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 'x'

    def __init__(self, max_gaussians=5, min_gaussians=1, smooth=False, **kwargs):
        self.max_gaussians = max_gaussians
        self.min_gaussians = min_gaussians
        self.smooth = smooth
        super().__init__(self, **kwargs)

    @staticmethod
    def fit_function(x: Union[float, np.ndarray],
                     *args, **kwargs
                     ) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            tau: Decay constant.
            amplitude:

        Returns:
            exponential data points
        """
        means = [val for key, val in kwargs.items() if key.startswith('mean')]
        sigmas = [val for key, val in kwargs.items() if key.startswith('sigma')]
        amplitudes = [val for key, val in kwargs.items() if key.startswith('A')]

        func = 0
        for mean, sigma, amplitude in zip(means, sigmas, amplitudes):
            func += amplitude * np.exp(- (x - mean) ** 2 / (2 * sigma) ** 2)
        return func

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

        max_err = 0
        for param_name, param in self.fit_result.params.items():
            if not np.isfinite(param.stderr):
                max_err = param.stderr
            elif param.value == 0:
                continue
            elif abs(param.stderr / param.value) > max_err:
                max_err = abs(param.stderr / param.value)

        if max_err > 2 or not np.isfinite(max_err):  # 200% std.dev in parameter
            idx = int(param_name.split('_')[-1])
            # idx = max([p[-1] for ])
            if (idx + 1 > self.min_gaussians):
                logger.warning(
                    f'Parameter {param_name} has std. dev. of {max_err*100:.0f}%. '
                    f'Removing final parameter idx and redoing fit.')
                parameters.pop(f'sigma_{idx}')
                parameters.pop(f'mean_{idx}')
                parameters.pop(f'A_{idx}')
                self.perform_fit(parameters, print=False, plot=None, **kwargs)
            else:
                # There is only 1 exponential, finish fitting
                pass

        if print:
            self.print_results()

        if plot is not None:
            self.add_to_plot(plot)

        return self.fit_result

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

        peak_idxs = None
        if self.smooth:
            ydata = gaussian_filter1d(ydata, 2)
        try:
            for ratio in [9 / 10, 2 / 3,  1 / 3, 1 / 10, 1 / 33, 1 / 100]:
                peak_idxs, _ = find_peaks(ydata, height=(max(ydata) - min(
                    ydata)) * ratio + min(ydata))
                if len(peak_idxs) >= self.max_gaussians:
                    peak_idxs = peak_idxs[:self.max_gaussians]
                    break
        finally:
            if peak_idxs is None:
                # linearly spaced points offset from 0
                peak_idxs = np.linspace(0, len(xvals), num=self.max_gaussians,
                                        dtype='int')

            if len(peak_idxs) < self.max_gaussians:
                peak_idxs = np.append(peak_idxs, [peak_idxs[-1]]*(self.max_gaussians - len(peak_idxs)))

        for k in range(self.max_gaussians):
            if not f'A_{k}' in initial_parameters:
                # initial_parameters[f'A_{k}'] = (ydata[1] - ydata[
                #     -1]) / self.max_gaussians
                initial_parameters[f'A_{k}'] = ydata[peak_idxs[k]]
            if not f'mean_{k}' in initial_parameters:
                initial_parameters[f'mean_{k}'] = xvals[peak_idxs[k]]
            if not f'sigma_{k}' in initial_parameters:
                # Attempt to find the FWHM of the Gaussian to lessening degrees of accuracy
                # Assuming the Gaussian is symmetric, I find the
                sigma = None
                try:
                    A = initial_parameters[f'A_{k}']
                    for ratio in [1 / 100, 1 / 33, 1 / 10, 1 / 3]:
                        idxs, = np.where(
                            abs(ydata[peak_idxs[k]:] - A / 2) <= A * ratio)
                        if len(idxs) >= 1:
                            sigma = 1 / (np.sqrt(2 * np.log(2))) * (
                                        xvals[idxs[0] + peak_idxs[k]] - xvals[
                                    peak_idxs[k]])
                            break
                finally:
                    if sigma is None:
                        # 5% of the x-axis
                        initial_parameters[f'sigma_{k}'] = (max(xvals) - min(
                            xvals)) / 20
                    else:
                        initial_parameters[f'sigma_{k}'] = sigma

                peak_idxs[k] + np.argmax(
                    abs(ydata[peak_idxs[k]:]) <= (max(ydata) - min(ydata)) / 10)
                # initial_parameters[f'sigma_{k}'] = (max(xvals) - min(
                #     xvals)) / 20

        self.peak_idxs = peak_idxs
        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])
            if key.startswith('sigma'):
                parameters[key].min = 0

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
    def __init__(self, *args, **kwargs):
        logger.warning("DoubleExponentialFit is deprecated, use SumExponentialFit.")
        super().__init__(*args, **kwargs)


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

class SumExponentialFit(Fit):
    """Fitting class for a sum of exponential functions.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 't'

    def __init__(self,  max_exponentials=5, min_exponentials=1, **kwargs):
        self.max_exponentials = max_exponentials
        self.min_exponentials = min_exponentials
        super().__init__(self, **kwargs)

    @staticmethod
    def fit_function(t: Union[float, np.ndarray],
                     *args, **kwargs
                     ) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            tau: Decay constant.
            amplitude:
            offset:

        Returns:
            exponential data points
        """
        taus = [val for key, val in kwargs.items() if key.startswith('tau')]
        amplitudes = [val for key, val in kwargs.items() if key.startswith('A')]
        offset = kwargs['offset']

        func = offset
        for tau, amplitude in zip(taus, amplitudes):
            func += amplitude * np.exp(-t / tau)
        return func

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

        max_err = 0
        for param_name, param in self.fit_result.params.items():
            if not np.isfinite(param.stderr):
                max_err = param.stderr
            elif param.value == 0:
                continue
            elif abs(param.stderr / param.value) > max_err:
                max_err = abs(param.stderr / param.value)

        if max_err > 2 or not np.isfinite(max_err) or max_err == 0: # 200% std.dev in parameter
            idx = int(param_name.split('_')[-1])
            # idx = param_name[-1]
            # idx = max([p[-1] for ])
            if (idx + 1 > self.min_exponentials):
                logger.warning(f'Parameter {param_name} has std. dev. of {max_err*100:.0f}%. '
                               f'Removing final tau and A and redoing fit.')
                parameters.pop(f'tau_{idx}')
                parameters.pop(f'A_{idx}')
                self.perform_fit(parameters, print=False, plot=None, **kwargs)
            else:
                # There is only 1 exponential, finish fitting
                pass

        if print:
            self.print_results()

        if plot is not None:
            self.add_to_plot(plot)

        return self.fit_result

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

        if not f'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[-1]

        for k in range(self.max_exponentials):
            if not f'A_{k}' in initial_parameters:
                initial_parameters[f'A_{k}'] = (ydata[1] - ydata[-1])/self.max_exponentials
            if not f'tau_{k}' in initial_parameters:
                exponent_val = (initial_parameters['offset']
                                + initial_parameters[f'A_{k}'] / np.exp(1))
                nearest_idx = np.abs(ydata - exponent_val).argmin()
                initial_parameters[f'tau_{k}'] = -(xvals[1] - xvals[nearest_idx])

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])
            if key.startswith('tau'):
                parameters[key].min = 0

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
        return amplitude_AM * np.sin(
            2 * np.pi * frequency_AM * t + phase_AM) * \
               amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

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

        # parameters['amplitude'].set(exp r='(1-offset)/(1+amplitude_AM)')
        parameters['amplitude'].set(min=0, max=1)
        parameters['amplitude_AM'].set(min=0, max=1)

        parameters['frequency'].set(min=0, max=np.Inf)
        parameters['frequency_AM'].set(min=0, max=np.Inf)

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
        parameters['frequency'].min = 0

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

        parameters.add('f_Rabi', expr='gamma/pi')
        # parameters.add('amplitude', expr='gamma^2/ Omega^2')

        return parameters


class BayesianUpdateFit(Fit):
    """Fitting class for a 'Bayesian update' function.

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
                     tau_2: float,
                     prior: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            prior:
            tau_down_out:
            tau_up_out:
            amplitude:
            offset:

        Returns:
            exponential data points
        """
        return prior * np.exp(-t / tau_1) / (
                np.exp(-t / tau_1) * prior + np.exp(-t / tau_2) * (
                        1 - prior)) + offset

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
        if not 'prior' in initial_parameters:
            initial_parameters['prior'] = ydata[1] - ydata[-1]
        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = ydata[-1]

        if not 'tau_1' in initial_parameters:
            exponent_val = (initial_parameters['offset']
                            + initial_parameters['prior'] / np.exp(1))
            nearest_idx = np.abs(ydata - exponent_val).argmin()
            initial_parameters['tau_1'] = -(xvals[1] - xvals[nearest_idx])

        if not 'tau_2' in initial_parameters:
            initial_parameters['tau_2'] = 0.5 * initial_parameters['tau_1']

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['tau_1'].min = 0
        parameters['tau_2'].min = 0
        return parameters


class DoubleFermiFit(Fit):
    """Fitting class for a double Fermi-Dirac distribution function.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    Note:
        The fitting routine uses lmfit, a wrapper package around scipy.optimize.
    """
    sweep_parameter = 'V'
    kB = 8.61733034e-5  # Boltzmann constant in eV

    @staticmethod
    def fit_function(V: Union[float, np.ndarray],
                     A_1: float,
                     A_2: float,
                     U_1: float,
                     U_2: float,
                     T: float,
                     alpha: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            V: electric potential being swept.
            A: rescaling factor for function
            lower: lower Fermi level (for T=0)
            upper: upper Fermi level (for T=0)
            T: System temperature (K)
            alpha : electrochemical potential lever arm from applied gate potential
            offset:

        Returns:
            exponential data points
        """

        return A_1 * (1 / (np.exp(
            (V - U_1) / (DoubleFermiFit.kB * T / alpha)) + 1) + A_2 / (np.exp(
            (V - U_2) / (DoubleFermiFit.kB * T / alpha)) + 1)) + \
               offset

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
        if not 'T' in initial_parameters:
            initial_parameters['T'] = 100e-3

        if not 'A_1' in initial_parameters:
            initial_parameters['A_1'] = max(ydata) - min(ydata)
        if not 'A_2' in initial_parameters:
            initial_parameters['A_2'] = max(ydata) - min(ydata)

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = np.min(ydata)

        if not 'U_1' in initial_parameters:
            initial_parameters['U_1'] = xvals[0]

        if not 'U_2' in initial_parameters:
            initial_parameters['U_2'] = xvals[1]

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters.add('E', value=initial_parameters.get('E', 165e-6),
                       vary=False)
        parameters.add('alpha', expr='E/(U_2-U_1)')

        parameters['T'].min = 0
        parameters['alpha'].min = 0
        parameters['offset'].min = 0

        return parameters