from typing import Union, Tuple, List, Optional
import numpy as np
from lmfit import Parameters, Model
from lmfit.model import ModelResult

from matplotlib import pyplot as plt
from matplotlib.axis import Axis
import matplotlib.cbook as cbook
import matplotlib.lines as mlines

import logging
from qcodes.data.data_array import DataArray

logger = logging.getLogger(__name__)


class Fit():
    """Base fitting class.

    To fit a specific function, this class should be subclassed.

    To fit data to a function, use the method `Fit.perform_fit`.
    This will find its initial parameters via `Fit.find_initial_parameters`,
    after which it will fit the data to `Fit.fit_function`.

    If both xvals and ydata are passed, or only ydata is passed but it is a
    Qcodes DataArray, the fit is automatically performed.
    Otherwise Fit.perform_fit must be called.

    Args:
        ydata: Data values to be fitted
        xvals: Sweep values, automatically extracted from ydata if not
        explicitly provided and ydata is a DataArray.
        fit: Automatically perform a fit if ydata is passed
        print: Print results
        plot: Axis on which to  the fit
        initial_parameters: Dict plot of initial guesses for fit parameters
        fixed_parameters: Dict of fixed values for fit parameters
        parameter_constraints: Parameter constraints
            e.g. {'frequency' : {'min' : 0}}
        **kwargs: Any other kwargs passed to Fit.perform_fit

    Note:
        - The fitting routine uses lmfit, a wrapper package around scipy.optimize.
        - Fitted parameters can be accessed via ``fit['{parameter_name}']``
        - The fit function can be evaluated with the fitted parameter values
          using ``fit({sweep_values})``
    """
    default_plot_kwargs = {'linestyle': '--', 'color': 'k', 'linewidth': 2}
    sweep_parameter = None

    def __init__(
            self,
            ydata: Union[DataArray, np.ndarray] = None,
            *,
            xvals: Optional[Union[DataArray, np.ndarray]] = None,
            fit: bool = True,
            print: bool = False,
            plot: Optional[Axis] = None,
            initial_parameters: Optional[dict] = None,
            fixed_parameters: Optional[dict] = None,
            parameter_constraints: Optional[dict] = None,
            **kwargs
    ):
        self.model = Model(self.fit_function, **kwargs)
        self.fit_result = None

        self.plot_handle = None

        self.xvals = None
        self.ydata = None
        self.weights = None
        self.parameters = None

        if ydata is not None:
            assert ydata.ndim == 1

            if xvals is None:
                assert isinstance(ydata, DataArray), 'Please provide xvals'
                xvals = ydata.set_arrays[0]

            self.get_parameters(
                xvals=xvals,
                ydata=ydata,
                initial_parameters=initial_parameters,
                fixed_parameters=fixed_parameters,
                parameter_constraints=parameter_constraints
            )
            if fit:
                self.perform_fit(print=print, plot=plot, **kwargs)

    def __getitem__(self, item) -> float:
        """Retrieve fitted parameter value"""
        if self.fit_result is not None:
            return self.fit_result.best_values[item]
        else:
            raise RuntimeError("No fit result to get parameters from.")

    def __call__(
            self, sweep_vals: Union[float, np.ndarray] = None, **kwargs
    ) -> Union[float, np.ndarray]:
        """Evaluate fit for sweep values and optionally override fitted parameters

        The fitted parameter values are used by default

        Args:
            sweep_vals: Value(s) to sweep. Can be either a number or array of
                numbers. If not provided, sweep values should be added as kwarg
            **kwargs: Optional parameter values to override the fitted values

        Returns:
             Value or array of values corresponding to evaluated fit function
        """

        params = kwargs
        if sweep_vals is not None:
            params[self.sweep_parameter] = sweep_vals
        return self.fit_result.eval(**params)

    def _ipython_key_completions_(self):
        """Tab completion for IPython, i.e. fit["{parameter_name}"] """
        try:
            return list(self.fit_result.best_values)
        except Exception:
            return []

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

    def get_parameters(self, xvals, ydata, initial_parameters=None,
                       fixed_parameters=None, parameter_constraints=None,
                       weights=None, **kwargs):
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
        initial_parameters = initial_parameters or {}
        fixed_parameters = fixed_parameters or {}
        parameter_constraints = parameter_constraints or {}

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
            if kwargs and self.parameters is None:
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
                    xscale: float = 1,
                    yscale: float = 1,
                    **kwargs):
        """Add fit to existing plot axis

        Args:
            ax: Axis to add plot to
            N: number of points to use as x values (to smoothe fit curve)
            xrange: Optional range for x values (min, max)
            xscale: value to multiple x values by to rescale axis
            yscale: value to multiple y values by to rescale axis
            kwargs: Additional plot kwargs. By default Fit.plot_kwargs are used

        Returns:
            plot_handle of fit curve
        """
        if xrange is None:
            x_vals = self.xvals
            x_vals_full = np.linspace(min(x_vals), max(x_vals), N)
        else:
            x_vals_full = np.linspace(*xrange, N)

        y_vals_full = self.fit_result.eval(
            **{self.sweep_parameter: x_vals_full})
        x_vals_full *= xscale
        y_vals_full *= yscale

        # Set default plot kwargs while de-aliasing (e.g. 'lw' -> 'linewidth')
        # kwargs to prevent duplicate keys
        kwargs = {**self.default_plot_kwargs,
                  **cbook.normalize_kwargs(kwargs, mlines.Line2D)}
        self.plot_handle, = ax.plot(
            x_vals_full, y_vals_full, **kwargs
        )
        return self.plot_handle


class LinearFit(Fit):
    """Fitting class for a linear function.

        To fit data to a function, use the method `Fit.perform_fit`.
        This will find its initial parameters via `Fit.find_initial_parameters`,
        after which it will fit the data to `Fit.fit_function`.

        Note:
            The fitting routine uses lmfit, a wrapper package around
            scipy.optimize.
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
        """Lorentzian function using x as x-coordinate

        Args:
            x: x-values
            mean: mean
            amplitude: amplitude
            gamma: standard deviation
            offset: offset

        Returns:
            lorentzian data points
        """
        return amplitude * (gamma / 2) / (
                (x - mean) ** 2 + (gamma / 2) ** 2) + offset

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
            # Attempt to find the FWHM of the Gaussian to lessening degrees
            # of accuracy
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
                    initial_parameters['gamma'] = (max(xvals) - min(xvals)) / 20

        if not 'mean' in initial_parameters:
            initial_parameters['mean'] = xvals[np.argmax(ydata)]

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = np.mean(ydata)

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['gamma'].min = 0

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
            # Attempt to find the FWHM of the Gaussian to lessening degrees
            # of accuracy
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
            # Attempt to find the FWHM of the Gaussian to lessening degrees
            # of accuracy
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
            x: x-values.
            mean: mean
            amplitude: amplitude
            sigma: standard deviation
            offset: offset

        Returns:
            gaussian data points
        """
        return amplitude * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2)) + offset

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
            initial_parameters['amplitude'] = np.max(ydata)
        if not 'mean' in initial_parameters:
            initial_parameters['mean'] = xvals[np.argmax(ydata)]
        if not 'sigma' in initial_parameters:
            # Attempt to find the FWHM of the Gaussian to lessening degrees
            # of accuracy
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
                    initial_parameters['sigma'] = (max(xvals) - min(xvals)) / 20
        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = np.mean(ydata)

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['sigma'].min = 0

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
                     exponent_factor: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            t: Time.
            tau: Decay constant.
            amplitude:
            exponent_factor:
            offset:

        Returns:
            exponential data points
        """
        return amplitude * np.exp(-np.power(t / tau, exponent_factor)) + offset

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
        if not 'exponent_factor' in initial_parameters:
            initial_parameters['exponent_factor'] = 1
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
        initial_parameters.setdefault('amplitude', (max(ydata) - min(ydata)) / 2)

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips) / 2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips), dt)[:int(len(fft_flips) / 2)]
        frequency_idx = np.argmax(fft_flips_abs[1:]) + 1

        initial_parameters.setdefault('frequency', fft_freqs[frequency_idx])

        if plot:
            plt.figure()
            plt.plot(fft_freqs, fft_flips_abs, 'o')
            plt.plot(initial_parameters['frequency'], fft_flips_abs[frequency_idx], 'o', ms=8)

        initial_parameters.setdefault('phase', np.pi / 2 + np.angle(fft_flips[frequency_idx]))
        initial_parameters.setdefault('offset', (max(ydata) + min(ydata)) / 2)

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
        initial_parameters.setdefault('amplitude', (max(ydata) - min(ydata)) / 2)

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips) / 2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips), dt)[:int(len(
            fft_flips) / 2)]
        frequency_idx = np.argmax(fft_flips_abs[1:]) + 1

        initial_parameters.setdefault('frequency', fft_freqs[frequency_idx])
        if plot:
            plt.figure()
            plt.plot(fft_freqs, fft_flips_abs, 'o')
            plt.plot(fft_freqs[frequency_idx], fft_flips_abs[frequency_idx], 'o', ms=8)

        initial_parameters.setdefault('phase', np.pi / 2 + np.angle(fft_flips[frequency_idx]))

        initial_parameters.setdefault('offset', (max(ydata) + min(ydata)) / 2)

        initial_parameters.setdefault('amplitude_AM', 0.1)

        initial_parameters.setdefault('frequency_AM', fft_freqs[frequency_idx])

        if plot:
            plt.figure()
            plt.plot(fft_freqs, fft_flips_abs, 'o')
            plt.plot(fft_freqs[frequency_idx], fft_flips_abs[frequency_idx], 'o', ms=8)

        initial_parameters.setdefault('phase_AM', 0)

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
        initial_parameters.setdefault('amplitude', (max(ydata) - min(ydata)) / 2)

        initial_parameters.setdefault('tau', xvals[-1] / 2)

        dt = (xvals[1] - xvals[0])
        fft_flips = np.fft.fft(ydata)
        fft_flips_abs = np.abs(fft_flips)[:int(len(fft_flips) / 2)]
        fft_freqs = np.fft.fftfreq(len(fft_flips), dt)[:int(len(fft_flips) / 2)]
        frequency_idx = np.argmax(fft_flips_abs[1:]) + 1

        initial_parameters.setdefault('frequency', fft_freqs[frequency_idx])

        if plot:
            plt.figure()
            plt.plot(fft_freqs, fft_flips_abs)
            plt.plot(fft_freqs[frequency_idx], fft_flips_abs[frequency_idx], 'o', ms=8)

        initial_parameters.setdefault('phase', np.pi / 2 + np.angle(fft_flips[frequency_idx]))

        initial_parameters.setdefault('offset', (max(ydata) + min(ydata)) / 2)

        initial_parameters.setdefault('exponent_factor', 1)

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['tau'].min = 0
        parameters['exponent_factor'].min = 0
        parameters['frequency'].min = 0

        return parameters


class RabiFrequencyFit(Fit):
    sweep_parameter = 'f'

    @staticmethod
    def fit_function(f, f0, gamma, t, offset, amplitude):
        Omega = np.sqrt(gamma ** 2 + (2 * np.pi * (f - f0)) ** 2 / 4)
        return amplitude * gamma ** 2 / Omega ** 2 * np.sin(Omega * t) ** 2 + offset

    def find_initial_parameters(self, xvals, ydata, initial_parameters={},
                                plot=False):
        parameters = Parameters()

        max_idx = np.argmax(ydata)
        max_frequency = xvals[max_idx]

        initial_parameters.setdefault('f0', max_frequency)

        if 'gamma' not in initial_parameters:
            if 't' in initial_parameters:
                initial_parameters['gamma'] = np.pi / initial_parameters['t'] / 2
            else:
                FWHM_min_idx = np.argmax(xvals > max_frequency / 2)
                FWHM_max_idx = len(xvals) - np.argmax(
                    (xvals > max_frequency / 2)[::-1]) - 1
                initial_parameters['gamma'] = 2 * np.pi * (
                        xvals[FWHM_max_idx] - xvals[FWHM_min_idx]) / 2

        initial_parameters.setdefault('t', np.pi / initial_parameters['gamma'] / 2)
        initial_parameters.setdefault('offset', np.min(ydata))
        initial_parameters.setdefault('amplitude', 1 - initial_parameters['offset'])

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


class FermiFit(Fit):
    """Fitting class for a Fermi-Dirac distribution function.

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
                     A: float,
                     U: float,
                     T: float,
                     offset: float) -> Union[float, np.ndarray]:
        """Exponential function using time as x-coordinate

        Args:
            V: electric potential being swept.
            A: rescaling factor for function
            U: Fermi level
            T: System temperature (K)
            offset:

        Returns:
            exponential data points
        """

        return A / (np.exp((V - U) / (FermiFit.kB * T)) + 1) + offset

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

        if not 'A' in initial_parameters:
            initial_parameters['A'] = max(ydata) - min(ydata)

        if not 'offset' in initial_parameters:
            initial_parameters['offset'] = np.min(ydata)

        if not 'U' in initial_parameters:
            initial_parameters['U'] = xvals[0]

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['T'].min = 0

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
            alpha : electrochemical potential lever arm from applied gate
            potential
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


class T1Fit(Fit):
    """Fitting class for a 1/T1 vs magnetic field (B) function.

        To fit data to a function, use the method `Fit.perform_fit`.
        This will find its initial parameters via `Fit.find_initial_parameters`,
        after which it will fit the data to `Fit.fit_function`.

        Note:
            The fitting routine uses lmfit, a wrapper package around
            scipy.optimize.
        """
    sweep_parameter = 'x'

    @staticmethod
    def fit_function(x: Union[float, np.ndarray],
                     K0: float,
                     K1: float,
                     K3: float,
                     K5: float,
                     K7: float) -> Union[float, np.ndarray]:
        """T1 function using B-field as x-coordinate

        Args:
            x: field.
            K0:
            K1:
            K5:
            K7:

        Returns:
            magic
        """
        return K0 + K1 * x + K3 * x ** 3 + K5 * x ** 5 + K7 * x ** 7

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
        if not 'K0' in initial_parameters:
            initial_parameters['K5'] = ydata[0]
        if not 'K1' in initial_parameters:
            initial_parameters['K1'] = ydata[-1] - ydata[0]
        if not 'K3' in initial_parameters:
            initial_parameters['K3'] = 0.01
        if not 'K5' in initial_parameters:
            initial_parameters['K5'] = 0.01
        if not 'K7' in initial_parameters:
            initial_parameters['K7'] = 0.01

        for key in initial_parameters:
            parameters.add(key, initial_parameters[key])

        parameters['K0'].min = 0
        parameters['K1'].min = 0
        parameters['K3'].min = 0
        parameters['K5'].min = 0
        parameters['K7'].min = 0

        return parameters
