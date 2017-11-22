import numpy as np
import warnings
import qcodes as qc
import silq
from qcodes.instrument.sweep_values import SweepValues
__all__ = ['SweepDependentValues', 'create_set_vals']


class SweepDependentValues(SweepValues):
    def __init__(self, parameter, step=None, step_percentage=None,
                 num=None, window=None,
                 center_val=None, min_val=None, max_val=None, reverse=False,
                 fix=True):

        super().__init__(parameter)

        self.step = step
        self.step_percentage = step_percentage
        self.num = num
        self.window = window
        self.center_val = center_val
        self.min_val = min_val
        self.max_val = max_val
        self.reverse = reverse
        self.fix = fix

        self._values = self.create_vals()

    def calculate_step(self, min_step, max_step, percentage, round_vals=True):
        val = min_step * (max_step / min_step) ** (percentage / 100)
        if round_vals:
            # Round values to one decimal after first digit
            digits = int(np.ceil(np.log10(1 / val))) + 1
            val = round(val, digits)
        return val

    def determine_step(self, parameter):
        if self.step_percentage is not None:
            parameter_config = silq.config.get('parameters')
            min_step, max_step = parameter_config[parameter.name]['steps']
            return self.calculate_step(min_step, max_step, self.step_percentage)
        else:
            return self.step


    def create_vals(self):
        if self.window is None:
            if self.min_val is not None and self.max_val is not None:
                self.window = self.max_val - self.min_val
            else:
                step = self.determine_step(self.parameter)
                window = step * (self.num - 1)

        if self.num is None:
            step = self.determine_step(self.parameter)
            num = int(round(window / step))
        else:
            num = self.num

        if self.min_val is None and self.max_val is None:
            if self.center_val is None:
                center_val = self.parameter()
            min_val = center_val - window / 2
            max_val = center_val + window / 2
        else:
            min_val = self.min_val
            max_val = self.max_val

        vals = list(np.linspace(min_val, max_val, num))

        if self.reverse:
            vals.reverse()

        return vals

    def __iter__(self):
        self.idx = 0
        if not self.fix:
            self._values = self.create_vals()
        return self

    def __next__(self):
        if self.idx >= len(self._values):
            raise StopIteration
        else:
            self.idx += 1
            return self._values[self.idx - 1]

    def __getitem__(self, key):
        if not self.fix:
            self._values = self.create_vals()
        return self._values[key]

    def __len__(self):
        if not self.fix:
            self._values = self.create_vals()
        return len(self._values)


    def __contains__(self, value):
        if not self.fix:
            self._values = self.create_vals()
        return value in self._values


# This function is to be replaced with SweepDependentValues
def create_set_vals(num_parameters=None, step=None, step_percentage=None,
                    points=None, window=None, set_parameters=None, silent=True,
                    center_val=None, min_val=None, max_val=None, reverse=False):
    warnings.warn('create_set_vals is deprecated, please use parameter.sweep()')
    parameter_config = silq.config.get('parameters')

    def calculate_step(min_step, max_step, percentage, round_vals=True):
        val = min_step * (max_step / min_step) ** (percentage / 100)
        if round_vals:
            # Round values to one decimal after first digit
            digits = int(np.ceil(np.log10(1 / val))) + 1
            val = round(val, digits)

        return val

    def determine_step(set_parameter):
        if step_percentage is not None:
            min_step, max_step = parameter_config[set_parameter.name]['steps']
            return calculate_step(min_step, max_step, step_percentage)
        else:
            return step

    def create_vals(set_parameter, points=None, window=None,
                    center_val=None, min_val=None, max_val=None):
        if window is None:
            if min_val is not None and max_val is not None:
                window = max_val - min_val
            else:
                step = determine_step(set_parameter)
                window = step * (points - 1)
        if points is None:
            step = determine_step(set_parameter)
            points = int(round(window / step))

        if min_val is None and max_val is None:
            if center_val is None:
                center_val = set_parameter()
            min_val = center_val - window / 2
            max_val = center_val + window / 2

        vals = list(np.linspace(min_val, max_val, points))
        if reverse:
            vals.reverse()

        if not silent:
            step = window / points
            print('{param}[{min_val}:{max_val}:{step}]'.format(
                param=set_parameter, min_val=min_val, max_val=max_val,
                step=step))

        return vals

    if set_parameters is None:
        # Get default parameters from station
        # the parameters to use depend on num_parameters
        station = qc.station.Station.default
        set_parameter_names = parameter_config['labels'][str(num_parameters)]
        set_parameters = [getattr(station, name) for name in
                          set_parameter_names]

    if isinstance(set_parameters, list):
        set_vals = []
        for k, set_parameter in enumerate(set_parameters):
            vals = create_vals(set_parameter, points=points, window=window,
                               center_val=center_val, min_val=min_val,
                               max_val=max_val)
            set_vals.append(set_parameter[vals])
        return set_vals
    else:
        # Set_parameters is a single parameter
        vals = create_vals(set_parameters, points=points, window=window,
                           center_val=center_val, min_val=min_val,
                           max_val=max_val)
        return set_parameters[vals]
