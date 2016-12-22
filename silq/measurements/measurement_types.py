import numpy as np

import qcodes as qc
from qcodes.data import hdf5_format, io

from silq.tools import data_tools, general_tools
from silq.tools.general_tools import SettingsClass, get_truth


class Measurement(SettingsClass):
    def __init__(self, name, conditions=[], mode=None,
                 acquisition_parameter=None, discriminant=None):
        SettingsClass.__init__(self)

        self.name = name
        self.mode = mode
        self.acquisition_parameter = acquisition_parameter
        self.conditions = conditions
        self.discriminant = discriminant
        self.dataset = None
        self.satisfied_arr = None
        if self.mode is not None:
            # Add mode to parameter name and label
            self.name += self.mode_str

        self.loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')

    def __repr__(self):
        return '{} measurement'.format(self.name)

    def __call__(self, *args):
        if len(args) == 0:
            return self.get()
        else:
            self.set(*args)

    @property
    def disk_io(self):
        return io.DiskIO(data_tools.get_latest_data_folder())

    def test_conditions(self, dataset, conditions):
        """
        Checks if a dataset satisfies a set of conditions
        Args:
            dataset:

        Returns:

        """
        # Determine dimensionality from discriminant array
        dims = getattr(dataset, self.discriminant).ndarray.shape

        # Start of with all set points satisfying conditions
        self.satisfied_arr = np.ones(dims)

        for (attribute, relation, target_val) in conditions:
            test_arr = getattr(dataset, attribute).ndarray
            # Determine which elements satisfy condition
            satisfied_single_arr = get_truth(test_arr, target_val, relation)

            # Update satisfied elements with the ones satisfying current
            # condition
            self.satisfied_arr = np.logical_and(self.satisfied_arr,
                                                satisfied_single_arr)

            self.satisfies_conditions = np.any(self.satisfied_arr)
        return self.satisfied_arr

    def get_optimal_val(self, dataset, satisfied_arr=None):
        discriminant_vals = getattr(dataset, self.discriminant)

        # Convert arrays to 1D
        measurement_vals_1D = np.ravel(discriminant_vals)

        if satisfied_arr is not None:
            # Filter 1D arrays by those satisfying conditions
            satisfied_arr_1D = np.ravel(satisfied_arr)
            satisfied_idx = np.nonzero(satisfied_arr_1D)[0]

            measurement_vals_1D = np.take(measurement_vals_1D, satisfied_idx)

        max_idx = np.argmax(measurement_vals_1D)
        self.optimal_set_vals = self.set_vals_from_idx(max_idx)
        self.optimal_val = measurement_vals_1D[max_idx]
        return self.optimal_set_vals, self.optimal_val


class Loop0DMeasurement(Measurement):
    def __init__(self, name, acquisition_parameter=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)

    def get(self):
        self.measurement = qc.Measure(self.acquisition_parameter)
        self.dataset = self.measurement.run(
            name='{}_{}'.format(self.name, self.acquisition_parameter.name),
            data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        self.test_conditions(self.dataset, self.conditions)
        if self.satisfies_conditions:
            self.get_optimal_val(self.dataset, self.satisfied_arr)

        self._single_settings.clear()
        return self.dataset


class Loop1DMeasurement(Measurement):
    def __init__(self, name, set_parameter=None, acquisition_parameter=None,
                 set_vals=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)
        self.set_parameter = set_parameter
        self.set_vals = set_vals

    def get(self):
        # Set data saving parameters
        self.measurement = qc.Loop(
            self.set_parameter[self.set_vals]).each(
                self.acquisition_parameter)
        self.dataset = self.measurement.run(
            name='{}_{}_{}'.format(self.name, self.set_parameter.name,
                                   self.acquisition_parameter.name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        self.test_conditions(self.dataset, self.conditions)

        self._single_settings.clear()
        return self.dataset

    def set(self, val):
        self.set_vals = val


class Loop2DMeasurement(Measurement):
    def __init__(self, name, set_parameters=None, acquisition_parameter=None,
                 set_vals=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)
        self.set_parameters = set_parameters
        self.set_vals = set_vals

    def get(self):
        # Set data saving parameters
        self.measurement = qc.Loop(
            self.set_parameters[0][self.set_vals[0]]).loop(
            self.set_parameters[1][self.set_vals[1]]).each(
            self.acquisition_parameter)

        self.dataset = self.measurement.run(
            name='{}_{}_{}_{}'.format(self.name, self.set_parameters[0].name,
                                      self.set_parameters[1].name,
                                      self.acquisition_parameter.name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        self.test_conditions(self.dataset, self.conditions)

        self._single_settings.clear()
        return self.dataset

    def set(self, val):
        self.set_vals = val

