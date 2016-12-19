import numpy as np
import logging

import qcodes as qc
from qcodes.data import hdf5_format, io
from qcodes import config
from qcodes.instrument.parameter import Parameter

from silq.tools import data_tools, general_tools
from silq.tools.general_tools import SettingsClass

properties_config = config['user'].get('properties', {})


class MeasurementParameter(SettingsClass, Parameter):
    def __init__(self, name, acquisition_parameter=None, mode=None, **kwargs):
        SettingsClass.__init__(self)
        Parameter.__init__(self, name, snapshot_value=False, **kwargs)

        if self.mode is not None:
            # Add mode to parameter name and label
            self.name += self.mode_str

        self.acquisition_parameter = acquisition_parameter

        self.silent = True

        self.loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')
        self._meta_attrs.extend(['acquisition_parameter_name'])

    def __repr__(self):
        return '{} measurement parameter'.format(self.name)

    @property
    def acquisition_parameter_name(self):
        return self.acquisition_parameter.name

    @property
    def disk_io(self):
        return io.DiskIO(data_tools.get_latest_data_folder())

    def _attribute_from_config(self, item):
        """
        Check if attribute exists somewhere in the config
        It first ill check properties config if a key matches the item
        with self.mode appended. This is only checked if the param has a mode.
        Finally, it will check if properties_config contains the item
        """
        # check if {item}_{self.mode} is in properties_config
        # if mode is None, mode_str='', in which case it checks for {item}
        item_mode = '{}{}'.format(item, self.mode_str)
        if item_mode in properties_config:
            return properties_config[item_mode]

        # Check if item is in properties config
        if item in properties_config:
            return properties_config[item]

        return None

    def print_results(self):
        if getattr(self, 'name', None) is not None:
            for name, result in zip(self.names, self.results):
                print('{}: {:.3f}'.format(name, result))
        elif hasattr(self, 'results'):
            print('{}: {:.3f}'.format(self.name, self.results))


class Loop0DParameter(MeasurementParameter):
    def __init__(self, name, acquisition_parameter=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)
        self.data = None

    def get(self):
        self.measurement = qc.Measure(self.acquisition_parameter)
        self.data = self.measurement.run(
            name='{}_{}'.format(self.name, self.acquisition_parameter_name),
            data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        if not self.silent:
            self.print_results()

        self._single_settings.clear()
        return self.data


class Loop1DParameter(MeasurementParameter):
    def __init__(self, name, set_parameter=None, acquisition_parameter=None,
                 set_vals=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)
        self.set_parameter = set_parameter
        self.set_vals = set_vals

        self.data = None

        self._meta_attrs.extend(['set_parameter_name'])

    @property
    def set_parameter_name(self):
        return self.set_parameter.name

    def get(self):
        # Set data saving parameters
        self.measurement = qc.Loop(
            self.set_parameter[self.set_vals]).each(
                self.acquisition_parameter)
        self.data = self.measurement.run(
            name='{}_{}_{}'.format(self.name, self.set_parameter_name,
                                   self.acquisition_parameter_name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        if not self.silent:
            self.print_results()

        self._single_settings.clear()
        return self.data

    def set(self, val):
        self.set_vals = val


class Loop2DParameter(MeasurementParameter):
    def __init__(self, name, set_parameters=None, acquisition_parameter=None,
                 set_vals=None, **kwargs):
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         **kwargs)
        self.set_parameters = set_parameters
        self.set_vals = set_vals

        self.data = None

        self._meta_attrs.extend(['set_parameters_names'])

    @property
    def set_parameters_names(self):
        return [set_parameter.name for set_parameter in self.set_parameters]

    def get(self):
        # Set data saving parameters
        self.measurement = qc.Loop(
            self.set_parameters[0][self.set_vals[0]]).loop(
                self.set_parameters[1][self.set_vals[1]]).each(
                    self.acquisition_parameter)

        self.data = self.measurement.run(
            name='{}_{}_{}_{}'.format(self.name, self.set_parameters[0].name,
                                      self.set_parameters[1].name,
                                      self.acquisition_parameter_name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)

        if not self.silent:
            self.print_results()

        self._single_settings.clear()
        return self.data

    def set(self, val):
        self.set_vals = val


class SelectFrequencyParameter(MeasurementParameter):
    def __init__(self, threshold=0.5,
                 discriminant=None,
                 frequencies=None,
                 acquisition_parameter=None, update_frequency=True, **kwargs):
        SettingsClass.__init__(self)

        self.frequencies = frequencies
        self.frequency = None

        self.discriminant = discriminant

        self.mode = kwargs.get('mode', None)
        names = [self.discriminant + spin_state
                 for spin_state in self.spin_states]
        if self.mode is not None:
            names.append('frequency_{}'.format(self.mode))
        else:
            names.append('frequency')

        super().__init__(name='select_frequency',
                         label='Select frequency',
                         names=names,
                         acquisition_parameter=acquisition_parameter,
                         **kwargs)

        self.update_frequency = update_frequency
        self.threshold = threshold

        self._meta_attrs.extend(['frequencies', 'frequency', 'update_frequency',
                                 'spin_states', 'threshold', 'discriminant'])

    @property
    def spin_states(self):
        spin_states_unsorted = self.frequencies.keys()
        return sorted(spin_states_unsorted)

    @property
    def discriminant_idx(self):
        return self.acquisition_parameter.names.index(self.discriminant)

    def get(self):
        self.results = []
        # Perform measurement for all frequencies
        for spin_state in self.spin_states:
            # Set adiabatic frequency
            self.acquisition_parameter(self.frequencies[spin_state])
            fidelities = self.acquisition_parameter()

            # Only add discriminant
            self.results.append(fidelities[self.discriminant_idx])

        optimal_idx = np.argmax(self.results)
        optimal_spin_state = self.spin_states[optimal_idx]

        frequency = self.frequencies[optimal_spin_state]
        self.results += [frequency]

        if self.update_frequency and max(self.results) > self.threshold:
            properties_config['frequency' + self.mode_str] = frequency
        elif not self.silent:
            logging.warning("Could not find frequency with high enough "
                            "contrast")

        # Print results
        if not self.silent:
            self.print_results()

        self._single_settings.clear()
        return self.results


class CalibrationParameter(MeasurementParameter):
    def __init__(self, name, operations, discriminant=None, set_parameters=None,
                 acquisition_parameter=None, conditions=None, **kwargs):
        """

        Args:
            name:
            set_parameters:
            acquisition_parameter:
            operations:
            discriminant:
            conditions: Must be of one of the following forms
                {'mode': 'measure'}
                {'mode': '1D_scan', 'span', 'set_points', 'set_parameter',
                 'center_val'(optional)
            **kwargs:
        """
        SettingsClass.__init__(self)

        self.discriminant = discriminant
        self.conditions = conditions
        self.operations = operations

        names = ['success', 'optimal_set_vals', self.discriminant]
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         names=names, **kwargs)

        self.set_parameters = {p.name: p for p in set_parameters}

        self._meta_attrs.extend(['acquisition_parameter_name', 'conditions',
                                 'operations', 'discriminant',
                                 'set_vals_1D', 'measure_vals_1D'])

    def satisfies_conditions(self, dataset, dims):
        # Start of with all set points satisfying conditions
        satisfied_final_arr = np.ones(dims)
        if self.conditions is None:
            return satisfied_final_arr
        for (attribute, target_val, relation) in self.conditions:
            test_vals = getattr(dataset, attribute).ndarray
            # Determine which elements satisfy condition
            satisfied_arr = general_tools.get_truth(test_vals, target_val,
                                                    relation)

            # Update satisfied elements with the ones satisfying current
            # condition
            satisfied_final_arr = np.logical_and(satisfied_final_arr,
                                                 satisfied_arr)
        return satisfied_final_arr

    def optimal_val(self, dataset, satisfied_set_vals=None):
        measurement_vals = getattr(dataset, self.key)

        set_vals_1D = np.ravel(self.set_vals)
        measurement_vals_1D = np.ravel(measurement_vals)

        if satisfied_set_vals is not None:
            # Filter 1D arrays by those satisfying conditions
            satisfied_set_vals_1D = np.ravel(satisfied_set_vals)
            satisfied_idx = np.nonzero(satisfied_set_vals)[0]

            set_vals_1D = np.take(set_vals_1D, satisfied_idx)
            measurement_vals_1D = np.take(measurement_vals_1D, satisfied_idx)

        max_idx = np.argmax(measurement_vals_1D)
        return set_vals_1D[max_idx], measurement_vals_1D[max_idx]

    def get_0D_loop(self, operation):
        self.set_vals = [None]
        loop_parameter = Loop0DParameter(
            name='calibration_0D',
            mode=self.mode,
            acquisition_parameter=self.acquisition_parameter)
        dims = (1)
        return loop_parameter, dims

    def get_1D_loop(self, operation):
        # Setup set vals
        set_parameter = self.set_parameters[operation['set_parameter']]

        # If no center_val provided, use current set_parameter val
        center_val = operation.get('center_val', set_parameter())
        span = operation['span']
        set_points = operation['set_points']
        self.set_vals = list(np.linspace(center_val - span / 2,
                                         center_val + span / 2,
                                         set_points))
        # Extract set_parameter
        loop_parameter = Loop1DParameter(
            name='calibration_1D',
            mode=self.mode,
            set_parameter=set_parameter,
            acquisition_parameter=self.acquisition_parameter,
            set_vals=self.set_vals)
        dims = (set_points)
        return loop_parameter, dims

    def get(self):
        self.loop_parameters = []
        self.datasets = []

        if hasattr(self.acquisition_parameter, 'setup'):
            self.acquisition_parameter.setup()

        self.success = False
        self.finished = False

        for k, operation in enumerate(self.operations):
            if operation['mode'] == 'measure':
                loop_parameter, dims = self.get_0D_loop(operation)
            elif operation['mode'] == '1D_scan':
                loop_parameter, dims = self.get_1D_loop(operation)
            elif operation['mode'] == '2D_scan':
                loop_parameter, dims = self.get_1D_loop(operation)
            else:
                raise ValueError("Calibration mode {} not "
                                 "implemented".format(operation['mode']))

            self.loop_parameters.append(loop_parameter)

            dataset = loop_parameter()
            self.datasets.append(dataset)

            satisfied_set_vals = self.satisfies_conditions(dataset, dims)
            if np.any(satisfied_set_vals):
                optimal_set_val, optimal_get_val = self.optimal_val(
                    dataset, satisfied_set_vals)
                cal_success = k
                break
        else:
            logging.warning('Could not find calibration point satisfying '
                            'conditions. Choosing best alternative')
            optimal_set_val, optimal_get_val = self.optimal_val(dataset)
            cal_success = -1

        if optimal_set_val is not None:
            set_parameter(optimal_set_val)
            # TODO implement for 2D
        else:
            optimal_set_val = set_parameter()

        self._single_settings.clear()
        return cal_success, optimal_set_val, optimal_get_val
