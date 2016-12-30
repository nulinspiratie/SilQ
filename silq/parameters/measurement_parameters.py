import numpy as np
import logging

import qcodes as qc
from qcodes.data import hdf5_format, io
from qcodes import config
from qcodes.instrument.parameter import Parameter
from qcodes.config.config import DotDict

from silq.tools import data_tools, general_tools
from silq.tools.general_tools import SettingsClass, clear_single_settings
from silq.measurements.measurement_types import Loop0DMeasurement, \
    Loop1DMeasurement, Loop2DMeasurement, ConditionSet
from silq.measurements.measurement_modules import MeasurementSequence

properties_config = config['user'].get('properties', {})


class MeasurementParameter(SettingsClass, Parameter):
    def __init__(self, name, acquisition_parameter=None, mode=None, **kwargs):
        SettingsClass.__init__(self)
        Parameter.__init__(self, name, snapshot_value=False, **kwargs)

        self.measurement_sequence = MeasurementSequence()

        self.mode = mode
        if self.mode is not None:
            # Add mode to parameter name and label
            self.name += self.mode_str

        self.acquisition_parameter = acquisition_parameter

        self.measurement = None

        self.loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')
        self._meta_attrs.extend(['acquisition_parameter_name'])

    def __repr__(self):
        return '{} measurement parameter'.format(self.name)

    @property
    def acquisition_parameter_name(self):
        return self.acquisition_parameter.name

    @property
    def base_folder(self):
        """
        Obtain measurement base folder (if any).
        Returns:
            If in a measurement, the base folder is the relative path of the
            data folder. Otherwise None
        """
        in_msmt = False

        if config['core']['legacy_mp']:
            # Multiprocessing is on, determining if in process 'Measurement'
            import multiprocessing as mp
            current_process = mp.process.current_process()
            in_msmt = (current_process.name == 'Measurement')
        else:
            logging.warning('Cannot determine if in msmt if not in bg mode')

        if in_msmt:
            return data_tools.get_latest_data_folder()
        else:
            return None

    def print_results(self):
        if getattr(self, 'names', None) is not None:
            for name, result in zip(self.names, self.results):
                print('{}: {:.3f}'.format(name, result))
        elif hasattr(self, 'results'):
            print('{}: {:.3f}'.format(self.name, self.results))


class SelectFrequencyParameter(MeasurementParameter):
    def __init__(self, threshold=0.5,
                 discriminant=None,
                 frequencies=None, mode=None,
                 acquisition_parameter=None, update_frequency=True, **kwargs):
        # Initialize SettingsClass first because its needed for
        # self.spin_states, self.discriminant etc.
        SettingsClass.__init__(self)
        self.mode = mode
        self.discriminant = discriminant

        names = ['{}_{}'.format(self.discriminant, spin_state)
                 for spin_state in self.spin_states]
        names.append('frequency' + self.mode_str)

        super().__init__(self, name='select_frequency',
                         label='Select frequency',
                         names=names,
                         **kwargs)

        self.acquisition_parameter = acquisition_parameter
        self.update_frequency = update_frequency
        self.threshold = threshold
        self.frequencies = frequencies

        self.frequency = None
        self.samples = None

        self._meta_attrs.extend(['frequencies', 'frequency', 'update_frequency',
                                 'spin_states', 'threshold', 'discriminant'])

    @property
    def spin_states(self):
        spin_states_unsorted = self.frequencies.keys()
        return sorted(spin_states_unsorted)

    @property
    def discriminant_idx(self):
        return self.acquisition_parameter.names.index(self.discriminant)

    @clear_single_settings
    def get(self):
        # Initialize frequency to the current frequency (default in case none
        #  of the measured frequencies satisfy conditions)
        frequency = self.acquisition_parameter.frequency
        self.acquisition_parameter.temporary_settings(samples=self.samples)

        frequencies = [self.frequencies[spin_state]
                       for spin_state in self.spin_states]
        self.condition_sets = ConditionSet(
            (self.discriminant, '>', self.threshold))

        # Create Measurement object and perform measurement
        self.measurement = Loop1DMeasurement(
            name=self.name, acquisition_parameter=self.acquisition_parameter,
            set_parameter=self.acquisition_parameter,
            base_folder=self.base_folder,
            condition_sets=self.condition_sets)
        self.measurement(frequencies)
        self.measurement()

        self.results = getattr(self.measurement.dataset, self.discriminant)

        # Determine optimal frequency and update config entry if needed
        self.condition_result = self.measurement.check_condition_sets()
        if self.condition_result['is_satsisfied']:
            frequency = self.measurement.optimal_set_vals[0]
            if self.update_frequency:
                properties_config['frequency' + self.mode_str] = frequency
        else:
            if not self.silent:
                logging.warning("Could not find frequency with high enough "
                                "contrast")

        self.results += [frequency]

        self.acquisition_parameter.clear_settings()

        # Print results
        if not self.silent:
            self.print_results()

        return self.results


class CalibrationParameter(SettingsClass, Parameter):
    def __init__(self, name, measurement_sequence, set_parameters=None,
                 acquisition_parameter=None, **kwargs):
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

        self.measurement_sequence = measurement_sequence

        names = ['success_dx', 'optimal_set_vals', self.discriminant]
        super().__init__(name, acquisition_parameter=acquisition_parameter,
                         names=names, **kwargs)

        # Get a dict of set_parameters, either from kwarg set_parameters,
        # or if this is equal to None, it is retrieved from config
        self.set_parameters = set_parameters
        station = qc.station.Station.default
        # Convert parameters that are given as strings to the actual objects
        self.set_parameters = [parameter if type(parameter) != str
                               else getattr(station, parameter)
                               for parameter in self.set_parameters]

        self._meta_attrs.extend(['conditions', 'measurement_sequence',
                                 'discriminant'])

    @clear_single_settings
    def get(self):
        self.measurement_sequence()

        optimal_set_vals = self.measurement_sequence.optimal_set_vals
        optimal_set_vals = [optimal_set_vals.get(parameter.name, parameter())
                            for parameter in self.set_parameters]

        success_idx = self.measurement_sequence.success_idx
        optimal_val = self.measurement_sequence.optimal_val

        return success_idx, optimal_set_vals, optimal_val
