import numpy as np
import logging

import qcodes as qc
from qcodes.data import hdf5_format, io
from qcodes import config
from qcodes.instrument.parameter import MultiParameter
from qcodes.config.config import DotDict

from silq.tools import data_tools, general_tools
from silq.tools.general_tools import SettingsClass, clear_single_settings, \
    attribute_from_config
from silq.tools.parameter_tools import create_set_vals
from silq.measurements.measurement_types import Loop0DMeasurement, \
    Loop1DMeasurement, Loop2DMeasurement, ConditionSet
from silq.measurements.measurement_modules import MeasurementSequence

properties_config = config['user'].get('properties', {})
parameter_config = qc.config['user']['properties'].get('parameters', {})
measurement_config = qc.config['user'].get('measurements', {})


class MeasurementParameter(SettingsClass, MultiParameter):
    def __init__(self, name, acquisition_parameter=None,
                 discriminant=None, silent=True, **kwargs):
        SettingsClass.__init__(self)
        MultiParameter.__init__(self, name, snapshot_value=False, **kwargs)

        self.discriminant = discriminant

        self.silent = silent
        self.acquisition_parameter = acquisition_parameter

        self.loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')
        self._meta_attrs.extend(['acquisition_parameter_name'])

    def __repr__(self):
        return '{} measurement parameter'.format(self.name)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return attribute_from_config(item)

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
        # TODO Currently broken!
        return None

    @property
    def discriminant(self):
        if self._discriminant is not None:
            return self._discriminant
        else:
            return self.acquisition_parameter.name

    @discriminant.setter
    def discriminant(self, val):
        self._discriminant = val


    @property
    def discriminant_idx(self):
        if self.acquisition_parameter.name is not None:
            return self.acquisition_parameter.names.index(self.discriminant)
        else:
            return None

    def print_results(self):
        if getattr(self, 'names', None) is not None:
            for name, result in zip(self.names, self.results):
                print('{}: {:.3f}'.format(name, result))
        elif hasattr(self, 'results'):
            print('{}: {:.3f}'.format(self.name, self.results))


class MeasurementSequenceParameter(MeasurementParameter):
    def __init__(self, name, measurement_sequence=None,
                 set_parameters=None, discriminant=None, **kwargs):
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
        self.set_parameters = set_parameters

        if isinstance(measurement_sequence, str):
            # Load sequence from dict
            load_dict = measurement_config[measurement_sequence]
            measurement_sequence = MeasurementSequence.load_from_dict(load_dict)
        self.measurement_sequence = measurement_sequence
        self.acquisition_parameter = measurement_sequence.acquisition_parameter

        super().__init__(
            name=name,
            names=[name+'_msmts', 'optimal_set_vals', self.discriminant],
            shapes=((), (len(self.set_parameters),), ()),
            discriminant=self.discriminant,
            acquisition_parameter=self.acquisition_parameter,
            **kwargs)


        self._meta_attrs.extend(['discriminant'])

    @clear_single_settings
    def get(self):
        result = self.measurement_sequence()
        num_measurements = self.measurement_sequence.num_measurements

        if result['action'] == 'success':
            # Retrieve dict of {param.name: val} of optimal set vals
            optimal_set_vals = self.measurement_sequence.optimal_set_vals
            # Convert dict to list of set vals
            optimal_set_vals = [optimal_set_vals.get(p.name, p())
                                for p in self.set_parameters]
        else:
            optimal_set_vals = [p() for p in self.set_parameters]

        optimal_val = self.measurement_sequence.optimal_val

        return num_measurements, optimal_set_vals, optimal_val


class SelectFrequencyParameter(MeasurementParameter):
    def __init__(self, threshold=0.5, discriminant=None,
                 frequencies=None, mode=None,
                 acquisition_parameter=None, update_frequency=True, **kwargs):
        # Initialize SettingsClass first because its needed for
        # self.spin_states, self.discriminant etc.
        SettingsClass.__init__(self)
        self.mode = mode
        self._discriminant = discriminant

        names = ['{}_{}'.format(self.discriminant, spin_state)
                 for spin_state in self.spin_states]
        names.append('frequency')

        super().__init__(self, name='select_frequency',
                         label='Select frequency',
                         names=names,
                         discriminant=self.discriminant,
                         **kwargs)

        self.acquisition_parameter = acquisition_parameter
        self.update_frequency = update_frequency
        self.threshold = threshold
        self.frequencies = frequencies

        self.frequency = None
        self.samples = None
        self.measurement = None

        self._meta_attrs.extend(['frequencies', 'frequency', 'update_frequency',
                                 'spin_states', 'threshold', 'discriminant'])

    @property
    def spin_states(self):
        spin_states_unsorted = self.frequencies.keys()
        return sorted(spin_states_unsorted)

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
                properties_config['frequency'] = frequency
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


class TrackPeakParameter(MeasurementParameter):
    def __init__(self, name, set_parameter=None, acquisition_parameter=None,
                 step_percentage=None, peak_width=None, points=None,
                 discriminant=None, threshold=None, **kwargs):
        SettingsClass.__init__(self)
        self.set_parameter = set_parameter
        self.acquisition_parameter = acquisition_parameter
        self._discriminant = discriminant
        names = ['optimal_set_vals', self.set_parameter.name + '_set',
                 self.discriminant]
        super().__init__(name=name, names=names, discriminant=self.discriminant,
                         acquisition_parameter=acquisition_parameter, **kwargs)

        self.step_percentage = step_percentage
        self.peak_width = peak_width
        self.points = points
        self.threshold = threshold

        self.condition_sets = None
        self.measurement = None

    @property
    def set_vals(self):
        if self.peak_width is None and \
                (self.points is None or self.step_percentage is None):
            # Retrieve peak_width from parameter config only if above
            # conditions are satisfied
            self.peak_width = parameter_config[self.set_parameter]['peak_width']

        return create_set_vals(num_parameters=1,
                               step_percentage=self.step_percentage,
                               points=self.points,
                               window=self.peak_width,
                               set_parameters=self.set_parameter)

    @property
    def shapes(self):
        return [(), (len(self.set_vals), ), (len(self.set_vals),)]

    @clear_single_settings
    def get(self):
        # Create measurement object
        if self.threshold is not None:
            # Set condition set
            self.condition_sets = \
                [ConditionSet((self.discriminant, '>', self.threshold))]

        self.measurement = Loop1DMeasurement(
            name=self.name, acquisition_parameter=self.acquisition_parameter,
            set_parameter=self.set_parameter,
            base_folder=self.base_folder,
            condition_sets=self.condition_sets,
            discriminant=self.discriminant,
            silent=self.silent, update=True)

        # Set loop values
        self.measurement(self.set_vals)
        # Obtain set vals as a list instead of a parameter iterable
        set_vals = self.set_vals[:]
        self.measurement()

        trace = getattr(self.measurement.dataset, self.discriminant)

        optimal_set_val = self.measurement.optimal_set_vals[
            self.set_parameter.name]
        self.result = [optimal_set_val, set_vals, trace]
        return self.result
