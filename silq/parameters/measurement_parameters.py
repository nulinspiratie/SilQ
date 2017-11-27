import numpy as np
import logging
from collections import Iterable

import qcodes as qc
from qcodes.loops import active_data_set, Loop, BreakIf
from qcodes.data import hdf5_format
from qcodes.instrument.parameter import MultiParameter

from silq import config
from silq.tools.general_tools import SettingsClass, clear_single_settings, \
    attribute_from_config, convert_setpoints, property_ignore_setter
from silq.tools.parameter_tools import create_set_vals
from silq.measurements.measurement_types import Loop0DMeasurement, \
    Loop1DMeasurement, Loop2DMeasurement, ConditionSet, TruthCondition
from silq.measurements.measurement_modules import MeasurementSequence
from silq.parameters import DCParameter, CombinedParameter

__all__ = ['MeasurementParameter', 'DCMultisweepParameter',
           'MeasurementSequenceParameter', 'SelectFrequencyParameter',
           'TrackPeakParameter']

logger = logging.getLogger(__name__)

properties_config = config.get('properties', {})
parameter_config = config.properties.get('parameters', {})
measurement_config = config.get('measurements', {})


class MeasurementParameter(SettingsClass, MultiParameter):

    def __init__(self, name, acquisition_parameter=None,
                 discriminant=None, silent=True, **kwargs):
        SettingsClass.__init__(self)
        MultiParameter.__init__(self, name, snapshot_value=False, **kwargs)

        self.discriminant = discriminant

        self.silent = silent
        self.acquisition_parameter = acquisition_parameter

        self._meta_attrs.extend(['acquisition_parameter_name'])

    def __repr__(self):
        return '{} measurement parameter'.format(self.name)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return attribute_from_config(item)

    @property
    def loc_provider(self):
        if self.base_folder is None:
            fmt = '{date}/#{counter}_{name}_{time}'
        else:
            fmt = self.base_folder + '/#{counter}_{name}_{time}'
        return qc.data.location.FormatLocation(fmt=fmt)

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
        active_dataset = active_data_set()
        if active_dataset is None:
            return None
        elif getattr(active_dataset, 'location', None):
            return active_dataset.location
        elif hasattr(active_dataset, '_location'):
            return active_dataset._location

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
            for name, result in self.results.items():
                print(f'{name}: {result:.3f}')
        elif hasattr(self, 'results'):
            print(f'{self.name}: {self.results[self.name]:.3f}')


class RetuneBlipsParameter(MeasurementParameter):
    def __init__(self,
                 name='retune_blips',
                 coulomb_peak_parameter=None,
                 blips_parameter=None,
                 sweep_parameter=None,
                 sweep_vals=None,
                 tune_to_coulomb_peak=True,
                 tune_to_optimum=True,
                 model_filepath=None,
                 voltage_limit=None,
                 **kwargs):
        # Load model here because it takes quite a while to load
        from keras.models import load_model

        super().__init__(name=name,
                         names=['optimal_vals', 'offsets'],
                         units=['mV', 'mV'],
                         shapes=((), ()),
                         **kwargs)

        self.sweep_parameter = sweep_parameter
        self.coulomb_peak_parameter = coulomb_peak_parameter
        self.blips_parameter = blips_parameter
        self.sweep_vals = sweep_vals
        self.tune_to_coulomb_peak = tune_to_coulomb_peak
        self.tune_to_optimum = tune_to_optimum

        self.initial_offsets = None
        self.voltage_limit = voltage_limit

        self.model_filepath = model_filepath
        if model_filepath is not None:
            self.model = self.model = load_model(self.model_filepath)
        else:
            logger.warning(f'No neural network model loaded for {self}')
            self.model = None

        self.continuous = True
        self.results = {}

        # Tools to only execute every nth call
        self.every_nth = None
        self.idx = 0

        self._meta_attrs.extend(['sweep_vals',
                                 'tune_to_coulomb_peak',
                                 'tune_to_optimum',
                                 'initial_offsets',
                                 'voltage_limit',
                                 'every_nth'])

    @property_ignore_setter
    def shapes(self):
        if isinstance(self.sweep_parameter, CombinedParameter):
            shape = (len(self.sweep_parameter.parameters), )
        else:
            shape = ()
        return (shape,) * len(self.names)

    def create_loop(self):
        loop = Loop(self.sweep_parameter[self.sweep_vals]).each(
            self.blips_parameter)
        return loop

    def calculate_optimum(self):
        if self.model is None:
            logger.warning('No Neural network model provided. skipping retune')
            return None

        blips_per_second = self.data.blips_per_second.ndarray
        mean_low_blip_duration = self.data.mean_low_blip_duration.ndarray
        mean_high_blip_duration = self.data.mean_high_blip_duration.ndarray

        if len(blips_per_second) != 21:
            raise RuntimeError(f'Must have 21 sweep vals, not {len(blips_per_second)}')

        data = np.zeros((len(blips_per_second), 3))

        # normalize data
        # Blips per second gets a gaussian normalization
        data[:,0] = (blips_per_second - np.mean(blips_per_second)) / np.std(blips_per_second)

        # blip durations get a logarithmic normalization, since the region
        # of interest has a low value
        log_offset = 1 # add offset since otherwise log(0) raises an error
        data[:,1] = np.log10(mean_low_blip_duration + log_offset)
        data[:,2] = np.log10(mean_high_blip_duration + log_offset)

        data = np.nan_to_num(data)
        data = np.expand_dims(data, 0)

        # Predict optimum value
        self.neural_network_results = self.model.predict(data)[0, 0]

        # Scale results
        # Neural network output is between -1 (sweep_vals[0]) and +1 (sweep_vals[-1])
        scale_factor = (self.sweep_vals[-1] - self.sweep_vals[0]) / 2
        self.optimal_val =  self.neural_network_results * scale_factor

        return self.optimal_val


    @clear_single_settings
    def get_raw(self):
        self.idx += 1
        if (self.every_nth is not None
            and (self.idx - 1) % self.every_nth != 0
            and self.results is not None):
            logger.debug(f'skipping iteration {self.idx} % {self.every_nth}')
            # Skip this iteration, return old results
            return [self.results[name] for name in self.names]


        initial_set_val = self.sweep_parameter()
        initial_offsets = self.sweep_parameter.offsets

        if self.initial_offsets is None:
            # Set initial offsets to ensure tuning does not reach out of bounds
            self.initial_offsets = initial_offsets

        # Get Coulomb peak
        if self.tune_to_coulomb_peak:
            self.coulomb_peak_parameter()

        self.loop = self.create_loop()
        self.data = self.loop.get_data_set(name='count_blips',
                                           location=self.loc_provider)

        try:
            self.loop.run(set_active=False, quiet=(active_data_set() is not None))
        finally:
            self.sweep_parameter(initial_set_val)

        if self.model_filepath is not None:
            optimum = self.calculate_optimum()

            if optimum is not None and self.tune_to_optimum:
                if self.voltage_limit is not None:
                    optimum_vals = self.sweep_parameter.calculate_individual_values(optimum)
                    voltage_differences = np.array(self.initial_offsets) - optimum_vals
                    if max(abs(voltage_differences)) > self.voltage_limit:
                        logging.warning(f'tune voltage {optimum_vals} outside '
                                        f'range, tuning back to initial value')
                        self.sweep_parameter.offsets = self.initial_offsets
                        self.sweep_parameter(0)
                    else:
                        self.sweep_parameter(optimum)
                        self.sweep_parameter.zero_offset()
                else:
                    self.sweep_parameter(optimum)
                    self.sweep_parameter.zero_offset()

        self.results = {
            'optimal_vals': self.sweep_parameter.offsets,
            'offsets': [offset - initial_offset for offset, initial_offset in
                        zip(self.sweep_parameter.offsets, initial_offsets)]
        }

        return [self.results[name] for name in self.names]


class CoulombPeakParameter(MeasurementParameter):
    def __init__(self,
                 name='coulomb_peak',
                 sweep_parameter=None,
                 acquisition_parameter=None,
                 combined_set_parameter=None,
                 DC_peak_offset=None,
                 tune_to_peak=True,
                 min_voltage=0.5,
                 **kwargs):

        if acquisition_parameter is None:
            acquisition_parameter = DCParameter()

        self.sweep_parameter = sweep_parameter
        self.sweep = {'range': [],
                      'step_percentage': None,
                      'num': None}
        self.min_voltage = min_voltage

        self.combined_set_parameter = combined_set_parameter
        self.DC_peak_offset = DC_peak_offset

        self.tune_to_peak = tune_to_peak

        self.results = {}

        super().__init__(name=name,
                         names=['optimum', 'max_voltage', 'DC_voltage'],
                         units=['V', 'V', 'V'],
                         shapes=((), (), ()),
                         acquisition_parameter=acquisition_parameter,
                         wrap_set=False, **kwargs)

        self._meta_attrs += ['min_voltage']

    def calculate_sweep_vals(self):
        if self.sweep_parameter is None:
            return None
        elif self.sweep['range']:
            return self.sweep_parameter[self.sweep['range']]
        elif self.sweep['step_percentage'] and self.sweep['num']:
            return self.sweep_parameter.sweep(
                step_percentage=self.sweep['step_percentage'],
            num=self.sweep['num'])
        else:
            return None

    @property_ignore_setter
    def names(self):
        sweep_vals = self.calculate_sweep_vals()
        if sweep_vals is not None:
            sweep_parameter = sweep_vals.parameter
            return [f'{sweep_parameter.name}_optimum', 'max_voltage', 'DC_voltage']
        else:
            return ['optimum', 'max_voltage', 'DC_voltage']

    @property_ignore_setter
    def shapes(self):
        sweep_vals = self.calculate_sweep_vals()
        if sweep_vals is not None:
            return ((), (), (len(sweep_vals),))
        else:
            return ((), (), ())

    def create_loop(self, sweep_vals):
        if sweep_vals is None:
            raise RuntimeError('Must define self.sweep_vals')

        loop = Loop(sweep_vals).each(self.acquisition_parameter)
        return loop

    @clear_single_settings
    def get_raw(self):
        if self.DC_peak_offset is not None:
            if self.combined_set_parameter is None:
                raise RuntimeError('Must specify combined_set_parameter')

            # Add DC offset
            self.combined_set_parameter(self.DC_peak_offset)

        #  Calculate sweep vals
        initial_set_val = self.sweep_parameter()
        sweep_vals = self.calculate_sweep_vals()

        self.loop = self.create_loop(sweep_vals=sweep_vals)

        self.data = self.loop.get_data_set(name=self.name,
                                           location=self.loc_provider)

        self.acquisition_parameter.setup()

        try:
            self.loop.run(set_active=False, quiet=(active_data_set() is not None))
        except:
            if self.DC_peak_offset is None:
                self.sweep_parameter(initial_set_val)
            else:
                self.combined_set_parameter(0)
                raise

        if self.min_voltage is not None and np.max(self.data.DC_voltage) < self.min_voltage:
            # Could not find coulomb peak
            self.results[self.names[0]] = np.nan

            # Tune back to original position
            if self.DC_peak_offset is None:
                self.sweep_parameter(initial_set_val)
            else:
                self.combined_set_parameter(0)
        else:
            # Found coulomb peak
            max_idx = np.argmax(self.data.DC_voltage)
            max_set_val = sweep_vals[max_idx]
            self.results[self.names[0]] = max_set_val

            if self.tune_to_peak:
                if self.DC_peak_offset is None:
                    self.sweep_parameter(max_set_val)
                else:
                    # Update combined_set_parameter offset
                    peak_offset = max_set_val - initial_set_val

                    sweep_parameter_idx = next(
                        index for index, item in enumerate(self.combined_set_parameter.parameters)
                        if item is self.sweep_parameter)

                    self.combined_set_parameter.offsets[sweep_parameter_idx] += peak_offset

                    self.combined_set_parameter(0)

        self.results['max_voltage'] = np.max(self.data.DC_voltage)
        self.results['DC_voltage'] = self.data.DC_voltage

        return [self.results[name] for name in self.names]


class DCMultisweepParameter(MeasurementParameter):
    def __init__(self, name, acquisition_parameter, x_gate, y_gate, **kwargs):
        super().__init__(name=name, names=['DC_voltage'], labels=['DC voltage'],
                         units=['V'], shapes=((1, 1),),
                         setpoint_names=((y_gate.name, x_gate.name),),
                         setpoint_units=(('V', 'V'),),
                         acquisition_parameter=acquisition_parameter)

        self.x_gate = x_gate
        self.y_gate = y_gate

        self.x_range = None
        self.y_range = None

        self.AC_range = 0.2
        self.pts = 120

        self.continuous = False

    @property
    def x_sweeps(self):
        return np.ceil((self.x_range[1] - self.x_range[0]) / self.AC_range)

    @property
    def y_sweeps(self):
        return np.ceil((self.y_range[1] - self.y_range[0]) / self.AC_range)

    @property
    def x_sweep_range(self):
        return (self.x_range[1] - self.x_range[0]) / self.x_sweeps

    @property
    def y_sweep_range(self):
        return (self.y_range[1] - self.y_range[0]) / self.y_sweeps

    @property
    def AC_x_vals(self):
        return np.linspace(-self.x_sweep_range / 2, self.x_sweep_range / 2,
                           self.pts + 1)[:-1]


    @property
    def AC_y_vals(self):
        return np.linspace(-self.y_sweep_range / 2, self.y_sweep_range / 2,
                           self.pts + 1)[:-1]

    @property
    def DC_x_vals(self):
        return np.linspace(self.x_range[0] + self.x_sweep_range / 2,
                           self.x_range[1] - self.x_sweep_range / 2,
                           self.x_sweeps).tolist()

    @property
    def DC_y_vals(self):
        return np.linspace(self.y_range[0] + self.y_sweep_range / 2,
                           self.y_range[1] - self.y_sweep_range / 2,
                           self.y_sweeps).tolist()

    @property_ignore_setter
    def setpoints(self):
        return convert_setpoints(np.linspace(self.y_range[0], self.y_range[1],
                                             self.pts * self.y_sweeps),
            np.linspace(self.x_range[0], self.x_range[1],
                        self.pts * self.x_sweeps)),

    @property_ignore_setter
    def shapes(self):
        return (len(self.DC_y_vals) * self.pts, len(self.DC_x_vals) * self.pts),

    def setup(self):
        self.acquisition_parameter.sweep_parameters.clear()
        self.acquisition_parameter.add_sweep(self.x_gate.name, self.AC_x_vals,
                                             connection_label=self.x_gate.name,
                                             offset_parameter=self.x_gate)
        self.acquisition_parameter.add_sweep(self.y_gate.name, self.AC_y_vals,
                                             connection_label=self.y_gate.name,
                                             offset_parameter=self.y_gate)
        self.acquisition_parameter.setup()

    def get(self):
        self.loop = qc.Loop(self.y_gate[self.DC_y_vals]).loop(
            self.x_gate[self.DC_x_vals]).each(self.acquisition_parameter)

        self.acquisition_parameter.temporary_settings(continuous=True)
        try:
            if not self.continuous:
                self.setup()
            self.data = self.loop.run(name=f'multi_2D_scan')
        # except:
        #     logger.debug('except stopping')
        #     self.layout.stop()
        #     self.acquisition_parameter.clear_settings()
        #     raise
        finally:
            if not self.continuous:
                logger.debug('finally stopping')
                self.layout.stop()
                self.acquisition_parameter.clear_settings()


        arr = np.zeros((len(self.DC_y_vals) * self.pts,
                        len(self.DC_x_vals) * self.pts))
        for y_idx in range(len(self.DC_y_vals)):
            for x_idx in range(len(self.DC_x_vals)):
                DC_data = self.data.DC_voltage[y_idx, x_idx]
                arr[y_idx * self.pts:(y_idx + 1) * self.pts,
                x_idx * self.pts:(x_idx + 1) * self.pts] = DC_data
        return arr,


class MeasurementSequenceParameter(MeasurementParameter):
    def __init__(self, name, measurement_sequence=None,
                 set_parameters=None, discriminant=None,
                 start_condition=None, **kwargs):
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
        self.start_condition = start_condition

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
        if self.start_condition is None:
            start_condition_satisfied = True
        elif isinstance(self.start_condition, TruthCondition):
            if self.acquisition_parameter.results is None:
                logger.info('Start TruthCondition satisfied because '
                            'acquisition parameter has no results')
                start_condition_satisfied = True
            else:
                start_condition_satisfied = \
                    self.start_condition.check_satisfied(
                        self.acquisition_parameter.results)[0]
                logger.info(f'Start Truth condition {self.start_condition} '
                            f'satisfied: {start_condition_satisfied}')
        else:
            start_condition_satisfied = self.start_condition()
            logger.info(f'Start condition function {self.start_condition} '
                        f'satisfied: {start_condition_satisfied}')

        if not start_condition_satisfied:
            num_measurements = -1
            optimal_set_vals = [p() for p in self.set_parameters]
            optimal_val = self.measurement_sequence.optimal_val
            return num_measurements, optimal_set_vals, optimal_val
        else:
            self.measurement_sequence.base_folder = self.base_folder
            result = self.measurement_sequence()
            num_measurements = self.measurement_sequence.num_measurements

            logger.info(f"Measurements performed: {num_measurements}")

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

        self.results = {self.discriminant: getattr(self.measurement.dataset,
                                                   self.discriminant)}

        # Determine optimal frequency and update config entry if needed
        self.condition_result = self.measurement.check_condition_sets()
        if self.condition_result['is_satsisfied']:
            frequency = self.measurement.optimal_set_vals[0]
            if self.update_frequency:
                properties_config['frequency'] = frequency
        else:
            if not self.silent:
                logger.warning("Could not find frequency with high enough "
                                "contrast")

        self['frequency'] = frequency

        self.acquisition_parameter.clear_settings()

        # Print results
        if not self.silent:
            self.print_results()

        return [self.results[name] for name in self.names]


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
