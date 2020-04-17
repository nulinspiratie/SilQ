import numpy as np
import logging
from collections import Iterable
import numbers
from scipy.interpolate import interp1d

import qcodes as qc
from qcodes import Instrument
from qcodes.loops import active_dataset, Loop, BreakIf
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

__all__ = ['MeasurementParameter', 'RetuneBlipsParameter',
           'CoulombPeakParameter', 'DCMultisweepParameter',
           'MeasurementSequenceParameter', 'SelectFrequencyParameter',
           'TrackPeakParameter']

logger = logging.getLogger(__name__)

properties_config = config.get('properties', {})
parameter_config = config.properties.get('parameters', {})
measurement_config = config.get('measurements', {})


class MeasurementParameter(SettingsClass, MultiParameter):
    """Base class for parameters that perform measurements.

    A `MeasurementParameter` usually consists of several acquisitions,
    which it uses for complex sequences.

    A `MeasurementParameter` usually uses a ``qcodes.Loop`` or a
    ``qcodes.Measure`` or several in succession. The results in the ``DataSet``
    are analysed and often some post-action is performed.

    Example:
        An example of a `MeasurementParameter` is a retuning sequence, which
        uses an `AcquisitionParameter`, and from that determines how much
        voltages have to be modified to retune the system
        (e.g. `RetuneBlipsParameter`).

    Note:
        The MeasurementParameter needs to be updated. It was originally created
        to be used with the `MeasurementSequence`, but this class turned out to
        be too rigid. Instead, measurements should be programmed by subclassing
        the MeasurementParameter.

    Args:
        Name: Parameter name
        acquisition_parameter: Acquisition_parameter to use. Fails in case of
            multiple or no acquisition parameters
        discriminant: data array in dataset to discriminate. Fails if there is
            no single discriminant.

    Parameters:
        silent (str): Print results during .get()

    Todo:
        * Clean up MeasurementParameter, remove attributes
          ``MeasurementParameter.discriminant`` and
          ``MeasurementParameter.acquisition_parameter``.

    """
    layout = None

    def __init__(self,
                 name,
                 acquisition_parameter=None,
                 discriminant=None, silent=True, **kwargs):
        SettingsClass.__init__(self)
        MultiParameter.__init__(self, name, snapshot_value=False, **kwargs)

        if self.layout is None:
            try:
                MeasurementParameter.layout = Instrument.find_instrument(
                    'layout')
            except KeyError:
                logger.warning(f'No layout found for {self}')

        self.discriminant = discriminant

        self.silent = silent
        self.acquisition_parameter = acquisition_parameter

        self._meta_attrs.extend(['acquisition_parameter_name'])

    def __repr__(self):
        return f'{self.name} measurement parameter'

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return attribute_from_config(item, config.properties)

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
        active_dataset = active_dataset()
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
                if isinstance(result, numbers.Number):
                    print(f'{name}: {result:.3f}')
                else:
                    print(f'{name}: {result}')
        elif hasattr(self, 'results'):
            print(f'{self.name}: {self.results[self.name]:.3f}')


class RetuneBlipsParameter(MeasurementParameter):
    """Parameter that retunes by analysing blips using a neural network

    The first (optional) stage is to use a CoulombPeakParameter to find the
    center of the Coulomb peak.

    Second, a sweep parameter is varied for a range of sweep values. For each
    sweep point, a trace is acquired, and its blips measured in a BlipsParameter.
    This information is then analysed by a neural network, from which the
    optimal tuning position is predicted.

    The Neural network is a Keras model that needs to be pre-trained with data.
    More info: Experiments/personal/Serwan/Neural networks/Retune blips.ipynb
    The following Neural Network seems to produce decent results:

    >>> model = Sequential()
    >>> model.add(Dense(3, activation='linear', input_shape=(21,3)))
    >>> model.add(Flatten())
    >>> model.add(Dense(1, activation='linear'))
    """
    def __init__(self,
                 name='retune_blips',
                 coulomb_peak_parameter=None,
                 blips_parameter=None,
                 sweep_parameter=None,
                 sweep_vals=None,
                 tune_to_coulomb_peak=True,
                 tune_to_optimum=True,
                 optimum_DC_offset=None,
                 model_filepath=None,
                 voltage_limit=None,
                 optimum_method='neural_network',
                 **kwargs):
        """
        Args:
            name: name of parameter (default `retune_blips`)
            coulomb_peak_parameter: CoulombPeakParameter that tunes to the
                center of the Coulomb peak. Should have all its settings
                predefined, including sweep_range, and DC_offset
            blips_parameter: BlipsParameter that measures blips properties from
                a trace, to be analysed by a neural network. Should have all its
                settings predefined, including duration
            sweep_parameter: sweep CombinedParameter with scales defined to
                remain compensated on the Coulomb peak. The offsets must be set
                such that zero approximately corresponds to the tuning position.
            sweep_vals: Range of sweep values to sweep the sweep_parameter and
                measure blips_parameter. It is important that both the number of
                sweep values and distance between sweep points is the same as
                in the training data.
            tune_to_coulomb_peak: Whether to use the coulomb_peak_parameter.
                Otherwise, it will always assume the system remains on the
                Coulomb peak
            tune_to_optimum: Tune to optimum values predicted by neural network.
                If True, the offsets of the sweep parameter are also zeroed.
            optimum_DC_offset: Additional offset to optimum determined from
                neural network model. Used if the model is not trained properly
                and has a constant offset in predictions.
                If single value, combined parameter will be offset.
                If tuple, offsets are per parameter in combined parameter.
            model_filepath: Filepath of Keras neural network (.h5 extension).
            voltage_limit: Maximum voltage difference from initial offsets to
                avoid the system .
                The first time this parameter is called, the sweep parameter's
                initial offsets are stored. If one of the optimal values is more
                than voltage_limit away from its initial offset, a warning is
                raised and the gates are returned to the initial offsets.
            **kwargs: kwargs passed to MeasurementParameter
        """
        # Load model here because it takes quite a while to load
        from keras.models import load_model
        import tensorflow as tf

        super().__init__(name=name,
                         names=['optimal_vals', 'offsets'],
                         units=['V', 'V'],
                         shapes=((), ()),
                         **kwargs)

        self.sweep_parameter = sweep_parameter
        self.coulomb_peak_parameter = coulomb_peak_parameter
        self.blips_parameter = blips_parameter
        self.sweep_vals = sweep_vals
        self.tune_to_coulomb_peak = tune_to_coulomb_peak
        self.tune_to_optimum = tune_to_optimum
        self.optimum_DC_offset = optimum_DC_offset

        self.initial_offsets = None
        self.voltage_limit = voltage_limit

        assert optimum_method in ['neural_network', 'max_blips']
        self.optimum_method = optimum_method

        self.model_filepath = model_filepath
        if model_filepath is not None:
            self.model = load_model(self.model_filepath)
            self.model._make_predict_function()
            self.graph = tf.get_default_graph()
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
            shape = (len(self.sweep_parameter.parameters),)
        else:
            shape = ()
        return (shape,) * len(self.names)

    def create_loop(self):
        """
        Create loop that sweep sweep_parameter over sweep_vals and measures
        blips_parameter at each sweep point.
        Returns: loop

        """
        loop = Loop(self.sweep_parameter[self.sweep_vals]).each(
            self.blips_parameter)
        return loop

    def calculate_optimum(self):
        """
        Calculate optimum of dataset from neural network
        Returns: optimal voltage of combined set parameter
        """
        blips_per_second = self.data.blips_per_second.ndarray
        blips_per_second = np.nan_to_num(blips_per_second)
        mean_low_blip_duration = self.data.mean_low_blip_duration.ndarray
        mean_low_blip_duration = np.nan_to_num(mean_low_blip_duration)
        mean_high_blip_duration = self.data.mean_high_blip_duration.ndarray
        mean_high_blip_duration = np.nan_to_num(mean_high_blip_duration)

        if self.optimum_method == 'max_blips':
            if not np.nanmax(blips_per_second):
                return None
            else:
                max_idx = np.nanargmax(blips_per_second)
                return self.sweep_vals[max_idx]

        if self.model is None:
            logger.warning('No Neural network model provided. skipping retune')
            return None

        if len(blips_per_second) != 21:
            raise RuntimeError(
                f'Must have 21 sweep vals, not {len(blips_per_second)}')

        data = np.zeros((len(blips_per_second), 3))

        # normalize data
        # Blips per second gets a gaussian normalization
        if np.std(blips_per_second):
            data[:, 0] = (blips_per_second - np.mean(blips_per_second)) / np.std(
                blips_per_second)
        else:
            return None

        # blip durations get a logarithmic normalization, since the region
        # of interest has a low value
        log_offset = 1e-3  # add offset since otherwise log(0) raises an error
        data[:, 1] = np.log10(mean_low_blip_duration + log_offset)
        data[:, 2] = np.log10(mean_high_blip_duration + log_offset)

        # Center logarithmic blip durations around zero
        for k in range(1, 3):
            data[:, k] += 1.5

        data = np.expand_dims(data, 0)

        # Predict optimum value
        try:
            with self.graph.as_default():
                self.neural_network_results = self.model.predict(data)[0, 0]
        except Exception as e:
            import traceback
            logger.error(traceback.print_exc())
            return

        # Scale results
        # Neural network output is between -1 (sweep_vals[0]) and +1 (sweep_vals[-1])
        scale_factor = (self.sweep_vals[-1] - self.sweep_vals[0]) / 2
        self.optimal_val = self.neural_network_results * scale_factor

        return self.optimal_val

    @clear_single_settings
    def get_raw(self):
        assert self.model_filepath is not None, "Must provide model filepath"

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
            self.loop.run(set_active=False, quiet=True)
        finally:
            self.sweep_parameter(initial_set_val)

        optimum = self.calculate_optimum()

        if optimum is None:
            tune_to_optimum = False
            optimal_vals = initial_offsets
            offsets = [0] * len(initial_offsets)
        else:
            optimal_vals = self.sweep_parameter.calculate_individual_values(optimum)
            offsets = [offset - initial_offset for offset, initial_offset in
                       zip(optimal_vals, initial_offsets)]

            if not self.tune_to_optimum:
                tune_to_optimum = False
            elif self.voltage_limit is None:
                tune_to_optimum = True
            else:
                voltage_differences = np.array(self.initial_offsets) - optimal_vals
                if max(abs(voltage_differences)) < self.voltage_limit:
                    tune_to_optimum = True
                else:
                    logging.warning(f'tune voltage {optimal_vals} outside '
                                    f'range, tuning back to initial value')
                    # self.sweep_parameter.offsets = self.initial_offsets
                    # self.sweep_parameter(0)
                    tune_to_optimum = False

        if tune_to_optimum:
            self.sweep_parameter(optimum)
            self.sweep_parameter.zero_offset()

            if isinstance(self.optimum_DC_offset, float):
                self.sweep_parameter(self.optimum_DC_offset)
            elif self.optimum_DC_offset:
                for parameter, offset in zip(self.sweep_parameter.parameters,
                                             self.optimum_DC_offset):
                    parameter(parameter() + offset)

        self.results = {
            'optimal_vals': optimal_vals,
            'offsets': offsets
        }

        return [self.results[name] for name in self.names]


class CoulombPeakParameter(MeasurementParameter):
    """
    Parameter that finds Coulomb peak and can tune to it.
    Finding the Coulomb peak is done by sweeping a gate and measuring the DC
    voltage at each point.
    """

    def __init__(self,
                 name='coulomb_peak',
                 sweep_parameter=None,
                 acquisition_parameter=None,
                 combined_set_parameter=None,
                 DC_peak_offset=None,
                 tune_to_peak=True,
                 min_voltage=0.5,
                 interpolate: bool = True,
                 sweep_settings: dict = None,
                 **kwargs):
        """
        Args:
            name: name of parameter (default coulomb_peak)
            sweep_parameter: gate parameter to sweep over Coulomb peak
            acquisition_parameter: parameter that measures DC voltage.
                If not provided, a default DCParameter is created.
            combined_set_parameter: CombinedParameter of gate parameters whose
                scale must be set to remain compensated on the Coulomb peak.
                Only needed if a DC_peak_offset is provided.
                Offsets must be set such that DC_peak_offset is relative to
                zero.
            DC_peak_offset: Any DC peak offset to set the combined_set_parameter
                to. Useful if you want to tune away from the transition when
                performing Coulomb peak scan.
            tune_to_peak: Tune to peak after scan is complete.
                If a combined_set_parameter and DC_peak_offset is provided,
                the offsets are zeroed after the scan.
                Otherwise, sweep_parameter is set to optimum
            min_voltage: Minimum voltage that Coulomb peak must have.
                If not satisfied, measurement has failed, and system is reset to
                initial value.
            sweep_settings: Kwargs passed to sweep_parameter to determine sweep vals.
                Can for instance be `window` and `num`
            **kwargs: kwargs passed to MeasurementParameter
        """

        if acquisition_parameter is None:
            acquisition_parameter = DCParameter()

        self.sweep_parameter = sweep_parameter
        self.sweep_settings = sweep_settings or {}
        self.min_voltage = min_voltage

        self.combined_set_parameter = combined_set_parameter
        self.DC_peak_offset = DC_peak_offset

        self.tune_to_peak = tune_to_peak
        self.interpolate = interpolate

        self.continuous = True

        self.results = {}

        super().__init__(name=name,
                         names=['peak_optimum', 'peak_offset',
                                'max_voltage', 'DC_voltage'],
                         units=['V', 'V', 'V', 'V'],
                         shapes=((), (), (), ()),
                         acquisition_parameter=acquisition_parameter,
                         wrap_set= False,
                         **kwargs)

        self._meta_attrs += ['min_voltage']

    @property_ignore_setter
    def names(self):
        return [f'{self.sweep_parameter.name}_optimum',
                f'{self.sweep_parameter.name}_offset',
                'max_voltage', 'DC_voltage']

    @property_ignore_setter
    def shapes(self):
        try:
            sweep_vals = self.calculate_sweep_vals()
            return ((), (), (), (len(sweep_vals),))
        except:
            return ((), (), (), ())

    def create_loop(self, sweep_vals):
        if sweep_vals is None:
            raise RuntimeError('Must define self.sweep_vals')

        loop = Loop(sweep_vals).each(self.acquisition_parameter)
        return loop

    def calculate_sweep_vals(self):
        return self.sweep_parameter.sweep(**self.sweep_settings)

    @clear_single_settings
    def get_raw(self):
        if self.DC_peak_offset is not None:
            if self.combined_set_parameter is None:
                raise RuntimeError('Must specify combined_set_parameter')

            # Add DC offset
            self.combined_set_parameter(self.DC_peak_offset)

        # Calculate sweep vals
        initial_set_val = self.sweep_parameter()
        sweep_vals = self.calculate_sweep_vals()

        self.loop = self.create_loop(sweep_vals=sweep_vals)

        self.data = self.loop.get_data_set(name=self.name,
                                           location=self.loc_provider)

        # Perform measurement
        if (self.layout.pulse_sequence != self.acquisition_parameter.pulse_sequence
                or self.layout.samples() != self.acquisition_parameter.samples):
            self.acquisition_parameter.setup()
        try:
            self.loop.run(set_active=False, quiet=True,
                          stop=not self.continuous)
        except:
            # Error occurred, reset to initial values and raise error
            if self.DC_peak_offset is None:
                self.sweep_parameter(initial_set_val)
            else:
                self.combined_set_parameter(0)
            raise

        # Analyse measurement results
        if self.min_voltage is not None and np.max(
                self.data.DC_voltage) < self.min_voltage:
            # Could not find coulomb peak
            self.results[self.names[0]] = np.nan
            self.results[self.names[1]] = np.nan

            # Tune back to original position
            if self.DC_peak_offset is None:
                self.sweep_parameter(initial_set_val)
            else:
                self.combined_set_parameter(0)
        else:
            # Found coulomb peak
            # Perform some smoothing of data
            self.smoothed_arr = self.data.DC_voltage.smooth(5)
            if self.interpolate:
                self.interpolation = interp1d(sweep_vals, self.smoothed_arr,
                                              'cubic')
                self.interpolation_x_vals = np.linspace(sweep_vals[0],
                                                        sweep_vals[-1], 100)
                self.interpolation_y_vals = self.interpolation(
                    self.interpolation_x_vals)
                max_idx = np.argmax(self.interpolation_y_vals)
                max_set_val = self.interpolation_x_vals[max_idx]
            else:
                max_idx = np.argmax(self.smoothed_arr)
                max_set_val = sweep_vals[max_idx]
            self.results[self.names[0]] = max_set_val
            self.results[self.names[1]] = max_set_val - initial_set_val

            if self.tune_to_peak:
                # Update parameter values
                if self.DC_peak_offset is None:
                    self.sweep_parameter(max_set_val)
                else:
                    # Update combined_set_parameter offset
                    peak_offset = max_set_val - initial_set_val

                    sweep_parameter_idx = next(
                        index for index, item in
                        enumerate(self.combined_set_parameter.parameters)
                        if item is self.sweep_parameter)

                    self.combined_set_parameter.offsets[
                        sweep_parameter_idx] += peak_offset

                    self.combined_set_parameter(0)
            elif self.DC_peak_offset is not None:
                self.combined_set_parameter(0)

        self.results['max_voltage'] = np.max(self.data.DC_voltage)
        self.results['DC_voltage'] = self.data.DC_voltage

        return [self.results[name] for name in self.names]


class MeasureFlipNucleusParameter(MeasurementParameter):
    def __init__(self,
                 name='measure_flip_nucleus',
                 measure_nucleus_parameter=None,
                 flip_nucleus_parameter=None,
                 max_attempts=3,
                 target_state=None,
                 final_measure=False,
                 silent=True,
                 condition=None,
                 **kwargs):

        self.measure_nucleus_parameter = measure_nucleus_parameter
        self.flip_nucleus_parameter = flip_nucleus_parameter
        self.max_attempts = max_attempts
        self.target_state = target_state
        self.final_measure = final_measure
        self.silent = silent
        self.condition = condition

        self.results = {}


        super().__init__(name=name, names=self.names, shapes=self.shapes,
                         wrap_set=False, **kwargs)

    @property_ignore_setter
    def names(self):
        names = ['nuclear_states',
                 'nucleus_up_proportions',
                 'nucleus_pulse_sequences',
                 'nucleus_flip_success']
        if self.final_measure:
            names.append('final state')
        return names

    @property_ignore_setter
    def shapes(self):
        shapes = [(self.max_attempts,),
                  (self.max_attempts, len(self.measure_nucleus_parameter.frequency_vals)),
                  (self.max_attempts,),
                  ()]
        if self.final_measure:
            shapes.append(())
        return tuple(shapes)

    @clear_single_settings
    def get_raw(self):
        if self.target_state is None:
            raise SyntaxError('Must provide MeasureFlipNucleusParameter.target_state')

        nuclear_states = np.nan * np.ones(self.max_attempts)
        nucleus_up_proportions = np.nan * np.ones((
            self.max_attempts, len(self.measure_nucleus_parameter.frequency_vals)))
        nucleus_pulse_sequences = np.nan * np.ones(self.max_attempts)
        nucleus_flip_success = False
        nucleus_state = np.nan

        if self.condition is None or self.condition():

            for k in range(self.max_attempts):
                self.measure_nucleus_parameter()
                results = self.measure_nucleus_parameter.results
                nucleus_state = nuclear_states[k] = results['nucleus_state']
                nucleus_up_proportions[k] = next(val for key, val in results.items()
                                                 if key.startswith('nucleus_up_proportion'))


                if not results['found_nucleus_state']:
                    logger.info('Could not determine nucleus state. perhaps in a '
                                'state for which ESR frequency is not known')
                    continue
                if nucleus_state == self.target_state:
                    logger.info(f'Nucleus is in target state {self.target_state}')
                    nucleus_flip_success = True
                    break
                else:
                    logger.info(f'Flipping nucleus from {nucleus_state} '
                                f'to {self.target_state}')
                    self.flip_nucleus_parameter(nucleus_state, self.target_state)
            else:
                nucleus_flip_success = False

        if not self.silent:
            self.print_results()

        self.results = {'nuclear_states': nuclear_states,
                        'nucleus_up_proportions': nucleus_up_proportions,
                        'nucleus_pulse_sequences': nucleus_pulse_sequences,
                        'nucleus_flip_success': nucleus_flip_success,
                        'final_state': nucleus_state}
        return [self.results[name] for name in self.names]

    def set(self, target_state, **kwargs):
        self.single_settings(target_state=target_state, **kwargs)
        return self()


class MeasureNucleusParameter(MeasurementParameter):
    def __init__(self,
                 name='measure_nucleus',
                 discriminant='contrast_ESR',
                 acquisition_parameter=None,
                 frequency_set_parameter=None,
                 frequency_vals=None,
                 frequency_order='descending',
                 threshold=0.5,
                 samples=None,
                 break_if_satisfied=True):
        super().__init__(name=name, acquisition_parameter=acquisition_parameter,
                         names=['found_nucleus_state',
                                'nucleus_state',
                                'max_nucleus_' + discriminant,
                                'nucleus_' + discriminant,
                                'average_' + discriminant],
                         discriminant=discriminant,
                         shapes=((), (), (), (), ()),
                         wrap_set=False)

        self.frequency_set_parameter = frequency_set_parameter
        self.frequency_vals = frequency_vals
        self.frequency_order = frequency_order

        self.threshold = threshold
        self.break_if_satisfied = break_if_satisfied
        self.samples = samples
        self.continuous = False
        self.silent = True

        self.results = {}
        self.loop = None
        self.data = None

    @property_ignore_setter
    def labels(self):
        return [name.replace('_', ' ').capitalize() for name in self.names]

    @property_ignore_setter
    def shapes(self):
        return ((), (), (), (len(self.frequency_vals), ), ())

    @property
    def frequency_vals(self):
        if self._frequency_vals is not None:
            frequency_vals = self._frequency_vals
        else:
            frequency_vals = {int(key): val for key, val
                              in config[f'environment:properties.ESR_vals'].items()}
        return frequency_vals

    @frequency_vals.setter
    def frequency_vals(self, vals):
        if vals is not None and not isinstance(vals, dict):
            raise SyntaxError('frequency_vals must be None or dict')
        else:
            self._frequency_vals = vals

    @property
    def frequency_order(self):
        frequency_vals = self.frequency_vals

        if self._frequency_order == 'descending':
            frequency_order = sorted(frequency_vals.keys(), reverse=True)
        elif self._frequency_order == 'ascending':
            frequency_order = sorted(frequency_vals.keys(), reverse=False)
        else:
            # Frequency order is a list
            frequency_order = self._frequency_order

        return frequency_order

    @frequency_order.setter
    def frequency_order(self, order):
        if isinstance(order, str) and order in ['ascending', 'descending']:
            self._frequency_order = order
        elif isinstance(order, Iterable):
            self._frequency_order = list(order)
        else:
            raise TypeError('frequency order must either be an iterable, '
                            'or `ascending` or `descending`')

    @property
    def frequency_vals_sorted(self):
        frequency_vals = self.frequency_vals
        return [frequency_vals[key] for key in self.frequency_order]

    def create_loop(self):
        def above_threshold():
            val = self.acquisition_parameter.results[self.discriminant]
            is_above_threshold = val > self.threshold
            logger.debug(f'{self.discriminant} {val} > {self.threshold}: {is_above_threshold}')
            return is_above_threshold

        if self.acquisition_parameter is None:
            raise RuntimeError('Must specify acquisition_parameter')
        if self.frequency_set_parameter is None:
            raise RuntimeError('Must specify frequency_set_parameter')

        actions = [self.acquisition_parameter]
        if self.break_if_satisfied:
            actions.append(BreakIf(above_threshold))

        loop = Loop(
            self.frequency_set_parameter[self.frequency_vals_sorted]
        ).each(
            *actions
        )
        return loop

    @clear_single_settings
    def get_raw(self):

        self.loop = self.create_loop()
        self.data = self.loop.get_data_set(name=self.name,
                                           location=self.loc_provider)

        temporary_settings = {"continuous": True,
                              "base_folder": self.data.location,
                              "subfolder": 'traces'}
        if self.samples is not None:
            temporary_settings['samples'] = self.samples

        self.acquisition_parameter.temporary_settings(**temporary_settings)

        if (self.layout.pulse_sequence != self.acquisition_parameter.pulse_sequence
            or self.layout.samples() != self.acquisition_parameter.samples):
            self.acquisition_parameter.setup()

        self.loop.run(set_active=False, quiet=True,
                      stop=not self.continuous)

        discriminant_vals = getattr(self.data, self.discriminant).ndarray
        with np.errstate(invalid='ignore'):
            if np.nansum(discriminant_vals >= self.threshold) == 1:
                # Successfully found nucleus state
                frequency_idx = np.nanargmax(discriminant_vals)
                self.results['found_nucleus_state'] = True
                self.results['nucleus_state'] = int(
                    self.frequency_order[frequency_idx])
            else:
                # Could not localize nucleus state
                self.results['found_nucleus_state'] = False
                self.results['nucleus_state'] = np.nan
            self.results['max_nucleus_' + self.discriminant] = np.nanmax(discriminant_vals)
            self.results['nucleus_' + self.discriminant] = discriminant_vals
            self.results['average_' + self.discriminant] = np.nanmean(discriminant_vals)

        if not self.silent:
            self.print_results()

        return [self.results[name] for name in self.names]

    def set(self, **kwargs):
        # Set single settings
        return self.single_settings(**kwargs)()


class DCMultisweepParameter(MeasurementParameter):
    formatter = None

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
            self.data = self.loop.run(name=f'multi_2D_scan',
                                      set_active=False,
                                      formatter=self.formatter,
                                      save_metadata=False)
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
            names=[name + '_msmts', 'optimal_set_vals', self.discriminant],
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
        return [(), (len(self.set_vals),), (len(self.set_vals),)]

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
