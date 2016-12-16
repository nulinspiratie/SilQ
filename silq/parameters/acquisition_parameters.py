from time import sleep
import numpy as np
import logging

from qcodes import config
from qcodes.instrument.parameter import Parameter
from qcodes.data import hdf5_format, io

from silq.pulses import PulseSequence, DCPulse, FrequencyRampPulse, \
    SteeredInitialization
from silq.analysis import analysis
from silq.tools import data_tools, general_tools

h5fmt = hdf5_format.HDF5Format()
properties_config = config['user'].get('properties', {})


class MeasurementParameter(Parameter):
    data_manager = None
    layout = None
    formatter = h5fmt

    def __init__(self, mode=None, average_mode='none', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.mode_str = '' if mode is None else '_{}'.format(mode)
        # TODO add mode to name
        self.pulse_kwargs = {'mode': self.mode}

        self.pulse_sequence = PulseSequence()
        self.silent = True
        self.average_mode = average_mode
        self.save_traces = False

        self.samples = None
        self.t_read = None
        self.t_skip = None
        self.trace_segments = None
        self.data = None
        self.data_set = None
        self.results = None

        # Change attribute data_manager from class attribute to instance
        # attribute. This is necessary to ensure that the data_manager is
        # passed along when the parameter is spawned from a new process
        self.data_manager = self.data_manager

        self._meta_attrs.extend(['pulse_sequence'])

    def __repr__(self):
        return '{} measurement parameter\n{}'.format(self.name,
                                                     self.pulse_sequence)

    def __getattribute__(self, item):
        """
        Used when requesting an attribute. If the attribute is explicitly set to
        None, it will check the config if the item exists.
        Args:
            item: Attribute to be retrieved

        Returns:

        """
        value = object.__getattribute__(self, item)
        if value is not None:
            return value

        value = self._attribute_from_config(item)
        return value

    @property
    def sample_rate(self):
        """ Acquisition sample rate """
        return self.layout.sample_rate()

    @property
    def pulse_pts(self):
        """ Number of points in the trace of each pulse"""
        return {pulse.name: int(round(pulse.duration / 1e3 * self.sample_rate))
                for pulse in self.pulse_sequence}

    @property
    def start_idx(self):
        return round(self.t_skip * 1e-3 * self.sample_rate)

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

    def update_settings(self, **kwargs):
        """
        Sets up the meta properties of a measurement parameter
        """
        for item, value in kwargs:
            if hasattr(self, item):
                setattr(self, item, value)
            else:
                raise ValueError('Setting {} not found'.format(item))

    def segment_trace(self, trace):
        trace_segments = {}
        idx = 0
        for pulse in self.pulse_sequence:
            if not pulse.acquire:
                continue
            trace_segments[pulse.name] = \
                trace[:, idx:idx + self.pulse_pts[pulse.name]]
            idx += self.pulse_pts[pulse.name]
        return trace_segments

    def store_traces(self, traces_dict, subfolder=None):
        # Store raw traces
        # Pause until data_manager is done measuring
        while self.data_manager.ask('get_measuring'):
            sleep(0.01)

        for traces_name, traces in traces_dict.items():
            self.data_set = data_tools.create_raw_data_set(
                name=traces_name,
                data_manager=self.data_manager,
                shape=traces.shape,
                formatter=self.formatter,
                subfolder=subfolder)
            data_tools.store_data(data_manager=self.data_manager,
                                  result=traces)

    def print_results(self):
        if self.names is not None:
            for name, result in zip(self.names, self.results):
                print('{}: {:.3f}'.format(name, result))
        else:
            print('{}: {:.3f}'.format(self.name, self.results))

    def setup(self, **kwargs):
        self.layout.stop()
        self.layout.target_pulse_sequence(self.pulse_sequence)
        self.layout.setup(samples=self.samples, average_mode=self.average_mode,
                          **kwargs)

    def acquire(self, segment_traces=True, **kwargs):
        # Perform acquisition
        self.data = self.layout.do_acquisition(return_dict=True,
                                               **kwargs)
        if segment_traces:
            self.trace_segments = {
                ch_label: self.segment_trace(trace)
                for ch_label, trace in self.data['acquisition_traces'].items()}


class DC_Parameter(MeasurementParameter):
    # TODO implement continuous acquisition
    def __init__(self, **kwargs):
        super().__init__(name='DC_voltage',
                         label='DC voltage',
                         units='V',
                         average_mode='point',
                         snapshot_value=False,
                         **kwargs)

        self.samples = 1

        self.pulse_sequence.add([
            DCPulse(name='read', acquire=True, **self.pulse_kwargs),
            DCPulse(name='final', **self.pulse_kwargs)])

    def acquire(self, **kwargs):
        # Do not segment traces since we only receive a single value
        super().acquire(segment_traces=False)

    def get(self):
        self.acquire()
        return self.data['acquisition_traces']['output']


class EPR_Parameter(MeasurementParameter):
    def __init__(self, **kwargs):
        super().__init__(name='EPR',
                         label='Empty Plunge Read',
                         snapshot_value=False,
                         names=['contrast', 'dark_counts', 'voltage_difference',
                                'fidelity_empty', 'fidelity_load'],
                         **kwargs)

        self.pulse_sequence.add([
            DCPulse('empty', acquire=True, **self.pulse_kwargs),
            DCPulse('plunge', acquire=True, **self.pulse_kwargs),
            DCPulse('read', acquire=True, **self.pulse_kwargs),
            DCPulse('final', **self.pulse_kwargs)])

        self.analysis = analysis.analyse_EPR

    def get(self):
        self.acquire()

        fidelities = self.analysis(trace_segments=self.trace_segments['output'],
                                   sample_rate=self.sample_rate,
                                   t_skip=self.t_skip, t_read=self.t_read)
        self.results = [fidelities[name] for name in self.names]

        if self.save_traces:
            saved_traces = {
                'acquisition_traces': self.data['acquisition_traces']['output']}
            self.store_traces(saved_traces)

        if not self.silent:
            self.print_results()

        return self.results


class Adiabatic_Parameter(MeasurementParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to perform an adiabatic sweep
        """
        super().__init__(name='adiabatic',
                         label='Adiabatic',
                         snapshot_value=False,
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         **kwargs)

        self.pulse_sequence.add([
            SteeredInitialization('steered_initialization',
                                  enabled=False, **self.pulse_kwargs),
            DCPulse('plunge', acquire=True, **self.pulse_kwargs),
            DCPulse('read', acquire=True, **self.pulse_kwargs),
            DCPulse('final', **self.pulse_kwargs),
            FrequencyRampPulse('adiabatic', **self.pulse_kwargs)])

        # Disable previous pulse for adiabatic pulse, since it would
        # otherwise be after 'final' pulse
        self.pulse_sequence['adiabatic'].previous_pulse = None
        self.pulse_sequence.sort()

        self.analysis = analysis.analyse_PR

    @property
    def frequency(self):
        return self.pulse_sequence['adiabatic'].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['adiabatic'].frequency = frequency

    def acquire(self, **kwargs):
        super().acquire(return_initialization_traces=self.pulse_sequence[
            'steered_initialization'].enabled, **kwargs)

    def get(self):
        self.acquire()

        fidelities = self.analysis(trace_segments=self.trace_segments['output'],
                                   sample_rate=self.sample_rate,
                                   t_skip=self.t_skip, t_read=self.t_read)
        self.results = [fidelities[name] for name in self.names]

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            saved_traces = {
                'acquisition_traces': self.data['acquisition_traces']['output']}
            if 'initialization_traces' in self.data:
                saved_traces['initialization'] = \
                    self.data['initialization_traces']
            if 'post_initialization_traces' in self.data:
                saved_traces['post_initialization_output'] = \
                    self.data['post_initialization_traces']['output']
            self.store_traces(saved_traces, subfolder=self.subfolder)

        if not self.silent:
            self.print_results()

        return self.results

    def set(self, frequency):
        # Change center frequency
        self.frequency = frequency
        self.setup()


class SelectFrequency_Parameter(MeasurementParameter):
    def __init__(self, threshold=0.5, discriminant='contrast', **kwargs):
        self.frequencies = None
        self.frequency = None

        names = [discriminant + spin_state for spin_state in self.spin_states]
        if 'mode' in kwargs:
            names.append('frequency_{}'.format(kwargs['mode']))
        else:
            names.append('frequency')

        super().__init__(name='select_frequency',
                         label='Select frequency',
                         snapshot_value=False,
                         names=names,
                         **kwargs)

        self.update_frequency = True
        self.threshold = threshold
        self.discriminant = discriminant

        self.measure_parameter = Adiabatic_Parameter(**kwargs)

        self._meta_attrs.extend(['frequencies', 'frequency', 'update_frequency',
                                 'threshold', 'discriminant'])

    @property
    def spin_states(self):
        spin_states_unsorted = self.frequencies.values()
        return sorted(spin_states_unsorted)

    @property
    def discriminant_idx(self):
        return self.measure_parameter.names.index(self.discriminant)

    def get(self):
        self.results = []
        # Perform measurement for all frequencies
        for spin_state in self.spin_states:
            # Set adiabatic frequency
            self.measure_parameter(self.frequencies[spin_state])
            fidelities = self.measure_parameter()

            # Only add dark counts and contrast
            self.results.append(fidelities[self.discriminant_idx])

            # Store raw traces if self.save_traces is True
            if self.save_traces:
                saved_traces = {
                    'acquisition_traces': self.data['acquisition_traces'][
                        'output']}
                if 'initialization_traces' in self.data:
                    saved_traces['initialization'] = \
                        self.data['initialization_traces']
                if 'post_initialization_traces' in self.data:
                    saved_traces['post_initialization_output'] = \
                        self.data['post_initialization_traces']['output']
                self.store_traces(saved_traces, subfolder='{}_{}'.format(
                    self.subfolder, spin_state))

        optimal_idx = np.argmax(self.results)
        optimal_spin_state = self.spin_states[optimal_idx]

        frequency = self.frequencies[optimal_spin_state]
        self.results += [frequency]

        # Print results
        if not self.silent:
            self.print_results()

        if self.update_frequency and max(self.results) > self.threshold:
            properties_config['frequency' + self.mode_str] = frequency
        elif not self.silent:
            logging.warning("Could not find frequency with high enough "
                            "contrast")

        return self.results


class T1_Parameter(MeasurementParameter):
    def __init__(self, **kwargs):
        super().__init__(name='T1_wait_time',
                         label='T1 wait time',
                         snapshot_value=False,
                         names=['T1_up_proportion', 'T1_num_traces'],
                         **kwargs)
        self.pulse_sequence.add([
            SteeredInitialization('steered_initialization',
                                  enabled=False, **self.pulse_kwargs),
            DCPulse('plunge', **self.pulse_kwargs),
            DCPulse('read', acquire=True, **self.pulse_kwargs),
            DCPulse('final', **self.pulse_kwargs),
            FrequencyRampPulse('adiabatic', **self.pulse_kwargs)])
        # Disable previous pulse for adiabatic pulse, since it would
        # otherwise be after 'final' pulse
        self.pulse_sequence['adiabatic'].previous_pulse = None
        self.pulse_sequence.sort()

        self.analysis = analysis.analyse_read

        self.readout_threshold_voltage = None

        self._meta_attrs.append('readout_threshold_voltage')

    @property
    def wait_time(self):
        return self.pulse_sequence['plunge'].duration

    def acquire(self, **kwargs):
        super().acquire(return_initialization_traces=self.pulse_sequence[
            'steered_initialization'].enabled, **kwargs)

    def get(self):
        self.acquire()

        # Analysis
        fidelities = self.analysis(traces=self.trace_segments['output']['read'],
                                   threshold_voltage=
                                   self.readout_threshold_voltage,
                                   start_idx=self.start_idx)
        self.results = [fidelities[name] for name in self.names]

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            saved_traces = {'acquisition_traces':
                                self.data['acquisition_traces']['output']}
            if 'initialization_traces' in self.data:
                saved_traces['initialization'] = \
                    self.data['initialization_traces']
            if 'post_initialization_traces' in self.data:
                saved_traces['post_initialization_output'] = \
                    self.data['post_initialization_traces']['output']
            self.store_traces(saved_traces,
                              name=self.plunge_duration,
                              subfolder=self.subfolder)

        if not self.silent:
            self.print_results()

        return self.results

    def set(self, wait_time):
        self.pulse_sequence['plunge'].duration = wait_time
        self.setup()


class DarkCounts_Parameter(MeasurementParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to perform an adiabatic sweep
        """
        super().__init__(name='dark_counts',
                         label='Dark counts',
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add([
            SteeredInitialization('steered_initialization',
                                  enabled=True, **self.pulse_kwargs),
            DCPulse('read', acquire=True, **self.pulse_kwargs)])

        self.analysis = analysis.analyse_read

        self.readout_threshold_voltage = None

        self._meta_attrs.append('readout_threshold_voltage')

    def acquire(self, **kwargs):
        super().acquire(return_initialization_traces=self.pulse_sequence[
            'steered_initialization'].enabled, **kwargs)

    def get(self):
        self.acquire()

        fidelities = self.analysis(traces=self.trace_segments['output']['read'],
                                   threshold_voltage=
                                   self.readout_threshold_voltage,
                                   start_idx=self.start_idx)
        self.results = fidelities['up_proportion']

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            saved_traces = {'acquisition_traces':
                                self.data['acquisition_traces']['output']}
            if 'initialization_traces' in self.data:
                saved_traces['initialization'] = \
                    self.data['initialization_traces']
            if 'post_initialization_traces' in self.data:
                saved_traces['post_initialization_output'] = \
                    self.data['post_initialization_traces']['output']
            self.store_traces(saved_traces, subfolder=self.subfolder)

        if not self.silent:
            self.print_results()

        return self.results

    def set(self, frequency):
        # Change center frequency
        self.frequency = frequency
        self.setup()


class VariableRead_Parameter(MeasurementParameter):
    def __init__(self, **kwargs):
        super().__init__(name='variable_read_voltage',
                         label='Variable read voltage',
                         snapshot_value=False,
                         **kwargs)
        empty_pulse = DCPulse(name='empty', acquire=True, **self.pulse_kwargs)
        load_pulse = DCPulse(name='plunge', acquire=True, **self.pulse_kwargs)
        read_pulse = DCPulse(name='read', acquire=True, **self.pulse_kwargs)
        final_pulse = DCPulse(name='final', **self.pulse_kwargs)

        self.samples = 50

        pulses = [empty_pulse, load_pulse, read_pulse, final_pulse]
        self.pulse_sequence.add(pulses)

    def setup(self, samples=None, **kwargs):
        if samples:
            self.samples = samples

        self.layout.target_pulse_sequence(self.pulse_sequence)

        self.layout.setup(samples=self.samples, average_mode='trace')

        # Setup parameter metadata
        self.names = self.layout.acquisition.names
        self.labels = self.layout.acquisition.labels
        self.shapes = self.layout.acquisition.shapes

        super().setup(**kwargs)

    def get(self):
        self.traces = self.layout.acquisition()
        return self.traces

    def set(self, read_voltage):
        # Change read stage voltage.
        self.read_voltage = read_voltage
        self.pulse_sequence['read'].voltage = self.read_voltage

        self.setup()


class AutoCalibration_Parameter(Parameter):
    def __init__(self, name, set_parameters, measure_parameter,
                 calibration_operations, key, conditions=None, **kwargs):
        """

        Args:
            name:
            set_parameters:
            measure_parameter:
            calibration_operations:
            key:
            conditions: Must be of one of the following forms
                {'mode': 'measure'}
                {'mode': '1D_scan', 'span', 'set_points', 'set_parameter',
                 'center_val'(optional)
            **kwargs:
        """
        super().__init__(name, **kwargs)

        self.key = key
        self.conditions = conditions
        self.calibration_operations = calibration_operations

        self.set_parameters = {p.name: p for p in set_parameters}
        self.measure_parameter = measure_parameter

        self.names = ['success', 'optimal_set_val', self.key]
        self.labels = self.names
        self._meta_attrs.extend(['measure_parameter_name', 'conditions',
                                 'calibration_operations', 'key',
                                 'set_vals_1D', 'measure_vals_1D'])

    @property
    def measure_parameter_name(self):
        return self.measure_parameter.name

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

    def get(self):
        self.loop_parameters = []
        self.datasets = []

        if hasattr(self.measure_parameter, 'setup'):
            self.measure_parameter.setup()

        for k, calibration_operation in enumerate(self.calibration_operations):
            if calibration_operation['mode'] == 'measure':
                dims = (1)
                self.set_vals = [None]
                loop_parameter = Loop0D_Parameter(
                    name='calibration_0D',
                    measure_parameter=self.measure_parameter)

            elif calibration_operation['mode'] == '1D_scan':
                # Setup set vals
                set_parameter_name = calibration_operation['set_parameter']
                set_parameter = self.set_parameters[set_parameter_name]

                center_val = calibration_operation.get(
                    'center_val', set_parameter())
                span = calibration_operation['span']
                set_points = calibration_operation['set_points']
                self.set_vals = list(np.linspace(center_val - span / 2,
                                                 center_val + span / 2,
                                                 set_points))
                dims = (set_points)
                # Extract set_parameter
                loop_parameter = Loop1D_Parameter(
                    name='calibration_1D',
                    set_parameter=set_parameter,
                    measure_parameter=self.measure_parameter,
                    set_vals=self.set_vals)
            else:
                raise ValueError("Calibration mode not implemented")

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
            return cal_success, optimal_set_val, optimal_get_val
        else:
            return cal_success, 1, optimal_get_val
