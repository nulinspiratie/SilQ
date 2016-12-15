from time import sleep
import numpy as np
from collections import OrderedDict
import logging

import qcodes as qc
from qcodes import config
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data import hdf5_format, io
h5fmt = hdf5_format.HDF5Format()

from silq.pulses import PulseSequence, DCPulse, FrequencyRampPulse, \
    SteeredInitialization
from silq.analysis import analysis
from silq.tools import data_tools, general_tools

properties_config = config['user'].get('properties', {})


class MeasurementParameter(Parameter):
    data_manager = None
    def __init__(self, layout, formatter=h5fmt, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.mode_str = '' if mode is None else '_' + mode
        self.pulse_kwargs = {'mode': self.mode}

        self.layout = layout
        self.pulse_sequence = PulseSequence()

        self.formatter = formatter

        self.silent = True
        self.save_traces = False

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

    def setup(self, save_traces=None, formatter=None, silent=None,
              **kwargs):
        """
        Sets up the meta properties of a measurement parameter.
        Args:
            formatter:
            **kwargs:

        Returns:

        """
        if save_traces is not None:
            self.save_traces = save_traces
        if silent is not None:
            self.silent = silent

        if formatter is not None:
            self.formatter = formatter

        sample_rate = self.layout.sample_rate()
        self.pts = {pulse.name: int(round(pulse.duration/ 1e3 * sample_rate))
                    for pulse in self.pulse_sequence}

    def segment_trace(self, trace):
        trace_segments = {}
        idx = 0
        for pulse in self.pulse_sequence:
            if not pulse.acquire:
                continue
            trace_segments[pulse.name] = \
                trace[:, idx:idx + self.pts[pulse.name]]
            idx += self.pts[pulse.name]
        return trace_segments

    def store_traces(self, traces_dict, name=None, subfolder=None):
        # Store raw traces
        if name is None:
            name = self.name

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


class DC_Parameter(MeasurementParameter):

    def __init__(self, layout, **kwargs):
        super().__init__(name='DC_voltage',
                         label='DC voltage',
                         units='V',
                         layout=layout,
                         snapshot_value=False,
                         **kwargs)

        read_pulse = DCPulse(name='read', acquire=True, **self.pulse_kwargs)
        final_pulse = DCPulse(name='final', **self.pulse_kwargs)
        pulses = [read_pulse, final_pulse]
        self.pulse_sequence.add(pulses)

    def setup(self, duration=20):
        # Stop instruments in case they were already running
        self.layout.stop()
        self.layout.target_pulse_sequence(self.pulse_sequence)

        self.layout.setup(samples=1, average_mode='point')

        self.layout.start()

    def get(self):
        signal = self.layout.do_acquisition(start=False, stop=False,
                                            return_dict=True)
        return signal['acquisition_traces']['output']


class EPR_Parameter(MeasurementParameter):

    def __init__(self, layout, **kwargs):
        super().__init__(name='EPR',
                         label='Empty Plunge Read',
                         layout=layout,
                         snapshot_value=False,
                         **kwargs)

        self.subfolder = 'EPR'

        empty_pulse = DCPulse(name='empty', t_start=0, acquire=True,
                              **self.pulse_kwargs)
        plunge_pulse = DCPulse(name='plunge', acquire=True, **self.pulse_kwargs)
        read_pulse = DCPulse(name='read', acquire=True, **self.pulse_kwargs)
        final_pulse = DCPulse(name='final', **self.pulse_kwargs)

        self.pulse_sequence.add(
            [empty_pulse, plunge_pulse, read_pulse, final_pulse])

        self.samples = 100
        self.t_skip = 0.1
        self.t_read = 20

        self.analysis = analysis.analyse_EPR

        # Setup parameter metadata
        self.names = ['contrast', 'dark_counts', 'voltage_difference',
                      'fidelity_empty', 'fidelity_load']
        self.labels = self.names
        self.shapes = ((), (), (), (), ())

        self._meta_attrs.extend(['t_skip', 't_read'])

    def setup(self, samples=None, t_skip=None, t_read=None, **kwargs):
        if samples is not None:
            self.samples = samples
        if t_skip is not None:
            self.t_skip = t_skip
        if t_read is not None:
            self.t_read = t_read

        self.layout.target_pulse_sequence(self.pulse_sequence)

        self.layout.setup(samples=self.samples, average_mode='none', **kwargs)

        self.analysis_settings = {'sample_rate': self.layout.sample_rate(),
                                  't_skip': self.t_skip,
                                  't_read': self.t_read}

        super().setup(**kwargs)

    def acquire(self):
        if 'steered_initialization' in self.pulse_sequence and \
                self.pulse_sequence['steered_initialization'].enabled:
            return_initialization_traces = True
        else:
            return_initialization_traces = False

        self.data = self.layout.do_acquisition(
            return_dict=True,
            return_initialization_traces=return_initialization_traces)

        self.trace_segments = {
            ch_label: self.segment_trace(trace)
            for ch_label, trace in self.data['acquisition_traces'].items()}

    def print_fidelities(self):
        for name, fidelity in zip(self.names, self.fidelities):
            print('{}: {:.3f}'.format(name, fidelity))

    def get(self):
        self.acquire()

        fidelities = self.analysis(
            trace_segments=self.trace_segments['output'],
            **self.analysis_settings)
        self.fidelities = [fidelities[name] for name in self.names]

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            saved_traces = {'acquisition_traces':
                                self.data['acquisition_traces']['output']}
            self.store_traces(saved_traces, subfolder=self.subfolder)

        # Print results
        if not self.silent:
            self.print_fidelities()

        return self.fidelities


class AdiabaticSweep_Parameter(EPR_Parameter):
    def __init__(self, layout, **kwargs):
        """
        Parameter used to perform an adiabatic sweep
        """
        super().__init__(layout=layout, **kwargs)

        self.name = 'adiabatic_sweep' + self.mode_str
        self.label = 'Adiabatic {} sweep center frequency'.format(self.mode)

        self.subfolder = 'adiabatic' + self.mode_str

        adiabatic_pulse = FrequencyRampPulse(
            name='adiabatic' + self.mode_str, ** self.pulse_kwargs)
        steered_initialization = SteeredInitialization(
            name='steered_initialization', enabled=False, **self.pulse_kwargs)

        self.pulse_sequence['empty'].enabled = False
        self.pulse_sequence.add([adiabatic_pulse, steered_initialization])
        # Disable previous pulse for adiabatic pulse, since it would
        # otherwise be after 'final' pulse
        self.pulse_sequence['adiabatic' + self.mode_str].previous_pulse = None
        self.pulse_sequence.sort()

        self.analysis = analysis.analyse_PR

        self.names = ['contrast', 'dark_counts', 'voltage_difference']
        self.labels = self.names
        self.shapes = ((), (), ())

    @property
    def frequency(self):
        return self.pulse_sequence['adiabatic' + self.mode_str].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['adiabatic' + self.mode_str].frequency = frequency

    def get(self):
        self.acquire()

        fidelities = self.analysis(
            trace_segments=self.trace_segments['output'],
            **self.analysis_settings)
        self.fidelities = [fidelities[name] for name in self.names]

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

        # Print results
        if not self.silent:
            self.print_fidelities()

        return self.fidelities

    def set(self, frequency):
        # Change center frequency
        self.frequency = frequency
        self.setup()


class SelectFrequency_Parameter(AdiabaticSweep_Parameter):
    def __init__(self, layout, threshold=0.5, **kwargs):
        super().__init__(layout=layout, **kwargs)
        self.name = 'select_frequency' + self.mode_str
        self.label = 'Select frequency{} adiabatic'.format(self.mode_str)

        self.subfolder = 'select_frequency' + self.mode_str

        self.frequencies = None
        self.frequency = None

        self.update = True
        self.threshold = threshold

        self.names = ['contrast_up', 'dark_counts_up',
                      'contrast_down', 'dark_counts_down',
                      'frequency' + self.mode_str]
        self.labels = self.names
        self.shapes = ((), (), (), (), ())

        self._meta_attrs.extend(['frequencies' + self.mode_str,
                                 'frequency' + self.mode_str])

    def setup(self, update=None, threshold=None, **kwargs):
        super().setup(**kwargs)

        if update is not None:
            self.update = update

        if threshold is not None:
            self.threshold = threshold

    def get(self):
        self.fidelities = []
        self.contrasts = []
        # Perform measurement for both adiabatic frequencies
        for spin_state in ['up', 'down']:
            frequency = self.frequencies[spin_state]

            # Set adiabatic sweep frequency
            self(frequency)
            self.acquire()

            fidelities = self.analysis(
                trace_segments=self.trace_segments['output'],
                **self.analysis_settings)
            # Only add dark counts and contrast
            self.fidelities += [fidelities['contrast'],
                                fidelities['dark_counts']]
            self.contrasts += [fidelities['contrast']]

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
                self.store_traces(saved_traces, subfolder='{}_{}'.format(
                    self.subfolder, spin_state))

        if self.contrasts[0] > self.contrasts[1]:
            # Nucleus is in the up state
            frequency = self.frequencies['up']
        else:
            # Nucleus is in the down state
            frequency = self.frequencies['down']
        self.fidelities += [frequency]

        # Print results
        if not self.silent:
            self.print_fidelities()

        if self.update and max(self.contrasts) > self.threshold:
            properties_config['frequency' + self.mode_str] = frequency
        elif not self.silent:
            logging.warning("Could not find frequency with high enough "
                            "contrast")

        return self.fidelities


class T1_Parameter(AdiabaticSweep_Parameter):

    def __init__(self, layout, **kwargs):
        super().__init__(layout=layout, **kwargs)
        self.name = 'T1_wait_time'
        self.label = 'T1_wait_time'

        self.subfolder = 'T1'

        self.pulse_sequence['empty'].acquire = False
        self.pulse_sequence['plunge'].acquire = False

        self.analysis = analysis.analyse_read

        self.readout_threshold_voltage = None

        self.names = ['up_proportion', 'num_traces']
        self.labels = self.names
        self.shapes = ((), ())

    @property
    def plunge_duration(self):
        return self.pulse_sequence['plunge'].duration

    def setup(self, **kwargs):
        super().setup(**kwargs)

        self.analysis_settings = {
            'threshold_voltage': self.readout_threshold_voltage,
            'start_idx': round(self.t_skip * 1e-3 * self.layout.sample_rate())}

    def get(self):
        self.acquire()

        # Analysis
        fidelities = self.analysis(traces=self.trace_segments['output']['read'],
                                   **self.analysis_settings)
        self.fidelities = [fidelities[name] for name in self.names]

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

        # Print results
        if not self.silent:
            self.print_fidelities()

        return self.fidelities

    def set(self, plunge_duration):
            self.pulse_sequence['plunge'].duration = plunge_duration
            self.subfolder = 'T1_{}'.format(int(self.plunge_duration))

            self.setup()


class dark_counts_parameter(T1_Parameter):

    def __init__(self, layout, **kwargs):
            super().__init__(layout=layout, **kwargs)
            self.name = 'dark_counts'
            self.label = 'Dark counts'

            self.subfolder = 'dark_counts'

            self.pulse_sequence.remove('adiabatic')
            self.pulse_sequence['steered_initialization'].enabled = True


class VariableRead_Parameter(MeasurementParameter):
    def __init__(self, layout, **kwargs):
        super().__init__(name='variable_read_voltage',
                         label='Variable read voltage',
                         layout=layout,
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


class Loop_Parameter(Parameter):
    def __init__(self, name, measure_parameter, **kwargs):
        super().__init__(name, **kwargs)
        self.measure_parameter = measure_parameter

        self.loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')
        self._meta_attrs.extend(['measure_parameter_name'])

    @property
    def measure_parameter_name(self):
        return self.measure_parameter.name

    @property
    def disk_io(self):
        return io.DiskIO(data_tools.get_latest_data_folder())


class Loop0D_Parameter(Loop_Parameter):
    def __init__(self, name, measure_parameter, **kwargs):
        super().__init__(name, measure_parameter=measure_parameter, **kwargs)

    def get(self):

        self.measurement = qc.Measure(self.measure_parameter)
        self.data = self.measurement.run(
            name='{}_{}'.format(self.name, self.measure_parameter_name),
            data_manager=False,
            io=self.disk_io, location=self.loc_provider)
        return self.data

class Loop1D_Parameter(Loop_Parameter):
    def __init__(self, name, set_parameter, measure_parameter, set_vals=None,
                 **kwargs):
        super().__init__(name, measure_parameter=measure_parameter, **kwargs)
        self.set_parameter = set_parameter
        self.set_vals = set_vals

        self._meta_attrs.extend(['set_parameter_name', 'set_vals'])

    @property
    def set_parameter_name(self):
        return self.set_parameter.name

    def get(self):
        # Set data saving parameters
        self.measurement = qc.Loop(self.set_parameter[self.set_vals]
                                  ).each(self.measure_parameter)
        self.data = self.measurement.run(
            name='{}_{}_{}'.format(self.name, self.set_parameter_name,
                                   self.measure_parameter_name),
            background=False, data_manager=False,
            io=self.disk_io, location=self.loc_provider)
        return self.data

    def set(self, val):
        self.set_vals = val


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