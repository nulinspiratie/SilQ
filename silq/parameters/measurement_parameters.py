from time import sleep
import numpy as np
from collections import OrderedDict

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

        self.print_results = False

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

    def setup(self, save_traces=False, formatter=None, print_results=False,
              **kwargs):
        """
        Sets up the meta properties of a measurement parameter.
        Note that for save and print_results, the default behaviour
        is False, and so if a Parameter performs a set operation which also
        performs a setup routine, these will have to be manually overridden.
        Args:
            formatter:
            print_results:
            **kwargs:

        Returns:

        """
        self.save_traces = save_traces
        self.print_results = print_results

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
        self.names = ['contrast', 'dark_counts', 'SNR',
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

        self.fidelities = self.analysis(
            trace_segments=self.trace_segments['output'],
            **self.analysis_settings)

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            saved_traces = {'acquisition_traces':
                                self.data['acquisition_traces']['output']}
            self.store_traces(saved_traces, subfolder=self.subfolder)

        # Print results
        if self.print_results:
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

        ESR_pulse = FrequencyRampPulse(
            name='adiabatic' + self.mode_str, ** self.pulse_kwargs)
        steered_initialization = SteeredInitialization(
            name='steered_initialization', enabled=False, **self.pulse_kwargs)

        self.pulse_sequence['empty'].enabled = False
        self.pulse_sequence.add([ESR_pulse, steered_initialization])
        # Disable previous pulse for adiabatic pulse, since it would
        # otherwise be after 'final' pulse
        self.pulse_sequence['adiabatic' + self.mode_str].previous_pulse = None
        self.pulse_sequence.sort()

        self.analysis = analysis.analyse_PR

        self.names = ['contrast', 'dark_counts', 'SNR']
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

        self.fidelities = self.analysis(
            trace_segments=self.trace_segments['output'],
            **self.analysis_settings)

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
        if self.print_results:
            self.print_fidelities()

        return self.fidelities

    def set(self, frequency):
        # Change center frequency
        self.frequency = frequency
        self.setup()


class FindESR_Parameter(AdiabaticSweep_Parameter):
    def __init__(self, layout, **kwargs):
        super().__init__(layout=layout, **kwargs)
        self.name = 'find_ESR_frequency'
        self.label = 'Find ESR frequency adiabatic'

        self.subfolder = 'find_ESR'

        self.frequencies_ESR = None
        self.frequency_ESR = None

        self.names = ['contrast_up', 'dark_counts_up',
                      'contrast_down', 'dark_counts_down',
                      'ESR_frequency']
        self.labels = self.names
        self.shapes = ((), (), (), (), ())

        self._meta_attrs.extend(['ESR_frequencies', 'ESR_frequency'])

    def get(self):
        self.fidelities = []

        # Perform measurement for both adiabatic frequencies
        for spin_state in ['up', 'down']:
            frequency = self.frequencies_ESR[spin_state]

            # Set adiabatic sweep frequency
            self(frequency)
            self.acquire()

            fidelities = self.analysis(
                trace_segments=self.trace_segments['output'],
                **self.analysis_settings)
            # Only add dark counts and contrast
            self.fidelities += [fidelities[0], fidelities[1]]

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

        if self.fidelities[1] > self.fidelities[3]:
            # Nucleus is in the up state
            frequency_ESR = self.frequencies_ESR['up']
        else:
            # Nucleus is in the down state
            frequency_ESR = self.frequencies_ESR['down']
        self.fidelities += [frequency_ESR]

        # Print results
        if self.print_results:
            self.print_fidelities()

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
        up_proportion, num_traces_loaded, _ = self.analysis(
            traces=self.trace_segments['output']['read'],
            **self.analysis_settings)
        self.fidelities = (up_proportion, num_traces_loaded)

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
        if self.print_results:
            self.print_fidelities()

        return self.fidelities

    def set(self, plunge_duration):
            self.pulse_sequence['plunge'].duration = plunge_duration
            self.subfolder = 'T1_{}'.format(int(self.plunge_duration))

            self.setup(save_traces=self.save_traces)


class dark_counts_parameter(T1_Parameter):

    def __init__(self, layout, **kwargs):
            super().__init__(layout=layout, **kwargs)
            self.name = 'dark_counts'
            self.label = 'Dark counts'

            self.subfolder = 'dark_counts'

            self.pulse_sequence['adiabatic'].enabled = False
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


class Measure1D_Parameter(Parameter):
    def __init__(self, name, set_parameter, measure_parameter,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.set_parameter = set_parameter
        self.set_parameter_name = set_parameter.name
        self.measure_parameter = measure_parameter
        self.measure_parameter_name = measure_parameter.name
        self.set_vals = None

        self._meta_attrs.extend(['measure_parameter_name', 'set_vals'])

    def get(self):
        # Set data saving parameters
        disk_io = io.DiskIO(data_tools.get_latest_data_folder())
        loc_provider = qc.data.location.FormatLocation(
            fmt='#{counter}_{name}_{time}')

        self.loop = qc.Loop(self.set_parameter[self.set_vals]
                            ).each(self.measure_parameter)
        self.data = self.loop.run(
            name='1D_scan_{}_{}'.format(self.set_parameter_name,
                                        self.measure_parameter_name),
            background=False, data_manager=False,
            io=disk_io, location=loc_provider)
        return self.data

    def set(self, val):
        self.set_vals = val


# class AutoCalibration_Parameter(Parameter):
#     def __init__(self, name, set_parameters, measure_parameter,
#                  calibration_operations, key, constraints, **kwargs):
#         super().__init__(name, **kwargs)
#
#         self.key = key
#         self.constraints = constraints
#         self.calibration_operations = calibration_operations
#
#         self.set_parameters = {p.name: p for p in set_parameters}
#         self.measure_parameter = measure_parameter
#         self.measure_parameter_name = measure_parameter.name
#
#
#         self.cal1D_parameter = Measure1D_Parameter(
#             name='calibration_1D_{}_{}'.format(self.set_parameter_name,
#                                                self.measure_parameter_name),
#             measure_parameter=measure_parameter)
#
#         # TODO update
#         self._meta_attrs.extend(['set_parameter_name', 'measure_parameter_name',
#                                  'key', 'update', 'span', 'set_points'
#                                  'set_vals_1D', 'measure_vals_1D'])
#
#     def satisfies_condition(self, test_vals, target_val, relation):
#         return general_tools.get_truth(test_vals, target_val, relation)
#
#     def satisfies_conditions(self):
#
#     def get(self):
#         self.measure_parameter.setup()
#
#         for calibration_operation in self.calibration_operations:
#             if calibration_operation['mode'] == 'measure':
#                 vals = self.measure_parameter()
#
#         self.center_val = self.set_parameter()
#         self.set_vals_1D = list(np.linspace(self.center_val - self.span / 2,
#                                             self.center_val + self.span / 2,
#                                             self.set_points))
#
#         self.data_1D = self.cal1D_parameter()
#         self.measure_vals_1D = getattr(self.data_1D, self.key)
#         best_val_idx = np.argmax(self.measure_vals_1D)
#         best_val = self.set_vals_1D[best_val_idx]
#
#             if self.update:
#                 self.set_parameter(best_val)
#
#             self._save_val(best_val)
#             return best_val
#
#         elif self.ndims == 2:
#             raise NotImplementedError("Two parameters not implemented")
#         else:
#             raise ValueError("set_parameters must be one or two")

