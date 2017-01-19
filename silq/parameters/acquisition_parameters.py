from time import sleep
import numpy as np
import logging

from qcodes.instrument.parameter import Parameter
from qcodes.data import hdf5_format, io

from silq.pulses import PulseSequence, DCPulse, FrequencyRampPulse, \
    SinePulse, SteeredInitialization
from silq.analysis import analysis
from silq.tools import data_tools
from silq.tools.general_tools import SettingsClass, clear_single_settings


h5fmt = hdf5_format.HDF5Format()


class AcquisitionParameter(SettingsClass, Parameter):
    data_manager = None
    layout = None
    formatter = h5fmt

    def __init__(self, mode=None, average_mode='none', **kwargs):
        SettingsClass.__init__(self)
        self.mode = mode

        if self.mode is not None:
            # Add mode to parameter name and label
            kwargs['name'] += self.mode_str
            kwargs['label'] += ' {}'.format(self.mode)
        Parameter.__init__(self, **kwargs)
        self.label = kwargs['label']

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
        self.layout = self.layout
        self.data_manager = self.data_manager

        self._meta_attrs.extend(['label', 'name', 'pulse_sequence'])

    def __repr__(self):
        return '{} acquisition parameter'.format(self.name)

    @property
    def sample_rate(self):
        """ Acquisition sample rate """
        return self.layout.sample_rate()

    @property
    def _pulse_pts(self):
        """ Number of points in the trace of each pulse"""
        return {pulse.name: int(round(pulse.duration / 1e3 * self.sample_rate))
                for pulse in self.pulse_sequence}

    @property
    def start_idx(self):
        return round(self.t_skip * 1e-3 * self.sample_rate)

    def segment_trace(self, trace, average_mode=None):
        # TODO this function should likely go to Layout.
        # Furthermore, average_mode should perhaps be done elsewhere
        trace_segments = {}
        idx = 0
        for pulse in self.pulse_sequence:
            if not pulse.acquire:
                continue
            if average_mode == 'point':
                trace_segments[pulse.name] = np.mean(
                    trace[:, idx:idx + self._pulse_pts[pulse.name]])
            else:
                trace_segments[pulse.name] = \
                    trace[:, idx:idx + self._pulse_pts[pulse.name]]
            idx += self._pulse_pts[pulse.name]
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
        # Create a hard copy of pulse sequence. This ensures that pulse
        # attributes no longer depend on pulse_config, and can therefore be
        # safely transferred to layout.
        pulse_sequence = self.pulse_sequence.copy()
        self.layout.target_pulse_sequence(pulse_sequence)

        samples = kwargs.pop('samples', self.samples)
        average_mode = kwargs.pop('average_mode', self.average_mode)
        self.layout.setup(samples=samples,
                          average_mode=average_mode,
                          **kwargs)

    def acquire(self, segment_traces=True, average_mode=None, **kwargs):
        # Perform acquisition
        self.data = self.layout.do_acquisition(return_dict=True,
                                               **kwargs)
        if segment_traces:
            self.trace_segments = {
                ch_label: self.segment_trace(trace, average_mode=average_mode)
                for ch_label, trace in self.data['acquisition_traces'].items()}


class DCParameter(AcquisitionParameter):
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
            DCPulse(name='read', acquire=True),
            DCPulse(name='final')])

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.layout.start()

    def acquire(self, **kwargs):
        # Do not segment traces since we only receive a single value
        super().acquire(start=False, stop=False, segment_traces=False)

    @clear_single_settings
    def get(self):
        # Note that this function does not have a setup, and so the setup
        # must be done once beforehand.
        self.acquire()
        return self.data['acquisition_traces']['output']


class DCPulseSweepParameter(AcquisitionParameter):
    def __init__(self, sweep_name=None, **kwargs):
        super().__init__(name='DC_voltage',
                         label='DC voltage',
                         snapshot_value=False,
                         shape=(1,),
                         setpoint_names=[sweep_name],
                         setpoint_labels=[sweep_name],
                         **kwargs)

        self.pulse_settings = {'duration': 20}

        self.additional_pulses = []
        self._sweep_voltages = []
        self.sweep_pulse_names = []
        self.sweep_name = sweep_name

    @property
    def sweep_voltages(self):
        return self._sweep_voltages

    @sweep_voltages.setter
    def sweep_voltages(self, sweep_voltages):
        self._sweep_voltages = sweep_voltages
        self.pulse_sequence.clear()

        self.pulse_sequence.add([
            DCPulse('sweep_{:.3f}'.format(sweep_voltage),
                    acquire=True,
                    amplitude=sweep_voltage,
                    t_start=k*self.pulse_settings['duration'],
                    **self.pulse_settings)
            for k, sweep_voltage in enumerate(sweep_voltages)])
        self.sweep_pulse_names = ['sweep_{:.3f}'.format(sweep_voltage)
                                  for sweep_voltage in sweep_voltages]

        self.pulse_sequence.add(
            DCPulse(name='final',
                    connection_requirements=self.pulse_settings.get(
                        'connection_requirements', {})))

        self.pulse_sequence.add([pulse for pulse in self.additional_pulses])

        # Update metadata
        self.shape = tuple([len(sweep_voltages)])
        self.setpoints = (tuple(sweep_voltages), )

    @clear_single_settings
    def get(self):
        self.setup()
        self.acquire(average_mode='point')
        self.results = [self.trace_segments['output'][pulse_name]
                        for pulse_name in self.sweep_pulse_names]
        return self.results


class EPRParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='EPR',
                         label='Empty Plunge Read',
                         snapshot_value=False,
                         names=['contrast', 'dark_counts', 'voltage_difference',
                                'fidelity_empty', 'fidelity_load'],
                         **kwargs)

        self.pulse_sequence.add([
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True, mode='long'),
            DCPulse('final')])

        self.analysis = analysis.analyse_EPR

    @clear_single_settings
    def get(self):
        self.setup()

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


class AdiabaticParameter(AcquisitionParameter):
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
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True, mode='long'),
            DCPulse('final'),
            FrequencyRampPulse('adiabatic', mode=self.mode)])

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

    @clear_single_settings
    def get(self):
        self.setup()

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


class RabiParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to determine the Rabi frequency
        """
        super().__init__(name='rabi',
                         label='rabi',
                         snapshot_value=False,
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         **kwargs)

        self.pulse_sequence.add([
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            SinePulse('rabi', duration=0.1, mode=self.mode)])

        # Disable previous pulse for sine pulse, since it would
        # otherwise be after 'final' pulse
        self.pulse_sequence['rabi'].previous_pulse = None
        self.pulse_sequence.sort()

        self.analysis = analysis.analyse_PR

    @property
    def frequency(self):
        return self.pulse_sequence['rabi'].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['rabi'].frequency = frequency

    def acquire(self, **kwargs):
        super().acquire(return_initialization_traces=self.pulse_sequence[
            'steered_initialization'].enabled, **kwargs)

    @clear_single_settings
    def get(self):
        self.setup()

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


class RabiDriveParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to drive Rabi oscillations
        """
        super().__init__(name='rabi_drive',
                         label='rabi_drive',
                         snapshot_value=False,
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         **kwargs)

        self.pulse_sequence.add([
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            SinePulse('rabi', duration=0.1, mode=self.mode)])

        # Disable previous pulse for sine pulse, since it would
        # otherwise be after 'final' pulse
        self.pulse_sequence['rabi'].previous_pulse = None
        self.pulse_sequence.sort()

        self.analysis = analysis.analyse_PR

    @property
    def frequency(self):
        return self.pulse_sequence['rabi'].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['rabi'].frequency = frequency

    @property
    def duration(self):
        return self.pulse_sequence['rabi'].duration

    @duration.setter
    def duration(self, duration):
        self.pulse_sequence['rabi'].duration = duration

    def acquire(self, **kwargs):
        super().acquire(return_initialization_traces=self.pulse_sequence[
            'steered_initialization'].enabled, **kwargs)

    @clear_single_settings
    def get(self):
        self.setup()

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

    def set(self, duration):
        # Change drive duration
        self.duration = duration
        self.setup()


class T1Parameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='T1_wait_time',
                         label='T1 wait time',
                         snapshot_value=False,
                         names=['up_proportion', 'num_traces'],
                         **kwargs)
        self.pulse_sequence.add([
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge'),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            FrequencyRampPulse('adiabatic', mode=self.mode)])
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

    @clear_single_settings
    def get(self):
        self.setup()

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


class DarkCountsParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to perform an adiabatic sweep
        """
        super().__init__(name='dark_counts',
                         label='Dark counts',
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add([
            SteeredInitialization('steered_initialization', enabled=True,
                                  mode='long'),
            DCPulse('read', acquire=True)])

        self.analysis = analysis.analyse_read

        self.readout_threshold_voltage = None

        self._meta_attrs.append('readout_threshold_voltage')

    def acquire(self, **kwargs):
        super().acquire(return_initialization_traces=self.pulse_sequence[
            'steered_initialization'].enabled, **kwargs)

    @clear_single_settings
    def get(self):
        self.setup()

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


class VariableReadParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='variable_read_voltage',
                         label='Variable read voltage',
                         snapshot_value=False,
                         **kwargs)
        self.pulse_sequence.add([
            DCPulse(name='empty', acquire=True),
            DCPulse(name='plunge', acquire=True),
            DCPulse(name='read', acquire=True),
            DCPulse(name='final')])

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
