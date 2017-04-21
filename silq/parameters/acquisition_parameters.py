from time import sleep
import numpy as np
import logging

from qcodes.instrument.parameter import MultiParameter
from qcodes.data import hdf5_format, io
from qcodes.data.data_array import DataArray
from qcodes.loops import active_loop

from silq.pulses import PulseSequence, DCPulse, FrequencyRampPulse, \
    SinePulse, SteeredInitialization
from silq.analysis import analysis
from silq.tools import data_tools
from silq.tools.general_tools import SettingsClass, clear_single_settings, \
    attribute_from_config


h5fmt = hdf5_format.HDF5Format()


class AcquisitionParameter(SettingsClass, MultiParameter):
    layout = None
    formatter = h5fmt

    def __init__(self, average_mode='none', **kwargs):
        SettingsClass.__init__(self)

        shapes = kwargs.pop('shapes', ((), ) * len(kwargs['names']))
        MultiParameter.__init__(self, shapes=shapes, **kwargs)

        self.pulse_sequence = PulseSequence()
        """Pulse sequence of acquisition parameter"""

        self.silent = True
        """Do not print results after acquisition"""

        self.average_mode = average_mode
        """Type of averaging performed on data"""

        self.save_traces = False
        """ Save traces in separate files (does not work)"""

        self.samples = None
        self.t_read = None
        self.t_skip = None
        self.trace_segments = None
        self.data = None
        self.dataset = None
        self.results = None

        self.subfolder = None

        # Change attribute data_manager from class attribute to instance
        # attribute. This is necessary to ensure that the data_manager is
        # passed along when the parameter is spawned from a new process
        self.layout = self.layout

        self._meta_attrs.extend(['label', 'name', 'pulse_sequence'])

    def __repr__(self):
        return '{} acquisition parameter'.format(self.name)

    def __getattribute__(self, item):
        try:
            super().__getattribute__(item)
        except AttributeError:
            return attribute_from_config(item)

    @property
    def sample_rate(self):
        """ Acquisition sample rate """
        return self.layout.sample_rate()

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
            pulse_pts = int(round(pulse.duration / 1e3 * self.sample_rate))
            if average_mode == 'point':
                trace_segments[pulse.name] = np.mean(
                    trace[:, idx:idx + pulse_pts])
            else:
                trace_segments[pulse.name] = \
                    trace[:, idx:idx + pulse_pts]
            idx += pulse_pts
        return trace_segments

    def store_traces(self, traces_dict, base_folder=None, subfolder=None):
        # Store raw traces
        if base_folder is None:
            # Extract base_folder from dataset of currently active loop
            active_dataset = active_loop().get_data_set()
            base_folder = active_dataset.location
        self.dataset = data_tools.create_data_set(name='traces',
                                                  base_folder=base_folder,
                                                  subfolder=subfolder)

        # Create dictionary of set arrays
        set_arrs = {}
        for traces_name, traces in traces_dict.items():
            number_of_traces, points_per_trace = traces.shape

            if traces.shape not in set_arrs:
                time_step = 1 / self.sample_rate * 1e3
                t_list = np.arange(0, points_per_trace * time_step, time_step)
                t_list_arr = DataArray(name='time',
                                       array_id='time',
                                       label=' Time (ms)',
                                       # shape=(points_per_trace, ),
                                       # preset_data=t_list,
                                       shape=traces.shape,
                                       preset_data=np.full(traces.shape,
                                                           t_list),
                                       is_setpoint=True)

                trace_num_arr = DataArray(name='trace_num',
                                          array_id='trace_num',
                                          label='Trace number',
                                          # shape=traces.shape,
                                          # preset_data=np.full(traces.shape[
                                          #                     ::-1],
                                          #                     np.arange(number_of_traces),
                                          #                     dtype=np.float64).transpose(),
                                          shape=(number_of_traces, ),
                                          preset_data=np.arange(
                                              number_of_traces, dtype=np.float64),
                                          is_setpoint=True)
                set_arrs[traces.shape] = (trace_num_arr, t_list_arr)

        # Add set arrays to dataset
        for k, (t_list_arr, trace_num_arr) in enumerate(set_arrs.values()):
            for arr in (t_list_arr, trace_num_arr):
                if len(set_arrs) > 1:
                    # Need to give individual array_ids to each of the set arrays
                    arr.array_id += '_{}'.format(k)
                self.dataset.add_array(arr)

        # Add trace arrs to dataset
        for traces_name, traces in traces_dict.items():
            t_list_arr, trace_num_arr = set_arrs[traces.shape]

            # Must transpose traces array
            trace_arr = DataArray(name=traces_name,
                                  array_id=traces_name,
                                  label=traces_name + ' signal (V)',
                                  shape=traces.shape,
                                  preset_data=traces,
                                  set_arrays=(t_list_arr, trace_num_arr))
            self.dataset.add_array(trace_arr)

        self.dataset.finalize()

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
        super().__init__(name='DC_acquisition',
                         names=['DC_voltage'],
                         labels=['DC voltage'],
                         units=['V'],
                         average_mode='point',
                         snapshot_value=False,
                         **kwargs)

        self.samples = 1

        self.pulse_sequence.add(
            DCPulse(name='read', acquire=True),
            DCPulse(name='final'))

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
        return [self.data['acquisition_traces']['output']]


class DCPulseSweepParameter(AcquisitionParameter):
    def __init__(self, sweep_name=None, **kwargs):
        super().__init__(name='DC_acquisition',
                         names=['DC_voltage'],
                         labels=['DC voltage'],
                         snapshot_value=False,
                         setpoint_names=[sweep_name],
                         setpoint_labels=[sweep_name],
                         **kwargs)

        self.pulse_settings = {'duration': 20}

        self.additional_pulses = []
        self._sweep_voltages = []
        self.sweep_pulse_names = []
        self.sweep_name = sweep_name
        self.samples = 1

    @property
    def sweep_voltages(self):
        return self._sweep_voltages

    @sweep_voltages.setter
    def sweep_voltages(self, sweep_voltages):
        self._sweep_voltages = sweep_voltages
        self.pulse_sequence.clear()

        self.pulse_sequence.add(
            DCPulse('sweep_{:.3f}'.format(sweep_voltage),
                    acquire=True,
                    amplitude=sweep_voltage,
                    t_start=k*self.pulse_settings['duration'],
                    **self.pulse_settings)
            for k, sweep_voltage in enumerate(sweep_voltages))
        self.sweep_pulse_names = ['sweep_{:.3f}'.format(sweep_voltage)
                                  for sweep_voltage in sweep_voltages]

        self.pulse_sequence.add(
            DCPulse(name='final',
                    connection_requirements=self.pulse_settings.get(
                        'connection_requirements', {})))

        self.pulse_sequence.add(pulse for pulse in self.additional_pulses)

        # Update metadata
        self.shapes = [tuple([len(sweep_voltages)])]
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
        super().__init__(name='EPR_acquisition',
                         names=['contrast', 'dark_counts', 'voltage_difference',
                                'fidelity_empty', 'fidelity_load'],
                         labels=['Contrast', 'Dark counts',
                                 'Voltage difference',
                                 'Fidelity empty', 'Fidelity load'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True, mode='long'),
            DCPulse('final'))

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
        super().__init__(name='adiabatic_acquisition',
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         labels=['Contrast', 'Dark counts',
                                 'Voltage difference'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True, mode='long'),
            DCPulse('final'),
            FrequencyRampPulse('adiabatic_ESR'))

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


class RabiParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to determine the Rabi frequency
        """
        super().__init__(name='rabi_acquisition',
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         labels=['Contrast', 'Dark counts',
                                'Voltage difference'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            SinePulse('rabi_ESR', duration=0.1))

        # Disable previous pulse for sine pulse, since it would
        # otherwise be after 'final' pulse
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

class RabiDriveParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to drive Rabi oscillations
        """
        super().__init__(name='rabi_drive',
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         labels=['Contrast', 'Dark counts',
                                'Voltage difference'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            SinePulse('rabi_ESR', duration=0.1))

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


class T1Parameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='T1_acquisition',
                         names=['up_proportion', 'num_traces'],
                         labels=['Up proportion', 'Number of traces'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge'),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            FrequencyRampPulse('adiabatic_ESR'))
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
            if self.subfolder is not None:
                subfolder = '{}/tau_{:.0f}'.format(self.subfolder,
                                               self.wait_time)
            else:
                subfolder = 'tau_{:.0f}'.format(self.wait_time)

            self.store_traces(saved_traces, subfolder=subfolder)

        if not self.silent:
            self.print_results()

        return self.results


class DarkCountsParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to perform an adiabatic sweep
        """
        super().__init__(name='dark_counts_acquisition',
                         names=['dark_counts'],
                         labels=['Dark counts'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=True,
                                  mode='long'),
            DCPulse('read', acquire=True))

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
        self.results = [fidelities['up_proportion']]

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

class VariableReadParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='variable_read_acquisition',
                         names=['read_voltage'],
                         labels='Read voltage',
                         average_mode='trace',
                         units=['V'],
                         snapshot_value=False,
                         **kwargs)
        self.pulse_sequence.add(
            DCPulse(name='empty', acquire=True),
            DCPulse(name='plunge', acquire=True),
            DCPulse(name='read', acquire=True),
            DCPulse(name='final'))

    @property
    def shape(self):
        return self.layout.acquisition.shapes[0]

    def get(self):
        self.setup()

        self.acquire(segment_traces=False)

        return self.data['acquisition_traces']['output']
