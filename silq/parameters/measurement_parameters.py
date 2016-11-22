from time import sleep

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data import hdf5_format
h5fmt = hdf5_format.HDF5Format()

from silq.pulses import PulseSequence, DCPulse, FrequencyRampPulse, \
    SteeredInitialization
from silq.analysis import analysis
from silq.tools import data_tools

class MeasurementParameter(Parameter):
    def __init__(self, layout, formatter=h5fmt, **kwargs):
        super().__init__(**kwargs)
        self.layout = layout
        self.pulse_sequence = PulseSequence()

        self.formatter = formatter

        self.print_flag = False
        self.return_traces = None
        self.data_manager = None

        self._meta_attrs.extend(['pulse_sequence'])

    def setup(self, return_traces=False, data_manager=None, formatter=None,
              print_flag=False, **kwargs):
        self.return_traces = return_traces
        self.data_manager = data_manager
        self.print_flag = print_flag
        if formatter is not None:
            self.formatter = formatter

        sample_rate = self.layout.sample_rate()
        self.pts = {pulse.name: round(pulse.duration/ 1e3 * sample_rate)
                    for pulse in self.pulse_sequence}

        if self.return_traces:
            self.names += self.layout.acquisition.names
            self.labels += self.layout.acquisition.labels
            self.shapes += self.layout.acquisition.shapes

    def segment_trace(self, trace):
        trace_segments = {}
        idx = 0
        for pulse in self.pulse_sequence:
            trace_segments[pulse.name] = \
                trace[:, idx:idx + self.pts[pulse.name]]
            idx += self.pts[pulse.name]
        return trace_segments

    def store_traces(self, traces, name=None):
        # Store raw traces if raw_data_manager is provided
        if name is None:
            name = self.name

        # Pause until data_manager is done measuring
        while self.data_manager.ask('get_measuring'):
            sleep(0.01)

        self.data_set = data_tools.create_raw_data_set(
            name=self.name,
            data_manager=self.data_manager,
            shape=traces.shape,
            formatter=self.formatter)
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

        read_pulse = DCPulse(name='read', amplitude=0,
                              duration=20, acquire=True)
        final_pulse = DCPulse(name='final', amplitude=0,
                              duration=1)
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
        return signal['output']


class ELR_Parameter(MeasurementParameter):

    def __init__(self, layout, **kwargs):
        super().__init__(name='ELR',
                         label='Empty Load Read',
                         layout=layout,
                         snapshot_value=False,
                         **kwargs)

        empty_pulse = DCPulse(name='empty', amplitude=-1.5,
                              t_start=0,duration=5, acquire=True)
        load_pulse = DCPulse(name='load', amplitude=1.5,
                             duration=5, acquire=True)
        read_pulse = DCPulse(name='read', amplitude=0,
                              duration=50, acquire=True)
        final_pulse = DCPulse(name='final', amplitude=0,
                              duration=2)
        pulses = [empty_pulse, load_pulse, read_pulse, final_pulse]
        self.pulse_sequence.add(pulses)

        self.samples = 100
        self.t_skip = 0.1
        self.t_read = 20


        self.analysis = analysis.analyse_ELR

        self._meta_attrs.extend(['t_skip', 't_read'])

    def setup(self, samples=None, t_skip=None, t_read=None,
              data_manager=None, **kwargs):
        if samples:
            self.samples = samples
        if t_skip:
            self.t_skip = t_skip
        if t_read:
            self.t_read = t_read

        self.layout.target_pulse_sequence(self.pulse_sequence)

        self.layout.setup(samples=self.samples, average_mode='none', **kwargs)

        # Setup parameter metadata
        self.names = ['fidelity_empty', 'fidelity_load', 'fidelity_read',
                      'up_proportion', 'dark_counts', 'contrast']
        self.labels = self.names
        self.shapes = ((), (), (), (), (), ())

        super().setup(data_manager=data_manager, **kwargs)

    def get(self):

        self.traces = self.layout.do_acquisition(return_dict=True)

        self.trace_segments = {ch_label: self.segment_trace(trace)
                               for ch_label, trace in self.traces.items()}

        fidelities = self.analysis(
            trace_segments=self.trace_segments['output'],
            sample_rate=self.layout.sample_rate(),
            t_skip=self.t_skip,
            t_read=self.t_read)

        # Store raw traces if raw_data_manager is provided
        if self.data_manager is not None:
            self.store_traces(self.traces['output'])

        # Print results
        if self.print_flag:
            for name, fidelity in zip(self.names, fidelities):
                print('{}: {:.3f}'.format(name, fidelity))

        if self.return_traces:
            return fidelities + tuple(self.traces.values())
        else:
            return fidelities


class AdiabaticSweep_Parameter(ELR_Parameter):
    def __init__(self, layout, **kwargs):
        super().__init__(layout=layout, **kwargs)
        self.name = 'adiabatic_sweep'
        self.label = 'Adiabatic sweep center frequency'

        frequency_center = 20e9  # Hz
        frequency_deviation = 10e6  # Hz
        power = 10  # dBm

        self.readout_threshold_voltage=0.4

        self.pulse_sequence.clear()
        self._load_pulse = DCPulse(name='load', amplitude=1.5,
                             duration=10, acquire=True)
        self._read_pulse = DCPulse(name='read', amplitude=0,
                              duration=50, acquire=True)
        self._final_pulse = DCPulse(name='final', amplitude=0,
                              duration=2)
        self._ESR_pulse = FrequencyRampPulse(
            name='adiabatic_sweep',
            power=power,
            t_start=9, duration=0.2,
            frequency_center=frequency_center,
            frequency_deviation=frequency_deviation)
        self._steered_initialization = SteeredInitialization(
            name='steered_initialization',
            t_no_blip=30, t_max_wait=200, t_buffer=20)
        pulses = [self._load_pulse, self._read_pulse, self._final_pulse,
                  self._ESR_pulse, self._steered_initialization]
        self.pulse_sequence.add(pulses)

        self.analysis = analysis.analyse_LR

    @property
    def steered_initialization(self):
        return 'steered_initialization' in self.pulse_sequence

    @steered_initialization.setter
    def steered_initialization(self, use_steered_initialization):
        if use_steered_initialization and \
                'steered_initialization' not in self.pulse_sequence:
            self.pulse_sequence.add(self.steered_initialization)
        elif not use_steered_initialization and \
                'steered_initialization' in self.pulse_sequence:
            self.pulse_sequence.remove('steered_initialization')

    def setup(self, readout_threshold_voltage, **kwargs):
        self.readout_threshold_voltage = readout_threshold_voltage
        super().setup(readout_threshold_voltage=readout_threshold_voltage,
                      **kwargs)
        self.names = ['fidelity_load', 'fidelity_read',
                      'up_proportion', 'dark_counts', 'contrast']
        self.labels = self.names
        self.shapes = ((), (), (), (), ())

    def set(self, frequency_center):
        # Change center frequency
        self.pulse_sequence['adiabatic_sweep'].frequency_center = \
            frequency_center

        self.setup(readout_threshold_voltage=self.readout_threshold_voltage)

class T1_Parameter(AdiabaticSweep_Parameter):

    def __init__(self, layout, **kwargs):
            super().__init__(layout=layout, **kwargs)
            self.name = 'T1_wait_time'
            self.label = 'T1_wait_time'

            frequency_center = 20e9  # Hz
            frequency_deviation = 10e6  # Hz
            power = 10  # dBm

            self.plunge_duration=5

            self.pulse_sequence.clear()
            #empty_pulse = DCPulse(name='empty', amplitude=-1.5,
                                  t_start=0, duration=5, acquire=True)
            self._adiabatic_plunge_pulse = DCPulse(name='adiabatic_plunge',
                                                   amplitude=1.5,
                                       duration=2, acquire=True)
            self._read_pulse = DCPulse(name='read', amplitude=0,
                                       duration=50, acquire=True)
            self._final_pulse = DCPulse(name='final', amplitude=0,
                                        duration=2)
            self._plunge_pulse = DCPulse(name='plunge', amplitude=1.5,
                                      duration=self.plunge_duration,
                                      acquire=False)
            self._ESR_pulse = FrequencyRampPulse(
                name='adiabatic_sweep',
                power=power,
                t_start=9, duration=0.2,
                frequency_center=frequency_center,
                frequency_deviation=frequency_deviation)
            self._steered_initialization = SteeredInitialization(
                name='steered_initialization',
                t_no_blip=30, t_max_wait=200, t_buffer=20)
            pulses = [self._steered_initialization,
                      self._adiabatic_plunge_pulse,
                      self._ESR_pulse,
                      #empty_pulse,
                      self._plunge_pulse,
                      self._read_pulse,
                      self._final_pulse]

            self.pulse_sequence.add(pulses)

            self.analysis = analysis.analyse_read

    def setup(self, readout_threshold_voltage, **kwargs):
            self.readout_threshold_voltage = readout_threshold_voltage
            super().setup(readout_threshold_voltage=readout_threshold_voltage,
                          **kwargs)
            self.names = ['up_proportion', 'num_traces_loaded']
            self.labels = self.names
            self.shapes = ((), ())


    def get(self):
        self.traces = self.layout.do_acquisition(return_dict=True)

        self.trace_segments = {ch_label: self.segment_trace(trace)
                                   for ch_label, trace in self.traces.items()}
        output_trace=self.trace_segments['output']
        up_proportion, num_traces_loaded, _ = self.analysis(
                traces=output_trace['read'],
                threshold_voltage=self.readout_threshold_voltage,
                start_idx=round(self.t_skip * 1e-3 * self.layout.sample_rate()))

            # Store raw traces if raw_data_manager is provided
        if self.data_manager is not None:
            self.store_traces(self.traces['output'])

        # Print results
        if self.print_flag:
            print('up_proportion: {:.3f}'.format(up_proportion))

        if self.return_traces:
            return up_proportion, num_traces_loaded, tuple(
                self.traces.values())
        else:
            return up_proportion, num_traces_loaded


    def set(self, plunge_duration):
            self.plunge_duration=plunge_duration
            self.pulse_sequence['plunge'].duration=self.plunge_duration

            self.setup(
                self.readout_threshold_voltage,
                print_flag=self.print_flag)


# class T1_Parameter(MeasurementParameter):
#
#     def __init__(self, layout, **kwargs):
#         super().__init__(name='T1_wait_time',
#                          label='T1 wait time',
#                          layout=layout,
#                          snapshot_value=False,
#                          **kwargs)
#         self.tau = 5
#
#         # Setup pulses
#         empty_pulse = DCPulse(name='empty', amplitude=-1.5,
#                               t_start=0,duration=5, acquire=True)
#         load_pulse = DCPulse(name='load', amplitude=1.5,
#                               t_start=5,duration=5, acquire=True)
#         read_pulse = DCPulse(name='read', amplitude=0,
#                               t_start=10,duration=20, acquire=True)
#         final_pulse = DCPulse(name='final', amplitude=0,
#                               t_start=30,duration=2)
#         pulses = [empty_pulse, load_pulse, read_pulse, final_pulse]
#         self.pulse_sequence.add(pulses)
#
#         self.samples=50
#         self.threshold_voltage = None
#         self.t_skip = 0.1
#
#         self._meta_attrs.extend(['threshold_voltage', 't_skip'])
#
#     def setup(self, samples=None, t_skip=None, threshold_voltage=None,
#               **kwargs):
#         if samples:
#             self.samples = samples
#         if t_skip:
#             self.t_skip = t_skip
#         self.threshold_voltage = threshold_voltage
#
#         self.layout.target_pulse_sequence(self.pulse_sequence)
#
#         self.layout.setup(samples=self.samples, average_mode='none')
#
#         self.names = ['up_proportion', 'num_traces_loaded']
#         self.labels = self.names
#         self.shapes = ((), ())
#
#         super().setup(**kwargs)
#
#     def get(self):
#         self.traces = self.layout.do_acquisition(return_dict=True)
#
#         self.trace_segments = {ch_label: self.segment_trace(trace)
#                                for ch_label, trace in self.traces.items()}
#         up_proportion, num_traces_loaded, _ = analysis.analyse_read(
#             traces=self.traces,
#             threshold_voltage=self.threshold_voltage,
#             start_idx=round(self.t_skip * 1e-3 * self.layout.sample_rate()))
#
#         # Store raw traces if raw_data_manager is provided
#         if self.data_manager is not None:
#             self.store_traces(self.traces['output'])
#
#         if self.return_traces:
#             return up_proportion, num_traces_loaded, tuple(self.traces.values())
#         else:
#             return up_proportion, num_traces_loaded
#
#     def set(self, tau):
#         self.tau = tau
#         # Change load stage duration.
#         self.pulse_sequence['load'].duration = self.tau
#
#         self.layout.target_pulse_sequence(self.pulse_sequence)
#
#         self.layout.setup(samples=self.samples, average_mode='none')


class VariableRead_Parameter(MeasurementParameter):
    def __init__(self, layout, **kwargs):
        super().__init__(name='variable_read_voltage',
                         label='Variable read voltage',
                         layout=layout,
                         snapshot_value=False,
                         **kwargs)
        empty_pulse = DCPulse(name='empty', amplitude=-1.5,
                              t_start=0, duration=5, acquire=True)
        load_pulse = DCPulse(name='load', amplitude=1.5,
                             duration=5, acquire=True)
        read_pulse = DCPulse(name='read', amplitude=0,
                             duration=20, acquire=True)
        final_pulse = DCPulse(name='final', amplitude=0,
                              duration=2)
        
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
