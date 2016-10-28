from time import sleep

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data import hdf5_format
h5fmt = hdf5_format.HDF5Format()

from silq.pulses import PulseSequence, DCPulse
from silq.analysis import analysis
from silq.tools import data_tools

class MeasurementParameter(Parameter):
    def __init__(self, layout, formatter=h5fmt, **kwargs):
        super().__init__(**kwargs)
        self.layout = layout
        self.pulse_sequence = PulseSequence()

        self.formatter = formatter

        self.print = False
        self.return_traces = None
        self.data_manager = None

        self._meta_attrs.extend(['pulse_sequence'])

    def setup(self, return_traces=False, data_manager=None, formatter=None,
              print=False):
        self.return_traces = return_traces
        self.data_manager = data_manager
        self.print = print
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

    def store_traces(self, traces):
        # Store raw traces if raw_data_manager is provided
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
                              t_start=5,duration=5, acquire=True)
        read_pulse = DCPulse(name='read', amplitude=0,
                              t_start=10,duration=20, acquire=True)
        final_pulse = DCPulse(name='final', amplitude=0,
                              t_start=30,duration=2)
        pulses = [empty_pulse, load_pulse, read_pulse, final_pulse]
        self.pulse_sequence.add(pulses)

    def setup(self, samples=100, **kwargs):
        self.samples = samples

        self.layout.target_pulse_sequence(self.pulse_sequence)

        self.layout.setup(samples=self.samples, average_mode='none')

        # Setup parameter metadata
        self.names = ['fidelity_empty', 'fidelity_load', 'fidelity_read',
                      'up_proportion', 'dark_counts', 'contrast']
        self.labels = self.names
        self.shapes = ((), (), (), (), (), ())

        super().setup(**kwargs)

    def get(self):
        self.layout.start()
        self.traces = self.layout.do_acquisition(return_dict=True)
        self.layout.stop()

        self.trace_segments = {ch_label: self.segment_trace(trace)
                               for ch_label, trace in self.traces.items()}

        fidelities = analysis.analyse_ELR(
            trace_segments=self.trace_segments['output'])

        # Store raw traces if raw_data_manager is provided
        if self.data_manager is not None:
            self.store_traces(self.traces['output'])

        # Print results
        if self.print:
            for name, fidelity in zip(self.names, fidelities):
                print('{}: {:.3f}'.format(name, fidelity))

        if self.return_traces:
            return fidelities + tuple(self.traces.values())
        else:
            return fidelities


class T1_Parameter(MeasurementParameter):

    def __init__(self, layout, **kwargs):
        super().__init__(name='T1_wait_time',
                         label='T1 wait time',
                         layout=layout,
                         snapshot_value=False,
                         **kwargs)
        self.tau = 5

        # Setup pulses
        empty_pulse = DCPulse(name='empty', amplitude=-1.5,
                              t_start=0,duration=5, acquire=True)
        load_pulse = DCPulse(name='load', amplitude=1.5,
                              t_start=5,duration=5, acquire=True)
        read_pulse = DCPulse(name='read', amplitude=0,
                              t_start=10,duration=20, acquire=True)
        final_pulse = DCPulse(name='final', amplitude=0,
                              t_start=30,duration=2)
        pulses = [empty_pulse, load_pulse, read_pulse, final_pulse]
        self.pulse_sequence.add(pulses)

        self.threshold_voltage = None
        self.start_point = None

        self._meta_attrs.extend(['threshold_voltage', 'start_point'])

    def setup(self, samples=50, t_start = 0.1,
              threshold_voltage=None, **kwargs):
        self.samples = samples
        self.threshold_voltage = threshold_voltage

        self.layout.target_pulse_sequence(self.pulse_sequence)

        self.layout.setup(samples=self.samples, average_mode='none')

        # Used to skip initial datapoints
        self.start_point = round(t_start * 1e-3 * self.layout.sample_rate())

        self.names = ['up_proportion', 'num_traces_loaded']
        self.labels = self.names
        self.shapes = ((), ())

    def get(self):
        self.traces = self.pulsemaster.acquisition()
        up_proportion, num_traces_loaded, _ = analysis.analyse_read(
            traces=self.traces,
            threshold_voltage=self.threshold_voltage,
            start_point=self.start_point)

        # Store raw traces if raw_data_manager is provided
        if self.data_manager is not None:
            self.store_traces(self.traces['output'])

        if self.return_traces:
            return up_proportion, num_traces_loaded, tuple(self.traces.values())
        else:
            return up_proportion, num_traces_loaded

    def set(self, tau):
        self.tau = tau
        # Change load stage duration.
        self.pulse_sequence['load'].duration = self.tau

        self.layout.target_pulse_sequence(self.pulse_sequence)

        self.layout.setup(samples=self.samples, average_mode='none')
#
# class VariableRead_Parameter(Parameter):
#
#     def __init__(self, pulsemaster, **kwargs):
#         super().__init__(name='variable_read_voltage',
#                          label='Variable read voltage',
#                          units='V',
#                          snapshot_value=False,
#                          **kwargs)
#         self.pulsemaster = pulsemaster
#
#         self.read_voltage = 0
#
#         self.stage_empty = {'name': 'empty', 'voltage': -1.5, 'duration': 5}
#         self.stage_load = {'name': 'load', 'voltage': 1.5, 'duration': 5}
#         self.stage_read = {'name': 'read', 'voltage': self.read_voltage, 'duration': 20}
#
#         self.stages = {stage['name']: stage for stage in
#                        [self.stage_empty, self.stage_load, self.stage_read]}
#
#         self._meta_attrs.extend(['stages'])
#
#     def setup(self, samples=50):
#
#         self.pulsemaster.stages(self.stages)
#         self.pulsemaster.sequence(['load', 'read', 'empty'])
#         self.pulsemaster.acquisition_stages(['load','read', 'empty'])
#
#         self.pulsemaster.samples(samples)
#         self.pulsemaster.arbstudio_channels([1,2,3])
#         self.pulsemaster.acquisition_channels('AC')
#         self.pulsemaster.setup(average_mode='trace')
#
#         self.names = self.pulsemaster.acquisition.names
#         self.labels = self.pulsemaster.acquisition.labels
#         self.shapes = self.pulsemaster.acquisition.shapes
#
#     def get(self):
#         traces, traces_AWG = self.pulsemaster.acquisition()
#         return traces, traces_AWG
#
#     def set(self, read_voltage):
#         self.read_voltage = read_voltage
#         # Change read stage voltage
#         self.stage_read['voltage'] = self.read_voltage
#         self.pulsemaster.stages(self.stages)
#         self.pulsemaster.setup(average_mode='trace')
#
# class DC_Parameter(Parameter):
#
#     def __init__(self, pulsemaster, **kwargs):
#         super().__init__(name='DC_voltage',
#                          label='DC voltage',
#                          units='V',
#                          snapshot_value=False,
#                          **kwargs)
#
#         self.pulsemaster = pulsemaster
#
#         self.stage_read = {'name': 'read', 'voltage': 0, 'duration': 20}
#         self.stage_marker = {'name': 'marker', 'voltage': 1, 'duration': 0.001}
#         self.stages = {'read': self.stage_read,
#                        'marker': self.stage_marker}
#
#     def setup(self, duration=20):
#         # Stop pulsemaster in case it was already running
#         self.pulsemaster.stop()
#
#         self.pulsemaster.stages(self.stages)
#         self.pulsemaster.sequence(['marker', 'read'])
#         self.pulsemaster.acquisition_stages(['read'])
#
#         self.pulsemaster.samples(1)
#         self.pulsemaster.arbstudio_channels([3])
#         self.pulsemaster.acquisition_channels('A')
#         self.pulsemaster.setup(average_mode='point')
#
#         # self.names = self.pulsemaster.acquisition.names
#         # self.labels = self.pulsemaster.acquisition.labels
#         # self.units = self.pulsemaster.acquisition.units
#
#         self.pulsemaster.start()
#
#     def get(self):
#         DC_signal, = self.pulsemaster.bare_acquisition()
#         return DC_signal
#
# class Up_Fidelity_Parameter(Parameter):
#     def __init__(self, name, ATS_controller, analysis, **kwargs):
#         super().__init__(name,
#                          snapshot_value=False,
#                          **kwargs)
#         self.ATS_controller = ATS_controller
#         self.analysis = analysis
#         self.traces = None
#
#     def get(self):
#         self.traces, self.traces_AWG = self.ATS_controller.acquisition()
#         return self.analysis.find_up_proportion(self.traces)