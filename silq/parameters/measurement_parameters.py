from time import sleep

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter
from silq.analysis import analysis
from silq.tools import data_tools

class ELR_Parameter(Parameter):

    def __init__(self, pulsemaster, **kwargs):
        super().__init__(name='ELR',
                         label='Empty Load Read',
                         snapshot_value=False,
                         **kwargs)
        self.pulsemaster = pulsemaster

        self.stage_empty = {'name': 'empty', 'voltage': -1.5, 'duration': 5}
        self.stage_load = {'name': 'load', 'voltage': 1.5, 'duration': 5}
        self.stage_read = {'name': 'read', 'voltage': 0, 'duration': 20}

        self.stages = {stage['name']: stage for stage in
                       [self.stage_empty, self.stage_load, self.stage_read]}

        self._meta_attrs.extend(['stages'])

    def setup(self, samples=100, return_traces=False):
        self.return_traces = return_traces

        self.pulsemaster.stages(self.stages )

        self.pulsemaster.sequence(['empty', 'load', 'read'])
        self.pulsemaster.acquisition_stages(['empty', 'load', 'read'])

        self.pulsemaster.samples(samples)
        self.pulsemaster.arbstudio_channels([1,2,3])
        self.pulsemaster.acquisition_channels('AC')
        self.pulsemaster.setup(average_mode='none')

        self.names = ['up_proportion', 'fidelity_empty', 'fidelity_load',
                      'fidelity_read' 'num_traces_loaded']
        self.labels = self.names.copy()
        self.shapes = ((), (), (), ())
        if self.return_traces:
            self.names += self.pulsemaster.acquisition.names
            self.labels += self.pulsemaster.acquisition.labels
            self.shapes += self.pulsemaster.acquisition.shapes

        digitizer_sample_rate = self.pulsemaster.digitizer_sample_rate()
        self.pts = {stage_name: round(stage['duration']/ 1e3 * digitizer_sample_rate)
                    for stage_name, stage in self.stages.items()}

    def segment_traces(self, traces):
        trace_segments = {}
        idx = 0
        for segment in ['empty', 'load', 'read']:
            stage = ''.join(char for char in segment if not char.isdigit())
            trace_segments[segment] = traces[:,idx:idx + self.pts[stage]]
            idx += self.pts[stage]
        return trace_segments

    def get(self):
        traces, traces_AWG = self.pulsemaster.acquisition()
        self.trace_segments = self.segment_traces(traces)
        fidelities = analysis.analyse_ELR(trace_segments=self.trace_segments)
        if self.return_traces:
            return fidelities + (traces, traces_AWG)
        else:
            return fidelities

class ELRLR_Parameter(Parameter):

    def __init__(self, pulsemaster, **kwargs):
        super().__init__(name='ELRLR',
                         label='Empty Load Read Load Read',
                         snapshot_value=False,
                         **kwargs)
        self.pulsemaster = pulsemaster

        self.stage_empty = {'name': 'empty', 'voltage': -1.5, 'duration': 5}
        self.stage_load = {'name': 'load', 'voltage': 1.5, 'duration': 5}
        self.stage_read = {'name': 'read', 'voltage': 0, 'duration': 20}

        self.stages = {stage['name']: stage for stage in
                       [self.stage_empty, self.stage_load, self.stage_read]}

        self._meta_attrs.extend(['stages'])

    def setup(self, samples=100, t_start=0.1,
              return_traces=False, print=False):
        self.return_traces = return_traces
        self.print = print

        self.start_point = round(t_start * 1e-3 * \
                           self.pulsemaster.digitizer_sample_rate())

        self.pulsemaster.stages(self.stages)

        self.pulsemaster.sequence(['empty', 'load', 'read', 'load', 'read'])
        self.pulsemaster.acquisition_stages(['empty', 'load', 'read', 'load', 'read'])

        self.pulsemaster.samples(samples)
        self.pulsemaster.arbstudio_channels([1,2,3])
        self.pulsemaster.acquisition_channels('AC')
        self.pulsemaster.setup(average_mode='none')

        self.names = ['fidelity_empty', 'fidelity_load', 'fidelity_read',
                      'up_proportion', 'dark_counts', 'contrast']
        self.labels = self.names.copy()
        self.shapes = ((), (), (), (), (), ())
        if self.return_traces:
            self.names += self.pulsemaster.acquisition.names
            self.labels += self.pulsemaster.acquisition.labels
            self.shapes += self.pulsemaster.acquisition.shapes

        digitizer_sample_rate = self.pulsemaster.digitizer_sample_rate()
        self.pts = {stage_name: round(stage['duration']/ 1e3 * digitizer_sample_rate)
                    for stage_name, stage in self.stages.items()}

    def segment_traces(self, traces):
        trace_segments = {}
        idx = 0
        for segment in ['empty', 'load1', 'read1', 'load2', 'read2']:
            stage = ''.join(char for char in segment if not char.isdigit())
            trace_segments[segment] = traces[:,idx:idx + self.pts[stage]]
            idx += self.pts[stage]
        return trace_segments

    def get(self):
        traces, traces_AWG = self.pulsemaster.acquisition()
        self.trace_segments = self.segment_traces(traces)
        fidelities = analysis.analyse_ELRLR(trace_segments=self.trace_segments,
                                            start_point=self.start_point)

        if self.print:
            for name, fidelity in zip(self.names, fidelities):
                print('{}: {:.3f}'.format(name, fidelity))

        if self.return_traces:
            return fidelities + (traces, traces_AWG)
        else:
            return fidelities

class T1_Parameter(Parameter):

    def __init__(self, pulsemaster, **kwargs):
        super().__init__(name='T1_wait_time',
                         label='T1 wait time',
                         snapshot_value=False,
                         **kwargs)
        self.pulsemaster = pulsemaster

        self.tau = 5

        self.stage_empty = {'name': 'empty', 'voltage': -1.5, 'duration': 5}
        self.stage_load = {'name': 'load', 'voltage': 1.5, 'duration': self.tau}
        self.stage_read = {'name': 'read', 'voltage': 0, 'duration': 20}

        self.stages = {stage['name']: stage for stage in
                       [self.stage_empty, self.stage_load, self.stage_read]}

        self._meta_attrs.extend(['stages'])

    def setup(self, samples=50, t_start = 0.1,
              return_traces=False, threshold_voltage=None,
              data_manager=None):
        self.threshold_voltage = threshold_voltage
        self.return_traces = return_traces
        self.data_manager = data_manager

        self.start_point = round(t_start * 1e-3 * \
                                 self.pulsemaster.digitizer_sample_rate())

        self.pulsemaster.stages(self.stages)
        self.pulsemaster.sequence(['empty', 'load', 'read'])
        self.pulsemaster.acquisition_stages(['read'])

        self.pulsemaster.samples(samples)
        self.pulsemaster.arbstudio_channels([1,2,3])
        self.pulsemaster.acquisition_channels('AC')
        self.pulsemaster.setup(average_mode='none')

        self.names = ['up_proportion', 'num_traces_loaded']
        self.labels = self.names.copy()
        self.shapes = ((),)
        if self.return_traces:
            self.names += self.pulsemaster.acquisition.names
            self.labels += self.pulsemaster.acquisition.labels
            self.shapes += self.pulsemaster.acquisition.shapes

    def get(self):
        traces, traces_AWG = self.pulsemaster.acquisition()
        up_proportion, num_traces_loaded, _ = analysis.analyse_read(
            traces=traces,
            threshold_voltage=self.threshold_voltage)

        # Store raw traces if raw_data_manager is provided
        if self.data_manager is not None:
            # Pause until data_manager is done measuring
            while self.data_manager.ask('get_measuring'):
                sleep(0.01)

            self.data_set = data_tools.create_raw_data_set(
                name='tau_{}'.format(round(self.tau)),
                data_manager=self.data_manager,
                shape=self.shape,
                formatter=self.formatter)
            data_tools.store_data(data_manager=self.data_manager,
                                  result=traces)

        if self.return_traces:
            return up_proportion, num_traces_loaded, traces, traces_AWG
        else:
            return up_proportion, num_traces_loaded

    def set(self, tau):
        self.tau = tau
        # Change load stage duration.
        self.stage_load['duration'] = self.tau
        self.pulsemaster.stages(self.stages)
        self.pulsemaster.setup()

class VariableRead_Parameter(Parameter):

    def __init__(self, pulsemaster, **kwargs):
        super().__init__(name='variable_read_voltage',
                         label='Variable read voltage',
                         units='V',
                         snapshot_value=False,
                         **kwargs)
        self.pulsemaster = pulsemaster

        self.read_voltage = 0

        self.stage_empty = {'name': 'empty', 'voltage': -1.5, 'duration': 5}
        self.stage_load = {'name': 'load', 'voltage': 1.5, 'duration': 5}
        self.stage_read = {'name': 'read', 'voltage': self.read_voltage, 'duration': 20}

        self.stages = {stage['name']: stage for stage in
                       [self.stage_empty, self.stage_load, self.stage_read]}

        self._meta_attrs.extend(['stages'])

    def setup(self, samples=50):

        self.pulsemaster.stages(self.stages)
        self.pulsemaster.sequence(['load', 'read', 'empty'])
        self.pulsemaster.acquisition_stages(['load','read', 'empty'])

        self.pulsemaster.samples(samples)
        self.pulsemaster.arbstudio_channels([1,2,3])
        self.pulsemaster.acquisition_channels('AC')
        self.pulsemaster.setup(average_mode='trace')

        self.names = self.pulsemaster.acquisition.names
        self.labels = self.pulsemaster.acquisition.labels
        self.shapes = self.pulsemaster.acquisition.shapes

    def get(self):
        traces, traces_AWG = self.pulsemaster.acquisition()
        return traces, traces_AWG

    def set(self, read_voltage):
        self.read_voltage = read_voltage
        # Change read stage voltage
        self.stage_read['voltage'] = self.read_voltage
        self.pulsemaster.stages(self.stages)
        self.pulsemaster.setup(average_mode='trace')

class DC_Parameter(Parameter):

    def __init__(self, pulsemaster, **kwargs):
        super().__init__(name='DC_voltage',
                         label='DC voltage',
                         units='V',
                         snapshot_value=False,
                         **kwargs)

        self.pulsemaster = pulsemaster

        self.stage_read = {'name': 'read', 'voltage': 0, 'duration': 20}
        self.stage_marker = {'name': 'marker', 'voltage': 1, 'duration': 0.001}
        self.stages = {'read': self.stage_read,
                       'marker': self.stage_marker}

    def setup(self, duration=20):
        # Stop pulsemaster in case it was already running
        self.pulsemaster.stop()

        self.pulsemaster.stages(self.stages)
        self.pulsemaster.sequence(['marker', 'read'])
        self.pulsemaster.acquisition_stages(['read'])

        self.pulsemaster.samples(1)
        self.pulsemaster.arbstudio_channels([3])
        self.pulsemaster.acquisition_channels('A')
        self.pulsemaster.setup(average_mode='point')

        self.pulsemaster.start()

    def get(self):
        DC_signal, = self.pulsemaster.bare_acquisition()
        return DC_signal

class Up_Fidelity_Parameter(Parameter):
    def __init__(self, name, ATS_controller, analysis, **kwargs):
        super().__init__(name,
                         snapshot_value=False,
                         **kwargs)
        self.ATS_controller = ATS_controller
        self.analysis = analysis
        self.traces = None

    def get(self):
        self.traces, self.traces_AWG = self.ATS_controller.acquisition()
        return self.analysis.find_up_proportion(self.traces)