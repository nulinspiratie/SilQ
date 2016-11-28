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

        self.print_results = False
        self.return_traces = None
        self.data_manager = None

        self._meta_attrs.extend(['pulse_sequence'])

    def setup(self, return_traces=False, data_manager=None, formatter=None,
              print_results=False, **kwargs):
        self.return_traces = return_traces
        self.data_manager = data_manager
        self.print_results = print_results
        if formatter is not None:
            self.formatter = formatter

        sample_rate = self.layout.sample_rate()
        self.pts = {pulse.name: int(round(pulse.duration/ 1e3 * sample_rate))
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


class EPR_Parameter(MeasurementParameter):

    def __init__(self, layout, **kwargs):
        super().__init__(name='EPR',
                         label='Empty Plunge Read',
                         layout=layout,
                         snapshot_value=False,
                         **kwargs)

        empty_pulse = DCPulse(name='empty', amplitude=-1.5,
                              t_start=0,duration=5, acquire=True)
        plunge_pulse = DCPulse(name='plunge', amplitude=1.5,
                               duration=5, acquire=True)
        read_pulse = DCPulse(name='read', amplitude=0,
                              duration=50, acquire=True)
        final_pulse = DCPulse(name='final', amplitude=0,
                              duration=2)

        self.pulse_sequence.add(
            [empty_pulse, plunge_pulse, read_pulse, final_pulse])

        self.samples = 100
        self.t_skip = 0.1
        self.t_read = 20

        self.analysis = analysis.analyse_EPR

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
        if self.print_results:
            for name, fidelity in zip(self.names, fidelities):
                print('{}: {:.3f}'.format(name, fidelity))

        if self.return_traces:
            return fidelities + tuple(self.traces.values())
        else:
            return fidelities


class AdiabaticSweep_Parameter(EPR_Parameter):
    def __init__(self, layout, **kwargs):
        super().__init__(layout=layout, **kwargs)
        self.name = 'adiabatic_sweep'
        self.label = 'Adiabatic sweep center frequency'

        frequency_center = 20e9  # Hz
        frequency_deviation = 10e6  # Hz
        power = 10  # dBm

        ESR_pulse = FrequencyRampPulse(
            name='adiabatic_sweep',
            power=power,
            t_start=9, duration=0.2,
            frequency_center=frequency_center,
            frequency_deviation=frequency_deviation)
        steered_initialization = SteeredInitialization(
            name='steered_initialization',
            t_no_blip=30, t_max_wait=200, t_buffer=20)

        self.pulse_sequence.add([ESR_pulse, steered_initialization])
        self.pulse_sequence['empty'].enabled = False

        self.analysis = analysis.analyse_PR

    def setup(self, readout_threshold_voltage=None, **kwargs):
        if readout_threshold_voltage is not None:
            self.readout_threshold_voltage = readout_threshold_voltage
        super().setup(readout_threshold_voltage=self.readout_threshold_voltage,
                      **kwargs)
        self.names = ['fidelity_load', 'fidelity_read',
                      'up_proportion', 'dark_counts', 'contrast']
        self.labels = self.names
        self.shapes = ((), (), (), (), ())

    def set(self, frequency_center):
        # Change center frequency
        self.pulse_sequence['adiabatic_sweep'].frequency_center = \
            frequency_center

        self.setup()


class T1_Parameter(AdiabaticSweep_Parameter):

    def __init__(self, layout, **kwargs):
            super().__init__(layout=layout, **kwargs)
            self.name = 'T1_wait_time'
            self.label = 'T1_wait_time'

            self.analysis = analysis.analyse_read

    @property
    def plunge_duration(self):
        return self.pulse_sequence['plunge'].duration

    def setup(self, readout_threshold_voltage=None, **kwargs):
        if readout_threshold_voltage is not None:
            self.readout_threshold_voltage = readout_threshold_voltage

        super().setup(readout_threshold_voltage=self.readout_threshold_voltage,
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
        if self.print_results:
            print('up_proportion: {:.3f}'.format(up_proportion))

        if self.return_traces:
            return up_proportion, num_traces_loaded, tuple(
                self.traces.values())
        else:
            return up_proportion, num_traces_loaded

    def set(self, plunge_duration):
            self.pulse_sequence['plunge'].duration = plunge_duration

            self.setup()


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
