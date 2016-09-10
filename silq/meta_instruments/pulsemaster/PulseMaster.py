import numpy as np

from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils import validators as vals


class PulseMaster(Instrument):
    shared_kwargs = ['pulseblaster', 'arbstudio', 'ATS', 'ATS_controller']

    def __init__(self, pulseblaster, arbstudio, ATS, ATS_controller, **kwargs):
        super().__init__('PulseMaster', **kwargs)
        self.pulseblaster = pulseblaster
        self.arbstudio = arbstudio
        self.ATS = ATS
        self.ATS_controller = ATS_controller

        self.traces = None

        self.final_delay = 2


        self.add_parameter(name='samples',
                           parameter_class=ManualParameter,
                           initial_value=10,
                           vals=vals.Ints()
                           )
        self.add_parameter(name="acquisition",
                           names=['channel_signal'],
                           get_cmd=self.do_acquisition,
                           shapes=((),),
                           snapshot_value=False
                           )
        self.add_parameter(name="bare_acquisition",
                           names=['channel_signal'],
                           get_cmd=self.do_bare_acquisition,
                           shapes=((),),
                           snapshot_value=False
                           )
        self.add_parameter(name='stages',
                           parameter_class=ManualParameter,
                           initial_value={},
                           vals=vals.Anything()
                           )
        self.add_parameter(name='acquisition_stages',
                           parameter_class=ManualParameter,
                           initial_value=['read'],
                           vals=vals.Anything()
                           )
        self.add_parameter(name='sequence',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Anything()
                           )
        self.add_parameter(name='channel_factors',
                           parameter_class=ManualParameter,
                           initial_value=[1, -1.47, 1],
                           vals=vals.Anything()
                           )
        self.add_parameter(name='acquisition_channels',
                           parameter_class=ManualParameter,
                           initial_value='AC',
                           vals=vals.Anything()
                           )
        self.add_parameter(name='arbstudio_channels',
                           parameter_class=ManualParameter,
                           initial_value=[1, 2, 3],
                           vals=vals.Anything()
                           )
        self.add_parameter(name='digitizer_sample_rate',
                           parameter_class=ManualParameter,
                           initial_value=2e5,
                           vals=vals.Numbers()
                           )

    def full_sequence(self):
        if self.final_delay > 0:
            final_delay_stage = {'name': 'final_delay',
                                 'voltage': self.stages()[self.sequence()[-1]]['voltage'],
                                 'duration': self.final_delay}
            self.stages()['final_delay'] = final_delay_stage
            return self.sequence() + ['final_delay']
        else:
            return self.sequence()

    def configure_pulseblaster(self, marker_cycles=100, sampling_rate=500):
        # Factor of 2 needed because apparently the core clock is not the same as the sampling rate
        ms = 2 * sampling_rate * 1e3

        self.pulseblaster.stop()
        self.pulseblaster.detect_boards()
        self.pulseblaster.select_board(0)
        self.pulseblaster.core_clock(sampling_rate)

        self.pulseblaster.start_programming()

        pulse = self.sequence()[0]
        self.pulseblaster.send_instruction(0, 'continue', 0, marker_cycles)
        start = self.pulseblaster.send_instruction(0, 'continue', 0,
                                                   self.stages()[pulse]['duration'] * ms - marker_cycles)

        for pulse in self.full_sequence()[1:]:
            self.pulseblaster.send_instruction(1, 'continue', 0, marker_cycles)
            self.pulseblaster.send_instruction(0, 'continue', 0,
                                               self.stages()[pulse]['duration'] * ms - marker_cycles)

        self.pulseblaster.send_instruction(1, 'branch', start, marker_cycles)

        self.pulseblaster.stop_programming()

    def configure_arbstudio(self):
        self.arbstudio.stop()
        for ch in self.arbstudio_channels():
            eval("self.arbstudio.ch{}_trigger_source('fp_trigger_in')".format(ch))
            eval("self.arbstudio.ch{}_trigger_mode('stepped')".format(ch))
            eval('self.arbstudio.ch{}_clear_waveforms()'.format(ch))
            waveforms = self.channel_factors()[ch - 1] * \
                        np.array([[self.stages()[stage]['voltage']] * 4 for stage in self.full_sequence()])
            for waveform in waveforms:
                eval('self.arbstudio.ch{}_add_waveform(waveform)'.format(ch))
            sequence = list(range(len(self.full_sequence())))
            eval('self.arbstudio.ch{}_sequence({})'.format(ch, sequence))
        self.arbstudio.load_waveforms(channels=self.arbstudio_channels())
        self.arbstudio.load_sequence(channels=self.arbstudio_channels())

    def configure_ATS(self):
        # Determine voltage change for the starting acquisition stage
        start_acquisition = self.acquisition_stages()[0]
        start_acquisition_idx = self.sequence().index(start_acquisition)
        start_acquisition_voltage = self.stages()[start_acquisition]['voltage']

        # Determine the voltage of the previous acquisition stage
        pre_acquisition_idx = start_acquisition_idx - 1 if start_acquisition_idx > 0 else -1
        pre_acquisition = self.sequence()[pre_acquisition_idx]
        pre_acquisition_voltage = self.stages()[pre_acquisition]['voltage']

        voltage_difference = pre_acquisition_voltage - start_acquisition_voltage
        assert abs(
            voltage_difference) > 0, "The start of acquisition must have a voltage difference from the previous stage"
        trigger_slope = 'TRIG_SLOPE_NEGATIVE' if voltage_difference > 0 else 'TRIG_SLOPE_POSITIVE'

        # trigger level <128 is below 0V, >128 is above 0V
        trigger_level = round(128 * (1 + 0.5 * (pre_acquisition_voltage - voltage_difference / 2)))

        self.ATS.config(trigger_source1='CHANNEL_C',
                        trigger_level1=trigger_level,
                        trigger_slope1=trigger_slope,
                        external_trigger_coupling='DC',
                        trigger_operation='TRIG_ENGINE_OP_J',
                        channel_range=2,
                        sample_rate=self.digitizer_sample_rate(),
                        coupling='DC')

    def configure_ATS_controller(self, average_mode='none'):
        read_length = sum([self.stages()[stage]['duration'] for stage in self.acquisition_stages()])
        sample_rate = self.ATS_controller._get_alazar_parameter('sample_rate')
        samples_per_record = int(16 * round(float(sample_rate * read_length * 1e-3) / 16))

        # TODO proper automatic buffer time out
        buffer_timeout = 20000  # max(20000, 2.5*total_duration)

        self.ATS_controller.average_mode(average_mode)
        self.ATS_controller.update_acquisition_kwargs(samples_per_record=samples_per_record,
                                                      records_per_buffer=1,
                                                      buffers_per_acquisition=self.samples(),
                                                      buffer_timeout=buffer_timeout,
                                                      channel_selection=self.acquisition_channels())

    def setup(self, average_mode='none'):
        self.configure_pulseblaster()
        self.configure_arbstudio()
        self.configure_ATS()
        self.configure_ATS_controller(average_mode=average_mode)

        self.acquisition.names = self.ATS_controller.acquisition.names
        self.acquisition.labels = self.ATS_controller.acquisition.labels
        self.acquisition.units = self.ATS_controller.acquisition.units
        self.acquisition.shapes = self.ATS_controller.acquisition.shapes

    def start(self):
        self.arbstudio.run(channels=self.arbstudio_channels())
        self.pulseblaster.start()

    def stop(self):
        self.arbstudio.stop()
        self.pulseblaster.stop()

    def do_acquisition(self):
        self.start()
        self.bare_acquisition()
        self.stop()
        return self.traces

    def do_bare_acquisition(self):
        self.traces = self.ATS_controller.acquisition()
        return self.traces

# class PulseMaster(Instrument):
#     shared_kwargs = ['instruments']
#     def __init__(self, name, instruments, **kwargs):
#         super().__init__(name, **kwargs)
#
#         self.instruments = {instrument.name: instrument for instrument in instruments}
#         self.no_instruments = len(instruments)
#
#         # self.add_parameter('instruments',
#         #                    parameter_class=ManualParameter,
#         #                    initial_value={},
#         #                    vals=vals.Anything())
#         #
#         self.add_parameter('trigger_instrument',
#                            parameter_class=ManualParameter,
#                            initial_value=None,
#                            vals=vals.Enum(*self.instruments.keys()))
#
#         self.add_parameter('acquisition_instrument',
#                            parameter_class=ManualParameter,
#                            initial_value=None,
#                            vals=vals.Enum(*self.instruments.keys()))
#
#     # def add_instrument(self, instrument,
#     #                    acquisition_instrument=False):
#     #     assert isinstance(instrument, Instrument),\
#     #            'Can only add instruments that are a subclass of Instrument'
#     #     assert instrument.name not in self.instruments.keys(),\
#     #            'Instrument with same name is already added'
#     #
#     #     if acquisition_instrument:
#     #         self.acquisition_instrument = instrument.name
#
# class Connection():
#     def __init__(self, PulseMaster,
#                master_instrument_name, master_instrument_channel,
#                slave_instrument_name, slave_instrument_channel,
#                delay=0):
#         self.PulseMaster = PulseMaster
#
#         self.master_instrument = PulseMaster.instruments[master_instrument_name]
#         self.master_instrument_channel = master_instrument_channel
#
#         self.slave_instrument = PulseMaster.instruments[slave_instrument_name]
#         self.slave_instrument_channel = slave_instrument_channel
#
#         self.delay = delay