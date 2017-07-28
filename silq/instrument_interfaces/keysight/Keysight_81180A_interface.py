import numpy as np
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, SinePulse, MarkerPulse, \
    PulseImplementation
from silq.tools.general_tools import arreqclose_in_list

from qcodes import ManualParameter
from qcodes import validators as vals


logger = logging.getLogger(__name__)


class Keysight81180AInterface(InstrumentInterface):
    """

    Parameters that should be set in Instrument driver:
        output_coupling
        sample_rate
    """

    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, output=True)
            for k in [1, 2]}

        # TODO add marker outputs
        self._channels = {
            **self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True),
            'sync': Channel(instrument_name=self.instrument_name(),
                            name='sync', output=True)}

        self.pulse_implementations = []

        self.add_parameter('trigger_in_duration',
                           parameter_class=ManualParameter, unit='us',
                           initial_value=0.1)
        self.add_parameter('active_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Lists(vals.Strings()))

    def get_additional_pulses(self, **kwargs):
        additional_pulses = []
        return additional_pulses

    def setup(self, **kwargs):
        self.active_channels(list({pulse.connection.output['channel'].name
                                   for pulse in self.pulse_sequence}))
        for ch_name in self.active_channels():
            instrument_channel = getattr(self.instrument, ch_name)

            instrument_channel.run_mode('sequenced')
            instrument_channel.sequence_mode('stepped')
            instrument_channel.trigger_source('external')
            instrument_channel.trigger_mode('override')

            pulses = self.pulse_sequence.get_pulses(input_channel=ch_name)

            # Add empty waveform, with minimum points (320)
            empty_segment = self.instrument.add_waveform(np.zeros(320))



    def start(self):
        for ch_name in self.active_channels():
            instrument_channel = getattr(self.instrument, ch_name)
            instrument_channel.on()

    def stop(self):
        self.instrument.ch1.off()
        self.instrument.ch2.off()

    class SinePulseImplementation(PulseImplementation):
        pulse_class = SinePulse

        def implement(self):
            pass

    class MarkerPulseImplementation(PulseImplementation):
        pulse_class = MarkerPulse

        def implement(self):
            pass