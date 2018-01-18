import numpy as np
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, DCRampPulse, TriggerPulse, SinePulse, \
    PulseImplementation
from silq.tools.general_tools import arreqclose_in_list

from qcodes.utils.validators import Lists, Enum


logger = logging.getLogger(__name__)

class AWG520Interface(InstrumentInterface):
    """

    Notes:
        - Sets first point of each waveform to final voltage of previous
          waveform because this is the value used when the previous waveform
          ended and is waiting for triggers.

    Todo:

        Add marker channels
    """
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, output=True)
            for k in [1,2]
        }

        self._channels = {
            **self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True)
        }

        # TODO: Add pulse implementations
        self.pulse_implementations = []

        self.add_parameter('final_delay',
                           unit='s',
                           set_cmd=None,
                           initial_value=.1e-6,
                           doc='Time subtracted from each waveform to ensure '
                               'that it is finished once next trigger arrives.')
        self.add_parameter('active_channels',
                           set_cmd=None,
                           vals=Lists(Enum(1,2)))

    def get_additional_pulses(self, **kwargs):
        additional_pulses = []

        # Return empty list if no pulses are in the pulse sequence
        if not self.pulse_sequence or self.is_primary():
            return additional_pulses

        # TODO test if first waveform needs trigger as well




    def setup(self, is_primary=False, **kwargs):
        if is_primary:
            logger.warning('AWG520 cannot function as primary instrument')

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()


class DCPulseImplementation(PulseImplementation):
    pass
