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
        self.add_parameter('trigger_in_duration',
                           unit='s',
                           set_cmd=None,
                           initial_value=.1e-6)
        self.add_parameter('active_channels',
                           set_cmd=None,
                           vals=Lists(Enum(1,2)))

    def get_additional_pulses(self, **kwargs):
        # Return empty list if no pulses are in the pulse sequence
        if not self.pulse_sequence or self.is_primary():
            return []

        active_channels = {pulse.connection.output['channel']
                           for pulse in self.pulse_sequence}

        # Add DCPulse(amplitude=0) for any gaps between pulses
        gap_pulses = []
        for channel in active_channels:
            pulses = sorted(self.pulse_sequence.get_pulses(output_channel=channel),
                            key=lambda pulse: pulse.t_start)
            t = 0
            for pulse in pulses:
                if pulse.t_start < t:
                    raise RuntimeError(f'Pulse {pulse} starts before previous'
                                       f'pulse is finished')
                elif pulse.t_start > t:
                    # Add DCPulse to fill gap between pulses
                    gap_pulse = DCPulse(t_start=t,
                                        t_stop=pulse.t_start,
                                        amplitude=0)
                    gap_pulses.append(self.get_pulse_implementation(gap_pulse))
                t = pulse.t_start

            if t < self.pulse_sequence.duration:
                # Add DCPulse to fill gap between pulses
                gap_pulse = DCPulse(t_start=t,
                                    t_stop=self.pulse_sequence.duration,
                                    amplitude=0)
                gap_pulses.append(self.get_pulse_implementation(gap_pulse))

        self.pulse_sequence.add(*gap_pulses)

        # If a pulse starts on one channel and needs a trigger while another
        # pulse is still active on the other channel, this would cause the other
        # pulse to move onto the next pulse prematurely. This only happens if
        # the other pulse finished its waveform and is waiting for a trigger.
        # Here we check that this does not occur.
        if len(active_channels) > 1:
            for t_start in self.pulse_sequence.t_start_list:
                if any(pulse.t_start < t_start < pulse.t_stop
                       and not pulse.implementation.is_full_waveform
                       for pulse in self.pulse_sequence):
                    raise RuntimeError(f'At t={t_start} s, a new pulse starts'
                                       f'while another pulse is still active.')
                    #TODO Implement splitting of pulses

        # TODO test if first waveform needs trigger as well
        additional_pulses = [
            TriggerPulse(t_start=t_start,
                         duration=self.trigger_in_duration(),
                         connection_requirements={
                             'input_instrument': self.instrument_name(),
                             'trigger': True})
            for t_start in self.pulse_sequence.t_start_list
        ]

        return additional_pulses

    def setup(self, is_primary=False, **kwargs):
        if is_primary:
            raise RuntimeError('AWG520 cannot function as primary instrument')

        

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()


class DCPulseImplementation(PulseImplementation):
    pass
