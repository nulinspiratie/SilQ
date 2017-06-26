import numpy as np
import logging

from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation


logger = logging.getLogger(__name__)

RESET_PHASE_FALSE = 0
RESET_PHASE_TRUE = 1

RESET_PHASE = RESET_PHASE_FALSE

DEFAULT_CH_INSTR = (0, 0, 0, 0, 0)


class PulseBlasterDDSInterface(InstrumentInterface):


    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            # Measured output ranged from -3V to 3 V @ 50 ohm Load.
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}',
                              id=k,
                              #output=(-3.0, 3.0)
                              output=True)
            for k in [0, 1]}

        self._used_frequencies = {ch: {} for ch in self._output_channels.keys()}
        self._used_phases      = {ch: {} for ch in self._output_channels.keys()}
        self._used_amplitudes  = {ch: {} for ch in self._output_channels.keys()}

        self._channels = {
            **self._output_channels,
            'software_trig_in': Channel(instrument_name=self.instrument_name(),
                                        name='software_trig_in',
                                        input_trigger=True)}

        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[])]

    def setup(self, final_instruction='loop', **kwargs):
        if not self.pulse_sequence:
            logger.warning('Cannot setup empty PulseSequence')
            return

        #Initial pulseblaster commands
        self.instrument.initialize()

        # Set channel registers for frequencies, phases, and amplitudes
        for channel in self.instrument.output_channels:
            # Channel name is apparently modified to include instrument name
            channel_name = f'ch{channel.idx}'

            frequencies = []
            phases = []
            amplitudes = []

            pulses = self.pulse_sequence.get_pulses(output_channel=channel_name)
            for pulse in pulses:
                if isinstance(pulse, SinePulseImplementation):
                    frequencies.append(pulse.frequency)
                    phases.append(pulse.phase)
                    amplitudes.append(pulse.amplitude)

            channel.frequencies(set(frequencies))
            channel.phases(set(phases))
            channel.amplitudes(set(amplitudes))


        # Determine points per time unit
        core_clock = self.instrument.core_clock.get_latest()
        # Factor of 2 needed because apparently the core clock is not the same
        # as the sampling rate
        # TODO check if this is correct
        us = 2 * core_clock # points per microsecond
        ms = us * 1e3 # points per millisecond

        # Iteratively increase time
        t = 0
        t_stop_max = max(self.pulse_sequence.t_stop_list)
        inst_list = []
        while t < t_stop_max:
            # find time of next event
            t_next = min(t_val for t_val in self.pulse_sequence.t_list
                         if t_val > t)

            # Send continue instruction until next event
            delay_duration = t_next - t
            delay_cycles = round(delay_duration * ms)
            # Either send continue command or long_delay command if the
            # delay duration is too long

            inst = ()
            # for each channel, search for active pulses and implement them
            for ch in sorted(self._output_channels.keys()):
                pulse = self.pulse_sequence.get_pulse(enabled=True,
                                                       t=t,
                                                       output_channel=ch)
                if pulse is not None:
                    inst = inst + tuple(pulse.implement())
                else:
                    inst = inst + tuple(self.DEFAULT_CH_INST)

            if delay_cycles < 1e9:
                inst = inst + (0, 'continue', 0, delay_cycles)
            else:
                # TODO: check to see if a call to long_delay sets the channel registers
                duration = round(delay_cycles - 100)
                divisor = int(np.ceil(duration / 1e9))
                delay = int(duration / divisor)
                inst = inst + (0, 'long_delay', divisor, delay)

            inst_list.append(inst)

            t = t_next

        # Add final instructions

        # Insert delay until end of pulse sequence
        # NOTE: This will disable all output channels and use default registers
        delay_duration = max(self.pulse_sequence.duration - t, 0)
        inst_default = DEFAULT_CH_INSTR + DEFAULT_CH_INSTR
        if delay_duration:
            delay_cycles = round(delay_duration * ms)
            if delay_cycles < 1e9:
                inst = inst_default + (0, 'continue', 0, delay_cycles)
            else:
                # TODO: check to see if a call to long_delay sets the channel registers
                duration = round(delay_cycles - 100)
                divisor = int(np.ceil(duration / 1e9))
                delay = int(duration / divisor)
                inst = inst_default + (0, 'long_delay', divisor, delay)

            t += self.pulse_sequence.duration

        inst_list.append(inst)
        inst = inst_default + (0, 'branch', 0, 50)
        inst_list.append(inst)
        self.instrument.program_pulse_sequence(inst_list)

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def get_final_additional_pulses(self, **kwargs):
        return []


class SinePulseImplementation(SinePulse, PulseImplementation):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SinePulse, **kwargs)

    @property
    def amplitude(self):
        return self.amplitude()

    def implement(self):

        # TODO: check if the output_channel_name is useful for our hash
        output_channel_name = self.connection.output['channel'].id
        inst_slice = (
            self._used_frequencies[output_channel_name][self.frequency],
            self._used_phases[output_channel_name][self.phase],
            self._used_amplitudes[output_channel_name][self.amplitude],
            1, # Enable channel
            RESET_PHASE)
        return inst_slice

#TODO: implement a DC pulse implementation and trigger pulse implementation to
#      be used in the instruction flags

#class TriggerPulseImplementation(TriggerPulse, PulseImplementation):
#    def __init__(self, **kwargs):
#        PulseImplementation.__init__(self, pulse_class=TriggerPulse, **kwargs)
#
#    @property
#    def amplitude(self):
#        return self.connection.output['channel'].output_TTL[1]
#
#    def implement(self):
#        output_channel_name = self.connection.output['channel'].name
#        # Split channel number from string (e.g. "ch3" -> 3)
#        output_channel = int(output_channel_name[2])
#        channel_value = 2**(output_channel-1)
#        return channel_value

#class MarkerPulseImplementation(MarkerPulse, PulseImplementation):
#    def __init__(self, **kwargs):
#        PulseImplementation.__init__(self, pulse_class=MarkerPulse, **kwargs)
#
#    @property
#    def amplitude(self):
#        return self.connection.output['channel'].output_TTL[1]
#
#    def implement(self):
#        output_channel_name = self.connection.output['channel'].name
#        # Split channel number from string (e.g. "ch3" -> 3)
#        output_channel = int(output_channel_name[2])
#        channel_value = 2**(output_channel-1)
#        return channel_value
