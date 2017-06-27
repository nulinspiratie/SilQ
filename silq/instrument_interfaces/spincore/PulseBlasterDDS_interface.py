import numpy as np
import logging

from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation, TriggerPulse


logger = logging.getLogger(__name__)

RESET_PHASE_FALSE = 0
RESET_PHASE_TRUE = 1

RESET_PHASE = RESET_PHASE_FALSE

DEFAULT_CH_INSTR = (0, 0, 0, 0, 0)
DEFAULT_INSTR = DEFAULT_CH_INSTR + DEFAULT_CH_INSTR

class PulseBlasterDDSInterface(InstrumentInterface):


    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)


        self._output_channels = {
            # Measured output ranged from -3V to 3 V @ 50 ohm Load.
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}',
                              id=k-1, # id is 0-based due to spinapi DLL
                              #output=(-3.0, 3.0)
                              output=True)
            for k in [1, 2]}

        self._channels = {
            **self._output_channels,
            'software_trig_in': Channel(instrument_name=self.instrument_name(),
                                        name='software_trig_in',
                                        input_trigger=True),
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in',
                               input_trigger=True,
                               invert=True)} # Going from high to low triggers

        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[])]

    def setup(self, final_instruction='loop', is_primary=True, **kwargs):
        #Initial pulseblaster commands
        self.instrument.setup()

        # Set channel registers for frequencies, phases, and amplitudes
        for channel in self.instrument.output_channels:
            frequencies = []
            phases = []
            amplitudes = []

            pulses = self.pulse_sequence.get_pulses(
                output_channel=channel.short_name)
            for pulse in pulses:
                if isinstance(pulse, SinePulseImplementation):
                    frequencies.append(pulse.frequency) # in MHz
                    phases.append(pulse.phase)
                    amplitudes.append(pulse.amplitude)
                else:
                    raise NotImplementedError(f'{pulse} not implemented')

            frequencies = list(set(frequencies))
            phases = list(set(phases))
            amplitudes = list(set(amplitudes))

            channel.frequencies(frequencies)
            channel.phases(phases)
            channel.amplitudes(amplitudes)

            self.instrument.set_frequencies(frequencies=frequencies,
                                            channel=channel.idx)
            self.instrument.set_phases(phases=phases, channel=channel.idx)
            self.instrument.set_amplitudes(amplitudes=amplitudes,
                                           channel=channel.idx)

        # Determine points per time unit
        core_clock = self.instrument.core_clock.get_latest()
        # Factor of 2 needed because apparently the core clock is not the same
        # as the sampling rate
        # TODO check if this is correct
        ms = 150e3 # points per millisecond

        # Iteratively increase time
        t = 0
        t_stop_max = max(self.pulse_sequence.t_stop_list)
        inst_list = []

        if not is_primary:
            # Wait for trigger
            inst_list.append(DEFAULT_INSTR + (0, 'wait', 0, 50))

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
                    inst = inst + tuple(pulse.implement(self))
                else:
                    inst = inst + tuple(DEFAULT_CH_INSTR)

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

        if is_primary:
            # Insert delay until end of pulse sequence
            # NOTE: This will disable all output channels and use default registers
            delay_duration = max(self.pulse_sequence.duration - t, 0)
            if delay_duration:
                delay_cycles = round(delay_duration * ms)
                if delay_cycles < 1e9:
                    inst = DEFAULT_INSTR + (0, 'continue', 0, delay_cycles)
                else:
                    # TODO: check to see if a call to long_delay sets the channel registers
                    duration = round(delay_cycles - 100)
                    divisor = int(np.ceil(duration / 1e9))
                    delay = int(duration / divisor)
                    inst = DEFAULT_INSTR + (0, 'long_delay', divisor, delay)

                inst_list.append(inst)

        # Loop back to beginning (wait if not primary)
        inst = DEFAULT_INSTR + (0, 'branch', 0, 50)
        inst_list.append(inst)

        # Note that this command does not actually send anything to the DDS,
        # this is done when DDS.start is called
        self.instrument.instruction_sequence(inst_list)

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def get_final_additional_pulses(self, **kwargs):
        return [TriggerPulse(t_start=0,
                             connection_requirements={
                                 'input_instrument': self.instrument_name(),
                                 'trigger': True})]


class SinePulseImplementation(SinePulse, PulseImplementation):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SinePulse, **kwargs)

    def implement(self, interface):

        channel_name = self.connection.output['channel'].name
        channel = interface.instrument.output_channels[channel_name]

        frequency_idx = channel.frequencies().index(self.frequency) # MHz
        phase_idx = channel.phases().index(self.phase)
        amplitude_idx = channel.amplitudes().index(self.amplitude)

        inst_slice = (
            frequency_idx,
            phase_idx,
            amplitude_idx,
            1, # Enable channel
            RESET_PHASE)
        return inst_slice