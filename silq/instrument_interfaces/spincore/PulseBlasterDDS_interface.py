import numpy as np
import logging
from typing import List

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, SinePulse, PulseImplementation, TriggerPulse


logger = logging.getLogger(__name__)

RESET_PHASE_FALSE = 0
RESET_PHASE_TRUE = 1

RESET_PHASE = RESET_PHASE_FALSE

DEFAULT_CH_INSTR = (0, 0, 0, 0, 0)
DEFAULT_INSTR = DEFAULT_CH_INSTR + DEFAULT_CH_INSTR


class PulseBlasterDDSInterface(InstrumentInterface):
    """ Interface for the Pulseblaster DDS

    When a `PulseSequence` is targeted in the `Layout`, the
    pulses are directed to the appropriate interface. Each interface is
    responsible for translating all pulses directed to it into instrument
    commands. During the actual measurement, the instrument's operations will
    correspond to that required by the pulse sequence.

    One important issue with the DDS is that it requires an inverted trigger,
    i.e. high voltage is the default, and a low voltage indicates a trigger.
    Not every interface has been programmed to handle this (so far only the
    PulseBlaster ESRPRO has).

    The interface also contains a list of all available channels in the
    instrument.

    Args:
        instrument_name: name of instrument for which this is an interface

    Note:
        For a given instrument, its associated interface can be found using
            `get_instrument_interface`
    """

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
                pulse_requirements=[('amplitude', {'min': 0, 'max': 1/0.6})])]

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List containing trigger pulse if not primary instrument
        """
        # Request one trigger at the start if not primary
        if not self.is_primary():
            return [TriggerPulse(t_start=0,
                                 connection_requirements={
                                     'input_instrument': self.instrument_name(),
                                     'trigger': True})]
        else:
            return []

    def setup(self,
              repeat: bool = True,
              **kwargs):
        """Set up instrument after layout has been targeted by pulse sequence.

        Args:
            repeat: Repeat the pulse sequence indefinitely. If False, calling
                `Layout.start` will only run the pulse sequence once.
            **kwargs: Ignored kwargs passed by layout.

        Returns:
            setup flags (see ``Layout.flags``)

        """
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
                if isinstance(pulse, SinePulse):
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

        s_to_ns = 1e9 # instruction delays expressed in ns

        # Iteratively increase time
        t = 0
        t_stop_max = max(self.pulse_sequence.t_stop_list)
        inst_list = []

        if not self.is_primary():
            # Wait for trigger
            inst_list.append(DEFAULT_INSTR + (0, 'wait', 0, 100))

        while t < t_stop_max:
            # find time of next event
            t_next = min(t_val for t_val in self.pulse_sequence.t_list
                         if t_val > t)

            # Send continue instruction until next event
            delay_duration = t_next - t
            delay_cycles = round(delay_duration * s_to_ns)
            # Either send continue command or long_delay command if the
            # delay duration is too long

            inst = ()
            # for each channel, search for active pulses and implement them
            for ch in sorted(self._output_channels.keys()):
                instrument_channel = self.instrument.output_channels[ch]

                pulse = self.pulse_sequence.get_pulse(enabled=True,
                                                      t=t,
                                                      output_channel=ch)
                if pulse is not None:
                    pulse_implementation = pulse.implementation.implement(
                        frequencies=instrument_channel.frequencies(),
                        phases=instrument_channel.phases(),
                        amplitudes=instrument_channel.amplitudes())
                    inst = inst + pulse_implementation
                else:
                    inst = inst + DEFAULT_CH_INSTR

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

        if self.is_primary():
            # Insert delay until end of pulse sequence
            # NOTE: This will disable all output channels and use default registers
            delay_duration = self.pulse_sequence.duration + self.pulse_sequence.final_delay - t
            if delay_duration > 1e-11:
                delay_cycles = round(delay_duration * s_to_ns)
                if delay_cycles < 1e9:
                    inst = DEFAULT_INSTR + (0, 'continue', 0, delay_cycles)
                else:
                    # TODO: check to see if a call to long_delay sets the channel registers
                    duration = round(delay_cycles - 100)
                    divisor = int(np.ceil(duration / 1e9))
                    delay = int(duration / divisor)
                    inst = DEFAULT_INSTR + (0, 'long_delay', divisor, delay)

                inst_list.append(inst)

        if repeat:
            # Loop back to beginning (wait if not primary)
            inst_list.append(DEFAULT_INSTR + (0, 'branch', 0, 100))
        else:
            # Stop pulse sequence
            inst_list.append(DEFAULT_INSTR + (0, 'stop', 0, 100))


        # Note that this command does not actually send anything to the DDS,
        # this is done when DDS.start is called
        self.instrument.instruction_sequence(inst_list)

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

        if not self.is_primary():
            # Return flag to ensure this instrument is started after its
            # triggering instrument. This is because it is triggered when its
            # trigger voltage reaches below a threshold, meaning that the triggering
            # voltage must be high when this instrument is started.
            return {'start_last': self}


    def start(self):
        """Start instrument"""
        self.instrument.start()

    def stop(self):
        """Stop instrument"""
        self.instrument.stop()


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def implement(self, frequencies, phases, amplitudes):
        frequency_idx = frequencies.index(self.pulse.frequency) # MHz
        phase_idx = phases.index(self.pulse.phase)
        amplitude_idx = amplitudes.index(self.pulse.amplitude)

        inst_slice = (
            frequency_idx,
            phase_idx,
            amplitude_idx,
            1, # Enable channel
            RESET_PHASE)
        return inst_slice
