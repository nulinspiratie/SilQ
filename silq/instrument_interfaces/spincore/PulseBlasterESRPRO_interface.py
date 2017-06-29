import numpy as np

from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, MarkerPulse, TriggerWaitPulse, \
    PulseImplementation

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class PulseBlasterESRPROInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            # Measured output TTL is half of 3.3V
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                                      name='ch{}'.format(k), id=k,
                                      output_TTL=(0, 3.3/2))
            for k in [1, 2, 3, 4]}
        self._channels = {
            **self._output_channels,
            'software_trig_in': Channel(instrument_name=self.instrument_name(),
                                        name='software_trig_in',
                                        input_trigger=True)}

        self.pulse_implementations = [
            TriggerPulseImplementation(
                pulse_requirements=[]
            ),
            MarkerPulseImplementation(
                pulse_requirements=[]
            )
        ]

    def setup(self, final_instruction='loop', **kwargs):
        # Determine points per time unit
        core_clock = self.instrument.core_clock.get_latest()
        # Factor of 2 needed because apparently the core clock is not the same
        # as the sampling rate
        # TODO check if this is correct
        us = 2 * core_clock # points per microsecond
        ms = us * 1e3 # points per millisecond

        #Initial pulseblaster commands
        self.instrument.stop()
        self.instrument.detect_boards()
        self.instrument.select_board(0)
        self.instrument.start_programming()

        if self.pulse_sequence:
            # Iteratively increase time
            t = 0
            t_stop_max = max(self.pulse_sequence.t_stop_list)

            # Generate pulses once for speedup reasons
            loop_idx = 0
            while t < t_stop_max:
                loop_idx += 1
                # Check for input pulses, such as waiting for software trigger
                # TODO check for better way to check active input pulses
                active_input_pulses = self.input_pulse_sequence.get_pulses(
                    t_start=t)
                for input_pulse in active_input_pulses:
                    if isinstance(input_pulse,TriggerWaitPulse):
                        self.instrument.send_instruction(0,'wait', 0, 50)

                # Segment remaining pulses into next pulses and others
                active_pulses = self.pulse_sequence.get_pulses(t=t)

                if not active_pulses:
                    channel_mask = 0
                else:
                    channel_mask = sum(
                        [pulse.implement() for pulse in active_pulses])

                # find time of next event
                t_next = min(t_val for t_val in self.pulse_sequence.t_list
                             if t_val > t)

                # Send wait instruction until next event
                wait_duration = t_next - t
                wait_cycles = round(wait_duration * ms)
                # Either send continue commandm or long_delay command if the
                # wait duration is too long
                if wait_cycles < 1e9:
                    self.instrument.send_instruction(
                        channel_mask, 'continue', 0, wait_cycles)
                else:
                    self.instrument.send_instruction(
                        channel_mask, 'continue', 0, 100)
                    duration = round(wait_cycles - 100)
                    divisor = int(np.ceil(duration / 1e9))
                    delay = int(duration / divisor)
                    self.instrument.send_instruction(
                        channel_mask, 'long_delay', divisor, delay)

                t = t_next
            else:
                # Add final instructions

                # Wait until end of pulse sequence
                wait_duration = max(self.pulse_sequence.duration - t, 0)
                if wait_duration:
                    wait_cycles = round(wait_duration * ms)
                    if wait_cycles < 1e9:
                        self.instrument.send_instruction(
                            0, 'continue', 0, wait_cycles)
                    else:
                        self.instrument.send_instruction(
                            0, 'continue', 0, 100)
                        duration = round(wait_cycles - 100)
                        divisor = int(np.ceil(duration / 1e9))
                        delay = int(duration / divisor)
                        self.instrument.send_instruction(
                            0, 'long_delay', divisor, delay)

                    t += self.pulse_sequence.duration

                self.instrument.send_instruction(0, 'branch', 0, 50)

        self.instrument.stop_programming()

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def get_additional_pulses(self, **kwargs):
        return []


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse

    @property
    def amplitude(self):
        return self.connection.output['channel'].output_TTL[1]

    def implement(self):
        output_channel_name = self.connection.output['channel'].name
        # Split channel number from string (e.g. "ch3" -> 3)
        output_channel = int(output_channel_name[2])
        channel_value = 2**(output_channel-1)
        return channel_value

class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse

    @property
    def amplitude(self):
        return self.connection.output['channel'].output_TTL[1]

    def implement(self):
        output_channel_name = self.connection.output['channel'].name
        # Split channel number from string (e.g. "ch3" -> 3)
        output_channel = int(output_channel_name[2])
        channel_value = 2**(output_channel-1)
        return channel_value