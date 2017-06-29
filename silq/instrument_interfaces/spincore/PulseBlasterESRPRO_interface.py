import numpy as np

from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.pulses import DCPulse, TriggerPulse, MarkerPulse, TriggerWaitPulse, \
    PulseImplementation


class PulseBlasterESRPROInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            # Measured output TTL is half of 3.3V
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k - 1, output_TTL=(0, 3.3 / 2))
            for k in [1, 2, 3, 4]}
        self._channels = {
            **self._output_channels,
            'software_trig_in': Channel(instrument_name=self.instrument_name(),
                                        name='software_trig_in',
                                        input_trigger=True)}

        self.pulse_implementations = [TriggerPulseImplementation(),
                                      MarkerPulseImplementation()]

    def setup(self, final_instruction='loop', output_connections=[], **kwargs):
        # Determine points per time unit
        core_clock = self.instrument.core_clock.get_latest()
        # Factor of 2 needed because apparently the core clock is not the same
        # as the sampling rate
        # TODO check if this is correct
        us = 2 * core_clock # points per microsecond
        ms = us * 1e3 # points per millisecond

        #Initial pulseblaster commands
        self.instrument.stop()

        # Set up instrument, includes counting boards
        self.instrument.setup(initialize=False)


        output_channels = [connection.output['channel']
                           for connection in output_connections]
        assert len(output_channels) == len(output_connections), \
            "Number of output channels and connections do not match"

        inactive_channel_mask = sum(2**connection.output['channel'].id
                                    if connection.input['channel'].invert else 0
                                    for connection in output_connections)

        self.instrument.start_programming()

        if self.pulse_sequence:
            # Iteratively increase time
            t = 0
            t_stop_max = max(self.pulse_sequence.t_stop_list)

            while t < t_stop_max:
                channel_mask = sum(pulse.implementation.implement(t=t)
                                   for pulse in self.pulse_sequence)

                # Check for input pulses, such as waiting for software trigger
                # TODO check for better way to check active input pulses
                active_input_pulses = self.input_pulse_sequence.get_pulses(
                    t_start=t)
                if [p for p in active_input_pulses
                    if isinstance(p, TriggerWaitPulse)]:
                    self.instrument.send_instruction(inactive_channel_mask,
                                                     'wait', 0, 50)

                # find time of next event
                t_next = min(t_val for t_val in self.pulse_sequence.t_list
                             if t_val > t)

                # Send wait instruction until next event
                wait_duration = t_next - t
                wait_cycles = round(wait_duration * ms)
                # Either send continue command or long_delay command if the
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
                            inactive_channel_mask, 'continue', 0, wait_cycles)
                    else:
                        self.instrument.send_instruction(
                            inactive_channel_mask, 'continue', 0, 100)
                        duration = round(wait_cycles - 100)
                        divisor = int(np.ceil(duration / 1e9))
                        delay = int(duration / divisor)
                        self.instrument.send_instruction(
                            inactive_channel_mask, 'long_delay', divisor, delay)

                self.instrument.send_instruction(
                    inactive_channel_mask, 'branch', 0, 50)

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

    def implement(self, t):
        output_channel = self.connection.output['channel']
        input_channel = self.connection.input['channel']
        channel_value = 2 ** output_channel.id

        if t >= self.t_start and t < self.t_stop:
            return 0 if input_channel.invert else channel_value
        else:
            return channel_value if input_channel.invert else 0

class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse

    @property
    def amplitude(self):
        return self.connection.output['channel'].output_TTL[1]

    def implement(self, t):
        output_channel = self.connection.output['channel']
        input_channel = self.connection.input['channel']
        channel_value = 2 ** output_channel.id

        if t >= self.t_start and t < self.t_stop:
            return 0 if input_channel.invert else channel_value
        else:
            return channel_value if input_channel.invert else 0
