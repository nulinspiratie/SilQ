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

    def setup(self, repeat=True, output_connections=[], **kwargs):
        # Determine points per time unit
        core_clock = self.instrument.core_clock.get_latest()
        # Factor of 2 needed because apparently the core clock is not the same
        # as the sampling rate
        # TODO check if this is correct
        # convert core_clock in MHz to sample_rate in Hz
        sample_rate = 2 * core_clock * 1e6

        #Initial pulseblaster commands
        self.instrument.stop()

        # Set up instrument, includes counting boards
        self.instrument.setup(initialize=False)


        output_channels = [connection.output['channel']
                           for connection in output_connections]
        assert len(output_channels) == len(output_connections), \
            "Number of output channels and connections do not match"

        self.instrument.start_programming()

        if not self.pulse_sequence:
            # No pulse sequence, stop programming
            self.instrument.stop_programming()
            return

        instructions = []

        # Determine signal to send when all channels are inactive (low).
        # This is not necessarily zero, as some channels are triggered when the
        # channgel voltage is below a threshold (e.g. PB DDS instrument).
        inactive_channel_mask = sum(2**connection.output['channel'].id
                                    if connection.input['channel'].invert else 0
                                    for connection in output_connections)
        if inactive_channel_mask != 0:
            # Some channels must have high signal to not trigger it.
            # Wait instructions must be preceded by another instruction.
            # These instructions are only performed once: loops return to third instruction
            instructions.append((inactive_channel_mask, 'continue', 0, 100))
            # Wait for software trigger (another call to pulseblaster.start())
            instructions.append((inactive_channel_mask, 'wait', 0, 100))

        # Iteratively increase time
        t = 0
        t_stop_max = max(self.pulse_sequence.t_stop_list)

        while t < t_stop_max:
            channel_mask = sum(pulse.implementation.implement(t=t)
                               for pulse in self.pulse_sequence)

            # Check for input pulses, such as waiting for software trigger
            # TODO check for better way to check active input pulses
            active_input_pulses = self.input_pulse_sequence.get_pulses(t_start=t)
            if [p for p in active_input_pulses
                if isinstance(p, TriggerWaitPulse)]:
                instructions.append((inactive_channel_mask, 'wait', 0, 50))

            # find time of next event
            t_next = min(t_val for t_val in self.pulse_sequence.t_list
                         if t_val > t)

            # Send wait instruction until next event
            wait_duration = t_next - t
            wait_cycles = round(wait_duration * sample_rate)
            # Either send continue command or long_delay command if the
            # wait duration is too long
            if wait_cycles < 1e9:
                instructions.append((channel_mask, 'continue', 0, wait_cycles))
            else:
                instructions.append((channel_mask, 'continue', 0, 100))
                duration = round(wait_cycles - 100)
                divisor = int(np.ceil(duration / 1e9))
                delay = int(duration / divisor)
                instructions.append((channel_mask, 'long_delay', divisor, delay))

            t = t_next
        else:
            # Add final instructions

            # Wait until end of pulse sequence
            wait_duration = max(self.pulse_sequence.duration - t, 0)
            if wait_duration:
                wait_cycles = round(wait_duration * sample_rate)
                if wait_cycles < 1e9:
                    instructions.append((inactive_channel_mask, 'continue', 0, wait_cycles))
                else:
                    instructions.append((inactive_channel_mask, 'continue', 0, 100))
                    duration = round(wait_cycles - 100)
                    divisor = int(np.ceil(duration / 1e9))
                    delay = int(duration / divisor)
                    instructions.append((inactive_channel_mask, 'long_delay', divisor, delay))

            if repeat:
                if inactive_channel_mask == 0:
                    # Return back to first action
                    instructions.append((inactive_channel_mask, 'branch', 0, 50))
                else:
                    # Return to third instruction, since first two are one-time actions
                    instructions.append((inactive_channel_mask, 'branch', 2, 50))
            else:
                instructions.append((inactive_channel_mask, 'stop', 0, 50))


        self.instrument.send_instructions(*instructions)
        self.instrument.stop_programming()

        if inactive_channel_mask != 0:
            # Add flag to trigger pulseblaster again after all other instruments
            # are started (done via PB.start()).
            return {'post_start_actions': [self.start]}

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def get_additional_pulses(self):
        return []


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse

    def target_pulse(self, pulse, interface, **kwargs):
        targeted_pulse = super().target_pulse(pulse, interface, **kwargs)
        amplitude = targeted_pulse.connection.output['channel'].output_TTL[1]
        targeted_pulse.amplitude = amplitude
        return targeted_pulse

    def implement(self, t):
        output_channel = self.pulse.connection.output['channel']
        input_channel = self.pulse.connection.input['channel']
        channel_value = 2 ** output_channel.id

        if t >= self.pulse.t_start and t < self.pulse.t_stop:
            return 0 if input_channel.invert else channel_value
        else:
            return channel_value if input_channel.invert else 0


class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse

    def target_pulse(self, pulse, interface, **kwargs):
        targeted_pulse = super().target_pulse(pulse, interface, **kwargs)
        amplitude = targeted_pulse.connection.output['channel'].output_TTL[1]
        targeted_pulse.amplitude = amplitude
        return targeted_pulse

    def implement(self, t):
        output_channel = self.pulse.connection.output['channel']
        input_channel = self.pulse.connection.input['channel']
        channel_value = 2 ** output_channel.id

        if t >= self.pulse.t_start and t < self.pulse.t_stop:
            return 0 if input_channel.invert else channel_value
        else:
            return channel_value if input_channel.invert else 0
