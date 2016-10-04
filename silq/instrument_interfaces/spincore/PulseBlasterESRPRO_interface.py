from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, PulseImplementation

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class PulseBlasterESRPROInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self.output_channels = {
            'ch{}'.format(k): Channel(self, name='ch{}'.format(k), id=k,
                                      output_trigger=True)
            for k in [1, 2, 3, 4]}
        self.channels = {**self.output_channels}

        self.pulse_implementations = [
            TriggerPulse.create_implementation(
                TriggerPulseImplementation,
                pulse_conditions=[]
            )
        ]

        self.add_parameter('ignore_first_trigger',
                           parameter_class=ManualParameter,
                           initial_value=False,
                           vals=vals.Bool())

    def setup(self):
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

        # Iteratively remove pulses from remaining_pulses as they are programmed
        remaining_pulses = self._pulse_sequence.pulses
        if remaining_pulses:
            # Determine trigger cycles
            trigger_duration = remaining_pulses[0].trigger_duration
            assert all([pulse.trigger_duration == trigger_duration
                        for pulse in remaining_pulses]), \
                "Cannot handle different pulse trigger durations yet."
            trigger_cycles = round(trigger_duration * us)

        t = 0
        while remaining_pulses:
            # Determine start of next pulse(s)
            t_start_list = [pulse.t_start for pulse in remaining_pulses]
            t_start_min = min(t_start_list)

            # Segment remaining pulses into next pulses and others
            active_pulses = [pulse for pulse in remaining_pulses
                             if pulse.t_start == t_start_min]
            remaining_pulses = [pulse for pulse in remaining_pulses
                                if pulse.t_start != t_start_min]

            if t_start_min > t:
                wait_duration = t_start_min - t
                wait_cycles = wait_duration * ms
                self.instrument.send_instruction(0, 'continue', 0, wait_cycles)

            # Ignore first trigger if parameter value is true.
            # Some sequence modes require the first trigger to be ignored
            if t_start_min == 0 and self.ignore_first_trigger():
                # TODO deal with this correctly
                pass

            total_channel_value = sum([self.implement_pulse(pulse)
                                       for pulse in active_pulses])
            self.instrument.send_instruction(total_channel_value, 'continue',
                                             0, trigger_cycles)
            t += trigger_duration

        self.instrument.stop_programming()


class TriggerPulseImplementation(PulseImplementation):
    def __init__(self, pulse_class, **kwargs):
        super().__init__(pulse_class, **kwargs)

    def implement_pulse(self, trigger_pulse):
        output_channel_name = trigger_pulse.connection.output['channel']

        # Split channel number from string (e.g. "ch3" -> 3)
        output_channel = int(output_channel_name[2])
        channel_value = 2**(output_channel-1)
        return channel_value