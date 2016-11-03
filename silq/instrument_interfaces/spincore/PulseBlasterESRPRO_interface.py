from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, MarkerPulse, PulseImplementation

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class PulseBlasterESRPROInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            # Measured output TTL is half of 3.3V
            'ch{}'.format(k): Channel(instrument_name=self.name,
                                      name='ch{}'.format(k), id=k,
                                      output_TTL=(0, 3.3/2))
            for k in [1, 2, 3, 4]}
        self._channels = {**self._output_channels}

        self.pulse_implementations = [
            TriggerPulseImplementation(
                pulse_requirements=[]
            ),
            MarkerPulseImplementation(
                pulse_requirements=[]
            )
        ]

    def setup(self, **kwargs):
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

        pulses = self._pulse_sequence.pulses

        if pulses:
            # Iteratively increase time
            t = 0
            t_start_list = [pulse.t_start for pulse in pulses]
            t_stop_list = [pulse.t_stop for pulse in pulses]
            t_list = t_start_list + t_stop_list
            t_stop_max = max(t_stop_list)
            while t < t_stop_max:
                # Segment remaining pulses into next pulses and others
                active_pulses = [pulse for pulse in pulses
                                 if pulse.t_start <= t < pulse.t_stop]
                if not active_pulses:
                    total_channel_value = 0
                else:
                    total_channel_value = sum([pulse.implement()
                                               for pulse in active_pulses])

                # find time of next event
                t_next = min(t_val for t_val in t_list if t_val > t)

                # Send wait instruction until next event
                wait_duration = t_next - t
                wait_cycles = round(wait_duration * ms)

                self.instrument.send_instruction(total_channel_value,
                                                 'continue', 0, wait_cycles)
                t += wait_duration
            else:
                # Add final instructions

                # Wait until end of pulse sequence
                wait_duration = self._pulse_sequence.duration - t
                if wait_duration:
                    wait_cycles = round(wait_duration * ms)
                    self.instrument.send_instruction(0, 'continue', 0,
                                                     wait_cycles)
                    t += wait_duration

                total_channel_value = 0
                self.instrument.send_instruction(total_channel_value,
                                                 'branch', 0, 50)

        self.instrument.stop_programming()

    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def get_final_additional_pulses(self):
        return []


class TriggerPulseImplementation(TriggerPulse, PulseImplementation):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=TriggerPulse, **kwargs)

    @property
    def amplitude(self):
        return self.connection.output['channel'].output_TTL[1]

    def implement(self):
        output_channel_name = self.connection.output['channel'].name
        # Split channel number from string (e.g. "ch3" -> 3)
        output_channel = int(output_channel_name[2])
        channel_value = 2**(output_channel-1)
        return channel_value

class MarkerPulseImplementation(MarkerPulse, PulseImplementation):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=MarkerPulse, **kwargs)

    @property
    def amplitude(self):
        return self.connection.output['channel'].output_TTL[1]

    def implement(self):
        output_channel_name = self.connection.output['channel'].name
        # Split channel number from string (e.g. "ch3" -> 3)
        output_channel = int(output_channel_name[2])
        channel_value = 2**(output_channel-1)
        return channel_value