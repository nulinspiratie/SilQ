import numpy as np

from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, MarkerPulse, TriggerWaitPulse, \
    PulseImplementation

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class PulseBlasterDDSInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._used_frequencies = {}
        self._used_phases      = {}
        self._used_amplitudes  = {}

        self._output_channels = {
            # Measured output ranged from -3V to 3 V @ 50 ohm Load.
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                                      name='ch{}'.format(k), 
                                      id=k,
                                      output=(-3.0, 3.0)
                                     )
            for k in [1, 2]
        }
        self._channels = {
            **self._output_channels,
            'software_trig_in': Channel(instrument_name=self.instrument_name(),
                                        name='software_trig_in',
                                        input_trigger=True)}

        self.pulse_implementations = [
#            TriggerPulseImplementation(
#                pulse_requirements=[]
#            ),
#            MarkerPulseImplementation(
#                pulse_requirements=[]
#            )
            SinePulseImplementation(
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
        self.instrument.detect_boards()
        self.instrument.select_board(0)
        
        if self._pulse_sequence:
            freq_i  = 0
            phase_i = 0
            amp_i   = 0
            for pulse in self._pulse_sequence:
                n = 0 # TODO: this is a hard coded channel
                if isinstance(pulse, SinePulse):
                    # TODO: Put error when max number of freqs exceeded 
                    if pulse.frequency not in self._used_frequencies and
                       freq_i < self.instrument.N_FREQ_REGS:
                        self._used_frequencies[pulse.frequency] = freq_i
                        # Set the instrument parameter
                        setter = getattr(self.instrument, 
                                         'frequency_n{}_r{}.set'.format(n, freq_i))
                        setter(pulse.frequency)
                        freq_i = freq_i + 1

                    if pulse.phase not in self._used_phases and
                       phase_i < self.instrument.N_PHASE_REGS:
                        self._used_phases[pulse.phase] = phase_i
                        # Set the instrument parameter
                        setter = getattr(self.instrument, 
                                         'phase_n{}_r{}.set'.format(n, phase_i))
                        setter(pulse.phase)
                        phase_i = phase_i + 1

                    if pulse.amplitude not in self._used_amplitudes 
                       and amp_i < self.instrument.N_AMPLITUDE_REGS:
                        self._used_amplitudes[pulse.amplitude] = amp_i
                        # Set the instrument parameter
                        setter = getattr(self.instrument, 
                                         'amplitude_n{}_r{}.set'.format(n, amp_i))
                        setter(pulse.amplitude)
                        amp_i = amp_i + 1
                        
        self.instrument.start_programming()
        if self._pulse_sequence:
            # Iteratively increase time
            t = 0
            t_stop_max = max(self._pulse_sequence.t_stop_list)
            inst_list = []
            while t < t_stop_max:
                active_input_pulses = [pulse for pulse
                                       in self._input_pulse_sequence
                                       if pulse.t_start == t]
                for input_pulse in active_input_pulses:
                    if isinstance(input_pulse,TriggerWaitPulse):
                        self.instrument.send_instruction(0,'wait', 0, 50)

                # Segment remaining pulses into next pulses and others
                active_pulses = [pulse for pulse in self._pulse_sequence
                                 if pulse.t_start <= t < pulse.t_stop]
                if not active_pulses:
                    channel_mask = 0
                else:
                # TODO this will need to be modified for the DDS version
                # there are 2 DDS channels, each with set frequency registers,
                # should check whether these values can be achieved
                    channel_mask = sum(
                        [pulse.implement() for pulse in active_pulses])

                # find time of next event
                t_next = min(t_val for t_val in self._pulse_sequence.t_list
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
            
            # Add final instructions

            # Wait until end of pulse sequence
            wait_duration = max(self._pulse_sequence.duration - t, 0)
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

                t += self._pulse_sequence.duration

            self.instrument.send_instruction(0, 'branch', 0, 50)

        self.instrument.stop_programming()

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
        # TODO implement this function
        amp = self.getVoltage()
        return 0

    def implement(self):
        output_channel_name = self.connection.output['channel'].name
        # Split channel number from string (e.g. "ch3" -> 3)
        output_channel = int(output_channel_name[2])
        channel_value = 2**(output_channel-1)
        return channel_value

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
