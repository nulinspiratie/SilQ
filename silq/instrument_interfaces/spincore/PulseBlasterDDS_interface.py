import numpy as np
import warnings

from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.pulses import SinePulse, DCPulse, TriggerPulse, MarkerPulse, \
    TriggerWaitPulse, PulseImplementation

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

RESET_PHASE_FALSE = 0
RESET_PHASE_TRUE  = 1

RESET_PHASE = RESET_PHASE_FALSE

DEFAULT_CH_INSTR = (0,0,0,0,0)

class PulseBlasterDDSInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            # Measured output ranged from -3V to 3 V @ 50 ohm Load.
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                                      name='ch{}'.format(k), 
                                      id=k,
                                      #output=(-3.0, 3.0)
                                      output=True
                                     )
            for k in [1, 2]
        }

        self._used_frequencies = {}
        self._used_phases      = {}
        self._used_amplitudes  = {}
        for ch in sorted(self._output_channels.keys()):
            self._used_frequencies[ch] = {}
            self._used_phases[ch]      = {}
            self._used_amplitudes[ch]  = {}

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
        if not self._pulse_sequence:
            warnings.warn('Cannot setup empty PulseSequence', RuntimeWarning)
            return

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


        freq_i  = [0,0]
        phase_i = [0,0]
        amp_i   = [0,0]

        for ch in sorted(self._output_channels.keys()):
            ch_num = ch[2:] # Get the numerical channel number
            for pulse in self._pulse_sequence.get_pulses(output_channel=ch):
                n = ch_num -1 # NOTE: the driver starts counting from 0
                if isinstance(pulse, SinePulse):
                    # TODO: Put error when max number of freqs exceeded
                    if pulse.frequency not in self._used_frequencies[ch] and \
                       freq_i < self.instrument.N_FREQ_REGS:
                        self._used_frequencies[ch][pulse.frequency] = freq_i
                        # Set the instrument parameter
                        self.instrument.parameters[
                           'frequency_n{}_r{}'.format(n, freq_i)
                        ].set(pulse.frequency)
                        freq_i[n] = freq_i[n] + 1

                    if pulse.phase not in self._used_phases[ch] and \
                       phase_i < self.instrument.N_PHASE_REGS:
                        self._used_phases[ch][pulse.phase] = phase_i
                        # Set the instrument parameter
                        self.instrument.parameters[
                           'phase_n{}_r{}'.format(n, phase_i)
                        ].set(pulse.phase)
                        phase_i[n] = phase_i[n] + 1

                    if pulse.amplitude not in self._used_amplitudes[ch] and \
                       amp_i < self.instrument.N_AMPLITUDE_REGS:
                        self._used_amplitudes[ch][pulse.amplitude] = amp_i
                        # Set the instrument parameter
                        self.instrument.parameters[
                           'amplitude_n{}_r{}'.format(n, amp_i)
                        ].set(pulse.amplitude)
                        amp_i[n] = amp_i[n] + 1

        # Iteratively increase time
        t = 0
        t_stop_max = max(self._pulse_sequence.t_stop_list)
        inst_list = []
        while t < t_stop_max:
            # NOTE: there are no input pulses to the DDS
            #active_input_pulses = [pulse for pulse
            #                       in self._input_pulse_sequence
            #                       if pulse.t_start == t]
            #for input_pulse in active_input_pulses:
            #    if isinstance(input_pulse,TriggerWaitPulse):
            #        self.instrument.send_instruction(0,'wait', 0, 50)

            # find time of next event
            t_next = min(t_val for t_val in self._pulse_sequence.t_list
                         if t_val > t)

            # Send continue instruction until next event
            delay_duration = t_next - t
            delay_cycles = round(delay_duration * ms)
            # Either send continue command or long_delay command if the
            # delay duration is too long

            inst = ()
            # for each channel, search for active pulses and implement them
            for ch in sorted(self._output_channels.keys()):
                pulse = self._pulse_sequence.get_pulse(enabled=True,
                                                       t=t,
                                                       output_channel=ch)
                if pulse is not None:
                    inst = inst + tuple(pulse.implement())
                else:
                    inst = inst + tuple(DEFAULT_CH_INST)

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
        delay_duration = max(self._pulse_sequence.duration - t, 0)
        inst_default = DEFAULT_CH_INST + DEFAULT_CH_INST
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

            t += self._pulse_sequence.duration

        inst_list.append(inst)
        inst = inst_default + (0, 'branch', 0, 50)
        inst_list.append(inst)
        self.instrument.program_pulse_sequence(inst_list)
        #self.instrument.send_instruction(0, 'branch', 0, 50)


    def start(self):
        self.instrument.start()

    def stop(self):
        self.instrument.stop()

    def get_additional_pulses(self, **kwargs):
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
