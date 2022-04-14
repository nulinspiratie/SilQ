from typing import List
import logging
from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, DCPulse, SinePulse, FrequencyRampPulse, \
    TriggerPulse, PulseImplementation, MarkerPulse
from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators as vals

logger = logging.getLogger(__name__)


class PCDDSInterface(InstrumentInterface):
    # Maximum pulse and instruction idx to be cleared.
    # Only used in decoupled mode
    pulse_max_idx = 200
    instruction_max_idx = 500

    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name=instrument_name, **kwargs)

        # Setup channels
        self._output_channels = {
            f'ch{k}': Channel(self.instrument_name(),
                              name=f'ch{k}', id=k)
            for k in range(4)}

        self._input_channels = {
            f'pxi{k}': Channel(self.instrument_name(),
                               name=f'pxi{k}', input=True)
            for k in range(4)}
        self._input_channels['trig_in'] =  Channel(self.instrument_name(),
                                                   name=f'trig_in', input=True)

        self._channels = {**self._output_channels, **self._input_channels}

        self.pulse_implementations = [
            DCPulseImplementation(),
            SinePulseImplementation(),
            FrequencyRampPulseImplementation(),
            MarkerPulseImplementation()
        ]

        self.use_trig_in = Parameter(
            initial_value=True,
            set_cmd=None,
            vals=vals.Bool(),
            docstring="Whether to use trig_in for triggering. " \
                      "All DDS channels listen simultaneosly to trig_in, " \
                      "while the pxi channels can trigger individual dds channels"
        )

        self.trigger_in_duration = Parameter(
            initial_value=.1e-6,
            set_cmd=None,
            vals=vals.Numbers(),
            docstring="Duration for a trigger input"
        )

        self.auto_advance = Parameter(
            initial_value=False,
            set_cmd=None,
            vals=vals.Bool(),
            docstring="Whether to only trigger once at the start or every pulse"
        )

        self.initial_instruction_idxs = Parameter(
            initial_value={'ch0': 0, 'ch1': 0, 'ch2': 0, 'ch3': 0},
            docstring="First instruction index for each channel to start, usually zero"
        )

        self.instruction_idx_offsets = Parameter(
            initial_value={'ch0': 0, 'ch1': 0, 'ch2': 0, 'ch3': 0},
            docstring="First instruction index for each channel during programming, usually zero"
        )
        self.pulse_idx_offsets = Parameter(
            initial_value={'ch0': 0, 'ch1': 0, 'ch2': 0, 'ch3': 0},
            docstring="First pulse index for each channel during programming, usually zero"
        )

    @property
    def decoupled_mode(self):
        from qcodes.instrument_drivers.CQC2T.PCDDS import PCDDS
        return not isinstance(self.instrument, PCDDS)

    @property
    def active_channel_ids(self):
        """Sorted list of active channel id's"""
        # First create a set to ensure unique elements
        active_channel_ids = {pulse.connection.output['channel'].id
                              for pulse in self.pulse_sequence}
        return sorted(active_channel_ids)

    @property
    def active_instrument_channels(self):
        return self.instrument.channels[self.active_channel_ids]

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Get list of pulses required by instrument (trigger pulses)

        If auto_advance == True, only one trigger pulse is requested at start
        of first pulse. Else a trigger pulse is returned for each pulse start and stop time.
        """
        assert self.use_trig_in(), "Interface not yet programmed for pxi triggering"
        if self.auto_advance():
            # Only trigger once at start of sequence
            t_list = [min(self.pulse_sequence.t_list)]
        else:
            # Get list of unique pulse start and stop times
            t_list = self.pulse_sequence.t_list

        trigger_pulses = [TriggerPulse(t_start=t,
                                       duration=self.trigger_in_duration(),
                                       connection_requirements={
                                           'input_instrument': self.instrument.name,
                                           'input_channel': self._input_channels['trig_in']
                                       })
                          for t in t_list if t != self.pulse_sequence.duration]
        return trigger_pulses

    def setup_decoupled(self):
        assert self.auto_advance(), "decoupled setup has only been programmed with auto_advance enabled"

        # Clear pulses and instruction up to a max idx
        # Max idx is added in case you want to keep high-index pulses/instructions
        for channel in self.instrument.channels:
            if self.pulse_max_idx is not None:
                pulses = [p for p in channel.pulses() if p['pulse_idx'] > self.pulse_max_idx]
            else:
                pulses = []
            channel.pulses().clear()
            channel.pulses().extend(pulses)

            if self.instruction_max_idx is not None:
                instructions = [
                    instr for instr in channel.instruction_sequence()
                    if instr['instruction_idx'] > self.instruction_max_idx
                ]
            else:
                instructions = []
            channel.instruction_sequence().clear()
            channel.instruction_sequence().extend(instructions)

        PCDDS_pulses = {channel.id: [] for channel in self.active_instrument_channels}
        PCDDS_instructions = {channel.id: [] for channel in self.active_instrument_channels}
        # Keep second pulse list without idxs for indexing
        _PCDDS_pulses_no_idxs = {channel.id: [] for channel in self.active_instrument_channels}

        def add_pulse_and_instruction(channel, pulse_implementation):
            instruction_offset = self.instruction_idx_offsets()[channel.name]
            pulse_offset = self.pulse_idx_offsets()[channel.name]
            try:
                pulse_idx = _PCDDS_pulses_no_idxs[channel.id].index(pulse_implementation)
                pulse_idx += pulse_offset
            except ValueError:
                _PCDDS_pulses_no_idxs[channel.id].append(pulse_implementation.copy())
                pulse_idx = len(PCDDS_pulses[channel.id])
                pulse_idx += pulse_offset
                pulse_implementation['pulse_idx'] = pulse_idx
                PCDDS_pulses[channel.id].append(pulse_implementation)

            # Add corresponding instruction
            instruction_idx = len(PCDDS_instructions[channel.id])
            PCDDS_instructions[channel.id].append({
                'instruction_idx': instruction_idx + instruction_offset,
                'pulse_idx': pulse_idx,
                'next_instruction': instruction_idx + instruction_offset + 1
            })

            return pulse_idx, instruction_idx

        # First pulses are 0V DC pulses
        # Only trigger on first pulse
        # t_start and duration must be set but are irrelevant
        DC_0V_pulse = self.get_pulse_implementation(
            DCPulse('initial_0V', t_start=0, duration=0, amplitude=0)
        )

        current_pulses = {channel.name: DC_0V_pulse
                          for channel in self.active_instrument_channels}

        for channel in self.active_instrument_channels:
            current_pulse = current_pulses[channel.name]
            pulse_implementation = current_pulse.implementation.implement()
            add_pulse_and_instruction(channel, pulse_implementation)

        # Use clock cycles for maximum accuracy
        clk = self.instrument.ch1.clk
        timing_offset = self.instrument.ch1.pulse_timing_offset

        t_min = min(self.pulse_sequence.t_list)
        cycles_min = int(round(t_min * clk))

        for channel in self.active_instrument_channels:
            cycles = cycles_min
            channel_pulses = self.pulse_sequence.get_pulses(output_channel=channel.name)
            for pulse in channel_pulses:
                start_cycles = int(round(pulse.t_start * clk))
                delta_cycles = start_cycles - cycles
                if delta_cycles > timing_offset:
                    # Add 0V pulse to bridge gap
                    pulse_implementation = DC_0V_pulse.implementation.implement()
                    pulse_implementation['duration'] = delta_cycles / clk
                    add_pulse_and_instruction(channel, pulse_implementation)

                    # Increment counters
                    cycles += delta_cycles
                elif delta_cycles > 1:
                    logger.warning(
                        f'Pulse {pulse} starts too early: {start_cycles} vs {cycles}. '
                        'Could mean the pulse has insufficient delay. '
                        'Next pulse will have a longer duration to bridge gap'
                    )
                elif delta_cycles < -1:
                    logger.warning(
                        f'Pulse {pulse} starts too early: {start_cycles} vs {cycles}. '
                        'This error should not occur unless pulses overlap. '
                        'Next pulse will have shorter duration to bridge gap'
                    )

                stop_cycles = int(round(pulse.t_stop * clk))
                delta_cycles = stop_cycles - cycles
                if delta_cycles <= timing_offset:
                    logger.warning(f'Pulse {pulse} is too short. '
                                   f'{stop_cycles} vs {cycles} cycles')
                    delta_cycles = 15

                # Implement pulse
                pulse_implementation = pulse.implementation.implement()
                pulse_implementation['duration'] = delta_cycles / clk
                add_pulse_and_instruction(channel, pulse_implementation)

                # Increment counters
                cycles += delta_cycles

            # Add a final DC pulse that requires triggering to restart
            pulse_implementation = DC_0V_pulse.implementation.implement()
            add_pulse_and_instruction(channel, pulse_implementation)

            instruction_idx_offset = self.instruction_idx_offsets()[channel.name]
            PCDDS_instructions[channel.id][-1]['next_instruction'] = instruction_idx_offset + 1
            # TODO: Add check that final DC pulse is not added if last pulse
            # ends at pulsesequence.duration, in which case the last pulse
            # should be triggered

            for pulse in PCDDS_pulses[channel.id]:
                channel.write_instr(pulse)

            for instruction in PCDDS_instructions[channel.id]:
                channel.write_instruction(**instruction)

            # Sort pulses and instructions on PCDDS channel
            pulses = sorted(channel.pulses(), key=(lambda pulse: pulse['pulse_idx']))
            channel.pulses().clear()
            channel.pulses().extend(pulses)
            instructions = sorted(
                channel.instruction_sequence(),
                key=(lambda instruction: instruction['instruction_idx'])
            )
            channel.instruction_sequence().clear()
            channel.instruction_sequence().extend(instructions)

    def setup_coupled(self):
        for channel in self.instrument.channels:
            channel.instruction_sequence().clear()

        # First pulses are 0V DC pulses
        # t_start and duration must be set but are irrelevant
        DC_0V_pulse = self.get_pulse_implementation(DCPulse('initial_0V',
                                                            t_start=0,
                                                            duration=0,
                                                            amplitude=0))
        current_pulses = {channel.name: DC_0V_pulse
                          for channel in self.active_instrument_channels}

        for channel in self.active_instrument_channels:
            current_pulse = current_pulses[channel.name]
            pulse_implementation = current_pulse.implementation.implement()
            pulse_implementation['pulse_idx'] = 0
            pulse_implementation['next_pulse'] = 1
            channel.write_instr(pulse_implementation)

        if self.auto_advance():
            # Only trigger on first pulse

            # Use clock cycles for maximum accuracy
            clk = self.instrument.ch1.clk
            timing_offset = self.instrument.ch1.pulse_timing_offset

            t_min = min(self.pulse_sequence.t_list)
            cycles_min = int(round(t_min * clk))

            for channel in self.active_instrument_channels:
                cycles = cycles_min
                channel_pulses = self.pulse_sequence.get_pulses(output_channel=channel.name)
                pulse_idx = 1
                for pulse in channel_pulses:
                    start_cycles = int(round(pulse.t_start * clk))
                    delta_cycles = start_cycles - cycles
                    if delta_cycles > timing_offset:
                        # Add 0V pulse to bridge gap
                        pulse_implementation = DC_0V_pulse.implementation.implement()
                        pulse_implementation['pulse_idx'] = pulse_idx
                        pulse_implementation['next_pulse'] = pulse_idx + 1
                        pulse_implementation['duration'] = delta_cycles / clk
                        channel.write_instr(pulse_implementation)

                        # Increment counters
                        pulse_idx += 1
                        cycles += delta_cycles
                    elif delta_cycles > 1:
                        logger.warning(
                            f'Pulse {pulse} starts too early: {start_cycles} vs {cycles}. '
                            'Could mean the pulse has insufficient delay. '
                            'Next pulse will have a longer duration to bridge gap'
                        )
                    elif delta_cycles < -1:
                        logger.warning(
                            f'Pulse {pulse} starts too early: {start_cycles} vs {cycles}. '
                            'This error should not occur unless pulses overlap. '
                            'Next pulse will have shorter duration to bridge gap'
                        )

                    stop_cycles = int(round(pulse.t_stop * clk))
                    delta_cycles = stop_cycles - cycles
                    if delta_cycles <= timing_offset:
                        logger.warning(f'Pulse {pulse} is too short. '
                                       f'{stop_cycles} vs {cycles} cycles')
                        delta_cycles = 15

                    # Implement pulse
                    pulse_implementation = pulse.implementation.implement()
                    pulse_implementation['pulse_idx'] = pulse_idx
                    pulse_implementation['next_pulse'] = pulse_idx + 1
                    pulse_implementation['duration'] = delta_cycles / clk
                    channel.write_instr(pulse_implementation)

                    # Increment counters
                    pulse_idx += 1
                    cycles += delta_cycles

                # Add a final DC pulse that requires triggering to restart
                pulse_implementation = DC_0V_pulse.implementation.implement()
                pulse_implementation['pulse_idx'] = pulse_idx
                pulse_implementation['next_pulse'] = 1
                channel.write_instr(pulse_implementation)
                # TODO: Add check that final DC pulse is not added if last pulse
                # ends at pulsesequence.duration, in which case the last pulse
                # should be triggered
        else:
            total_instructions = len(self.pulse_sequence.t_list)
            for pulse_idx, t in enumerate(self.pulse_sequence.t_list):
                if t == self.pulse_sequence.duration:
                    continue
                pulse_idx += 1  # We start with 1 since we have initial 0V pulse
                for channel in self.active_instrument_channels:
                    active_pulse = self.pulse_sequence.get_pulse(t_start=t,
                                                                 output_channel=channel.name)
                    if active_pulse is not None:  # New pulse starts
                        current_pulses[channel.name] = active_pulse
                    elif t >= current_pulses[channel.name].t_stop:
                        current_pulses[channel.name] = DC_0V_pulse

                    pulse_implementation = current_pulses[channel.name].implementation.implement()
                    pulse_implementation['pulse_idx'] = pulse_idx
                    if pulse_idx + 1 < total_instructions:
                        pulse_implementation['next_pulse'] = pulse_idx + 1
                    else:
                        # Loop back to second pulse (ignore first 0V pulse)
                        pulse_implementation['next_pulse'] = 1
                    channel.write_instr(pulse_implementation)

    def setup(self, **kwargs):
        self.instrument.channels.output_enable(False)
        self.instrument.channels.pcdds_enable(True)

        self.initial_instruction_idxs().update(ch0=0, ch1=0, ch2=0, ch3=0)

        assert self.use_trig_in(), "Interface not yet programmed for pxi triggering"

        # Perform different setup routine if FPGA image has pulses and
        # instructions separated
        if self.decoupled_mode:
            self.setup_decoupled()
        else:
            self.setup_coupled()

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def start(self):
        for channel in self.active_instrument_channels:
            channel.set_next_pulse(pulse=self.initial_instruction_idxs()[channel.name], update=True)
        self.active_instrument_channels.output_enable(True)

    def stop(self):
        self.instrument.channels.output_enable(False)


class DCPulseImplementation(PulseImplementation):
    pulse_class = DCPulse

    def implement(self, *args, **kwargs):
        return {
            'instr': 'dc',
            'amp': self.pulse.amplitude,
            'label': self.pulse.name
        }


class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse

    def implement(self, *args, **kwargs):
        return {
            'instr': 'dc',
            'amp': self.pulse.amplitude,
            'label': self.pulse.name
        }


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def implement(self, *args, **kwargs):
        # TODO distinguish between abolute / relative phase
        phase = self.pulse.phase

        return {
            'instr': 'sine',
            'freq': self.pulse.frequency,
            'amp': self.pulse.amplitude,
            'offset': self.pulse.offset,
            'phase': phase,
            'label': self.pulse.name
        }


class FrequencyRampPulseImplementation(PulseImplementation):
    pulse_class = FrequencyRampPulse

    def implement(self, *args, **kwargs):
        accumulation = (self.pulse.frequency_stop -
                        self.pulse.frequency_start) / self.pulse.duration
        return {
            'instr': 'chirp',
            'freq': self.pulse.frequency_start,
            'amp': self.pulse.amplitude,
            'offset': self.pulse.offset,
            'phase': getattr(self.pulse, 'phase', 0),
            'accum': accumulation,
            'label': self.pulse.name
        }
