import numpy as np
import logging
from typing import List, Union
from copy import copy

from silq import config
from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import (
    Pulse,
    DCPulse,
    TriggerPulse,
    SinePulse,
    FrequencyRampPulse,
    PulseImplementation,
    PulseSequence
)
from silq.tools.general_tools import find_approximate_divisor
from silq.tools.pulse_tools import pulse_to_waveform_sequence


from qcodes import ManualParameter, ParameterNode, MatPlot
from qcodes import validators as vals
from qcodes.utils.helpers import arreqclose_in_list
from qcodes.config.config import DotDict


logger = logging.getLogger(__name__)


class Keysight81180AInterface(InstrumentInterface):
    """

    Notes:
        - When the output is turned on, there is a certain ramping time of a few
          milliseconds. This negatively impacts the first repetition of a
          pulse sequence
        - To ensure voltage is fixed at final voltage of the pulse sequence
          during any pulse_sequence.final_delay, the first point of the first
          waveform of the sequence is set to that final voltage.
        - see ``interface.additional_settings`` for instrument settings that should
          be set manually
        - When the last waveform of the sequence is finished, the time until the
          next trigger is spent at the voltages of the first point of the first
          waveform in the sequence. This includes any PulseSequence.final_delay.
          However, the behaviour should be that during this final_delay, the
          voltage is kept at the last point of the last waveform. To ensure this,
          the first point of the first waveform is modified to that of the last
          point of the last waveform.
        - Creation of a sine waveform needs certain settings.
          Defaults are given in the SinePulseImplementation, but they can be
          overridden by setting the corresponding property in
          ``silq.config.properties.sine_waveform_settings``
    """

    def __init__(self, instrument_name, max_amplitude=1.5, **kwargs):
        assert max_amplitude <= 2

        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            f"ch{k}": Channel(
                instrument_name=self.instrument_name(), name=f"ch{k}", id=k, output=True
            )
            for k in [1, 2]
        }

        # TODO add marker outputs
        self._channels = {
            **self._output_channels,
            "trig_in": Channel(
                instrument_name=self.instrument_name(),
                name="trig_in",
                input_trigger=True,
            ),
            "event_in": Channel(
                instrument_name=self.instrument_name(),
                name="event_in",
                input_trigger=True,
            ),
            "sync": Channel(
                instrument_name=self.instrument_name(), name="sync", output=True
            ),
        }

        self.pulse_implementations = [
            DCPulseImplementation(
                pulse_requirements=[
                    ("amplitude", {"max": max_amplitude}),
                    ("duration", {"min": 100e-9}),
                ]
            ),
            SinePulseImplementation(
                pulse_requirements=[
                    ("frequency", {"min": -1.5e9, "max": 1.5e9}),
                    ("amplitude", {"min": 0, "max": max_amplitude}),
                    ("duration", {"min": 100e-9}),
                ]
            ),
            FrequencyRampPulseImplementation(
                pulse_requirements=[
                    ("frequency_start", {"min": -1.5e9, "max": 1.5e9}),
                    ("frequency_stop", {"min": -1.5e9, "max": 1.5e9}),
                    ("amplitude", {"min": 0, "max": max_amplitude}),
                    ("duration", {"min": 100e-9}),
                ]
            ),
        ]

        self.add_parameter(
            "trigger_in_duration", set_cmd=None, unit="s", initial_value=1e-6,
        )
        self.add_parameter(
            "active_channels",
            set_cmd=None,
            initial_value=[],
            vals=vals.Lists(vals.Strings()),
        )

        self.instrument.ch1.clear_waveforms()
        self.instrument.ch2.clear_waveforms()

        self.waveforms = {}  # List of waveform arrays for each channel
        # Optional initial waveform for each channel. Used to set the first point
        # to equal the last voltage of the final pulse (see docstring for details)
        self.waveforms_initial = {}
        self.sequences = {}  # List of sequence instructions for each channel
        # offsets list of actual programmed sample points versus expected points
        self.point_offsets = {}
        self.max_point_offsets = {}  # Maximum absolute sample point offset
        self.point = {}  # Current sample point, incremented as sequence is programmed
        # Maximum tolerable absolute point offset before raising a warning
        self.point_offset_limit = 100

        # Add parameters that are not set via setup
        self.additional_settings = ParameterNode()
        for instrument_channel in self.instrument.channels:
            channel = ParameterNode(instrument_channel.name)
            setattr(self.additional_settings, instrument_channel.name, channel)

            channel.output_coupling = instrument_channel.output_coupling
            channel.sample_rate = instrument_channel.sample_rate

    def get_additional_pulses(self, **kwargs):
        # Currently the only supported sequence_mode is `once`, i.e. one trigger
        # at the start of the pulse sequence

        # Request a single trigger at the start of the pulse sequence
        logger.info(f"Creating trigger for Keysight 81180A: {self.name}")
        return [
            TriggerPulse(
                name=self.name + "_trigger",
                t_start=0,
                duration=self.trigger_in_duration(),
                connection_requirements={
                    "input_instrument": self.instrument_name(),
                    "trigger": True,
                },
            )
        ]

    def setup(self, **kwargs):
        self.active_channels(
            list(
                {
                    pulse.connection.output["channel"].name
                    for pulse in self.pulse_sequence
                }
            )
        )
        for ch in self.active_channels():
            instrument_channel = getattr(self.instrument, ch)
            instrument_channel.off()

            instrument_channel.continuous_run_mode(False)
            instrument_channel.function_mode("sequenced")

            # TODO Are these needed? Or set via sequence?
            # instrument_channel.power(5)  # If coupling is AC
            # instrument_channel.voltage_DAC(voltage)  # If coupling is DAC (max 0.5)
            instrument_channel.voltage_DC(2)  # If coupling is DC (max 2)
            if not instrument_channel.output_coupling() == "DC":
                logger.warning(
                    "Keysight 81180 output coupling is not DC. The waveform "
                    "amplitudes might be off."
                )

            instrument_channel.voltage_offset(0)
            instrument_channel.output_modulation("off")

            # Trigger settings
            instrument_channel.trigger_input("TTL")
            instrument_channel.trigger_source("external")
            instrument_channel.trigger_slope("positive")
            # Goto next waveform immediately if trigger received
            instrument_channel.trigger_mode("override")
            instrument_channel.trigger_level(1)  # TODO Make into parameter
            instrument_channel.trigger_delay(0)
            # Only advance a single waveform/sequence per trigger.
            instrument_channel.burst_count(1)

            # Immediately skip to next waveform. Not sure what the difference is
            # with trigger_mode('override')
            instrument_channel.waveform_timing("immediate")

            # Should be either 'once' (finish entire sequence per trigger)
            # or 'stepped' (one trigger per waveform)
            instrument_channel.sequence_mode("once")
            # Do not repeat sequence multiple times
            instrument_channel.sequence_once_count(1)

        if self.instrument.is_idle() != "1":
            logger.warning("Not idle")

        self.instrument.ensure_idle = True
        self.generate_waveform_sequences()
        self.instrument.ensure_idle = False

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def _get_single_waveform_pulse_sequences(
        self, pulse_sequence=None, always_true=False
    ):
        """Select all pulse sequences with flag 'single_waveform'

        pulse sequences with this flag should program its pulses as a single
        waveform instead of individual waveforms per pulse. This ensures exact
        precision of the pulse sequence.

        A pulse sequence has the flag 'single_waveform' when:
        >>> 'single_waveform' in pulse_sequence.flags['<AWG_instrument_name>']
        where <AWG_instrument_name> is the 81180A instrument name

        A nested pulse sequence has the flag if it contains this flag, or if
        one of its parents has a flag. Consequently, this method is recursive

        Args:
            pulse_sequence: Pulse sequence for which to determine if it has
                the flag 'single_waveform'. It is a nested waveform when called
                recursively. If not specified, use interface.pulse_sequence.
            always_true: Set to True if its parent has the flag 'single_waveform'.
                In this case, the nested pulse sequence should also have the flag.

        Returns:
            List of pulse sequences that have flag 'single_waveform' or whose
            parent waveform has flag 'single_waveform'
        """
        # Use interface.pulse_sequence if not explicitly passed
        if pulse_sequence is None:
            pulse_sequence = self.pulse_sequence

        single_waveform_pulse_sequences = []

        # Check if (nested) pulse sequence contains flag 'single_waveform
        flags = pulse_sequence.flags.get(self.instrument.name, {})
        if "single_waveform" in flags or always_true:
            single_waveform_pulse_sequences.append(pulse_sequence)
            always_true = True

        # Recursively perform check for each of its nested pulse sequences
        for pulse_subsequence in pulse_sequence.pulse_sequences:
            single_waveform_pulse_sequences += self._get_single_waveform_pulse_sequences(
                pulse_subsequence, always_true=always_true
            )

        return single_waveform_pulse_sequences

    def get_channel_pulses_and_sequences(self, channel) -> List[Union[Pulse, PulseSequence]]:
        """Get all pulses and single_waveform pulse sequences for a channel

        Pulse sequences that have the flag 'single_waveform' should have all
        its pulses combined into a single waveform.

        Args:
            channel: Channel for which to return all pulses and single_waveform
                pulse sequences

        Returns:
            List of pulses and pulse sequences.
            Pulses are replaced by their pulse sequences if the pulse sequence
            has the flag 'single_waveform'.
        """

        # Get all pulses that should be output by channel
        pulses = self.pulse_sequence.get_pulses(output_channel=channel)

        # Replace all pulses that belong to a single pulse sequence with flag
        # 'single_waveform' with the corresponding pulse sequence.
        single_waveform_pulse_sequences = self._get_single_waveform_pulse_sequences()
        pulses_and_sequences = []
        for pulse in pulses:
            if pulse.parent in single_waveform_pulse_sequences:
                # Get top-most pulse sequence that has the flag 'single_waveform'
                pulse_sequence = pulse.parent
                flags = pulse_sequence.flags.get(self.instrument.name)
                while 'single_waveform' not in flags:
                    pulse_sequence = pulse_sequence.parent
                    flags = pulse_sequence.flags.get(self.instrument.name)

                if pulse_sequence not in pulses_and_sequences:
                    pulses_and_sequences.append(pulse_sequence)
            else:
                pulses_and_sequences.append(pulse)

        return pulses_and_sequences

    def generate_waveform_sequences(self):
        """Generate waveforms and sequences for AWG.

        Note that each channel has a memory limit (usually 16M points per channel)
        """
        self.waveforms = {ch: [] for ch in self.active_channels()}
        self.sequences = {ch: [] for ch in self.active_channels()}
        self.point = {ch: 0 for ch in self.active_channels()}
        self.point_offsets = {ch: [] for ch in self.active_channels()}

        for ch in self.active_channels():
            sample_rate = self.instrument.channels[ch].sample_rate()

            self.waveforms_initial[ch] = None

            # Always begin by waiting for a trigger/event pulse
            # Add empty waveform (0V DC), with minimum points (320)
            self.add_single_waveform(ch, waveform_array=np.zeros(320))

            elements = self.get_channel_pulses_and_sequences(channel=ch)

            t = 0  # Set start time t=0
            for element in elements:
                if isinstance(element, Pulse):
                    self.add_pulse(element, t=t, sample_rate=sample_rate, channel=ch)
                    t = element.t_stop
                else:
                    # Element is a pulse sequence whose pulses we must
                    # concatenate into a single waveform
                    self.add_pulse_sequence(
                        element, t=t, sample_rate=sample_rate, channel=ch
                    )
                    t = element.t_stop

            self.finalize_generate_waveforms_sequences(channel=ch, t=t)

    def _add_DC_waveform(
        self,
        channel_name: str,
        t_start: float,
        t_stop: float,
        amplitude: float,
        sample_rate: float,
        pulse_name="DC",
    ) -> List:
        # We fake a DC pulse for improved performance
        DC_pulse = DotDict(
            dict(
                t_start=t_start,
                t_stop=t_stop,
                duration=round(t_stop - t_start, 11),
                amplitude=amplitude,
            )
        )
        waveform = DCPulseImplementation.implement(
            pulse=DC_pulse, sample_rate=sample_rate
        )
        sequence_steps = self.add_pulse_waveforms(
            channel_name,
            **waveform,
            t_start=DC_pulse.t_start,
            t_stop=DC_pulse.t_stop,
            sample_rate=sample_rate,
            pulse_name=pulse_name,
        )

        return sequence_steps

    def add_single_waveform(
        self, channel_name: str, waveform_array: np.ndarray, allow_existing: bool = True
    ) -> int:
        """Add waveform to instrument, uploading if necessary

        If the waveform already exists on the instrument and allow_existing=True,
        the existing waveform is used and no new waveform is uploaded.

        Args:
            channel_name: Name of channel for which to upload waveform
            waveform_array: Waveform array
            allow_existing:

        Returns:
            Waveform index, used for sequencing

        Raises:
            SyntaxError if waveform contains less than 320 points
        """
        if len(waveform_array) < 320:
            raise SyntaxError(f"Waveform length {len(waveform_array)} < 320")

        self.waveforms.setdefault(channel_name, [])

        # Check if waveform already exists in waveform array
        if allow_existing:
            waveform_idx = arreqclose_in_list(
                waveform_array, self.waveforms[channel_name], atol=1e-3
            )
        else:
            waveform_idx = None

        # Check if new waveform needs to be created and uploaded
        if waveform_idx is not None:
            waveform_idx += 1  # Waveform index is 1-based
        else:
            # Add waveform to current list of waveforms
            self.waveforms[channel_name].append(waveform_array)

            # waveform index should be the position of added waveform (1-based)
            waveform_idx = len(self.waveforms[channel_name])

        return waveform_idx

    def add_pulse_waveforms(
        self,
        channel_name: str,
        waveform: np.ndarray,
        loops: int,
        waveform_initial: Union[np.ndarray, None],
        waveform_tail: Union[np.ndarray, None],
        t_start: float,
        t_stop: float,
        sample_rate: float,
        pulse_name=None,
    ) -> List[tuple]:
        sequence = []
        total_points = 0

        if pulse_name is None:
            pulse_name = "pulse"

        if waveform_initial is not None:
            # An initial waveform must be added. This initial waveform corresponds
            # to the beginning of a DC waveform at the start of a pulse sequence.
            # See interface docstring for details

            # Temporarily set waveform to None, will be set later to correct waveform
            self.waveforms[channel_name].append(None)
            waveform_initial_idx = len(self.waveforms[channel_name])
            # Temporarily store initial waveform in separate variable
            self.waveforms_initial[channel_name] = (
                waveform_initial_idx,
                waveform_initial,
            )
            # Add sequence step (waveform_idx, loops, jump_event)
            sequence.append((waveform_initial_idx, 1, 0, f"{pulse_name}_pre"))

            total_points += len(waveform_initial)  # Update total waveform points

        # Upload main waveform
        waveform_idx = self.add_single_waveform(channel_name, waveform)

        total_points += len(waveform) * loops  # Update total waveform points

        # Add sequence step (waveform_idx, loops, jump_event, label)
        sequence.append((waveform_idx, loops, 0, pulse_name))

        # Optionally add waveform tail
        if waveform_tail is not None:
            waveform_tail_idx = self.add_single_waveform(channel_name, waveform_tail)
            sequence.append((waveform_tail_idx, 1, 0, f"{pulse_name}_tail"))

            total_points += len(waveform_tail)  # Update total waveform points

        # Update the total number of sample points after having implemented
        # this pulse waveform.
        self.point[channel_name] += total_points

        # Compare the total number of sample points to the expected number of
        # sample points (t_stop * sample_rate). this may differ because waveforms
        # must have a multiple of 32 points
        expected_stop_point = int(t_stop * sample_rate)
        self.point_offsets[channel_name].append(
            self.point[channel_name] - expected_stop_point
        )

        return sequence

    def add_pulse(self, pulse, t, channel, sample_rate):
        # A waveform must have at least 320 points
        min_waveform_duration = 320 / sample_rate

        # Check if there is a gap between next pulse and current time t_pulse
        if pulse.t_start + 1e-11 < t:
            raise SyntaxError(
                f"Trying to add pulse {pulse} which starts before current "
                f"time position in waveform {t}"
            )
        elif 1e-11 < pulse.t_start - t < min_waveform_duration + 1e-11:
            # The gap between pulses is smaller than the minimum waveform
            # duration. Cannot create DC waveform to bridge the gap
            raise SyntaxError(
                f"Delay between pulse {pulse} start {pulse.t_start} s "
                f"and current time {t} s is less than minimum "
                f"waveform duration. cannot add 0V DC pulse to bridge gap"
            )
        elif pulse.t_start - t >= min_waveform_duration + 1e-11:
            # Add 0V DC pulse to bridge the gap between pulses
            self.sequences[channel] += self._add_DC_waveform(
                channel_name=channel,
                t_start=t,
                t_stop=pulse.t_start,
                amplitude=0,
                sample_rate=sample_rate,
                pulse_name="DC",
            )

        # Get waveform of current pulse
        waveform = pulse.implementation.implement(sample_rate=sample_rate,)

        # Add waveform and sequence steps
        sequence_steps = self.add_pulse_waveforms(
            channel,
            **waveform,
            t_start=pulse.t_start,
            t_stop=pulse.t_stop,
            sample_rate=sample_rate,
            pulse_name=pulse.name,
        )
        self.sequences[channel] += sequence_steps

    def add_pulse_sequence(self, pulse_sequence, t, channel, sample_rate):
        # A waveform must have at least 320 points
        min_waveform_duration = 320 / sample_rate

        pulse_sequence_name = pulse_sequence.name or 'unnamed_pulse_sequence'
        label = f"{pulse_sequence_name}_single_waveform"

        pulses = pulse_sequence.get_pulses(output_channel=channel)
        t_start = min(pulse.t_start for pulse in pulses)
        t_stop = max(pulse.t_stop for pulse in pulses)

        # Add DC waveform if first pulse start after start of sequence
        if 1e-11 < t_start - t < min_waveform_duration + 1e-11:
            # The gap between pulses is smaller than the minimum waveform
            # duration. Cannot create DC waveform to bridge the gap
            raise SyntaxError(
                f"Delay between single-waveform pulse sequence t_start {t_start}"
                f"and current time {t} s is less than minimum "
                f"waveform duration. cannot add 0V DC pulse to bridge gap"
            )
        elif t_start - t >= min_waveform_duration + 1e-11:
            # Add 0V DC pulse to bridge the gap until first pulse
            self.sequences[channel] += self._add_DC_waveform(
                channel_name=channel,
                t_start=t,
                t_stop=t_start,
                amplitude=0,
                sample_rate=sample_rate,
                pulse_name=f"DC_{label}_initial",
            )

        # Ensure t starts at exactly the right time
        t = self.point[channel] / sample_rate

        # Determine waveform points
        # Round up because otherwise the final pulse may have too many points
        points = int(np.ceil((t_stop - t) * sample_rate))

        # Add remaining points of the pulse sequence if it is sufficiently short
        remaining_points = int((pulse_sequence.t_stop - t_stop) * sample_rate)
        if remaining_points < 640:
            points += remaining_points

        # Round points up to multiple of 32
        if points % 32:
            points = 32 * (1 + points // 32)

        # Ensure points is at least 320 points
        points = max(points, 320)

        if points > 12e6:
            raise MemoryError(
                f"Cannot implement pulse sequence as single waveform, "
                f"too many points required: {points}"
            )

        waveform_array = np.zeros(points)
        for pulse in pulses:
            # Determine start_idx of pulse
            # Round up to ensure the corresponding t_start is not before the pulse
            start_idx = int(np.ceil((pulse.t_start - t) * sample_rate))
            # If for whatever reason the pulse starts before t, ensure the index
            # is at least zero
            start_idx = max(start_idx, 0)

            # t_start is first waveform point starting at/after pulse.t_start
            t_start = t + start_idx / sample_rate
            pulse_points = int(np.floor((pulse.t_stop - t_start) * sample_rate))
            t_list = t_start + np.arange(pulse_points) / sample_rate
            voltages = pulse.get_voltage(t_list)

            waveform_array[start_idx:start_idx+len(voltages)] = voltages

        # Add waveform, skip uploading if it already exists
        self.sequences[channel] += self.add_pulse_waveforms(
            channel_name=channel,
            waveform=waveform_array,
            loops=1,
            waveform_initial=None,
            waveform_tail=None,
            t_start=t,
            t_stop=t + points / sample_rate,
            sample_rate=sample_rate,
            pulse_name=label
        )
        t += points / sample_rate

        # Add final
        remaining_points = int((pulse_sequence.t_stop - t) * sample_rate)
        if remaining_points >= 320:
            # Add 0V DC pulse to bridge the gap until end of pulse sequence
            self.sequences[channel] += self._add_DC_waveform(
                channel_name=channel,
                t_start=t,
                t_stop=pulse_sequence.t_stop,
                amplitude=0,
                sample_rate=sample_rate,
                pulse_name=f"DC_{label}_final",
            )

    def finalize_generate_waveforms_sequences(self, channel, t):
        instrument_channel = self.instrument.channels[channel]
        sample_rate = instrument_channel.sample_rate()

        # A waveform must have at least 320 points
        min_waveform_duration = 320 / sample_rate

        # All channel pulses have been added
        # Add 0V pulse if last pulse does not stop at pulse_sequence.duration
        if self.pulse_sequence.duration - t >= min_waveform_duration + 1e-11:
            self.sequences[channel] += self._add_DC_waveform(
                channel_name=channel,
                t_start=t,
                t_stop=self.pulse_sequence.duration,
                amplitude=0,
                sample_rate=sample_rate,
                pulse_name="final_DC",
            )

        # Change voltage of first point of first waveform to that of last
        # point of last waveform
        last_waveform_idx = self.sequences[channel][-1][0]
        last_waveform = self.waveforms[channel][last_waveform_idx - 1]
        last_voltage = last_waveform[-1]
        if self.waveforms_initial[channel] is not None:
            waveform_initial_idx, waveform_initial = self.waveforms_initial[channel]

            logger.debug(f"Changing first point of first waveform to {last_voltage}")

            waveform_initial[0] = last_voltage
            self.waveforms[channel][waveform_initial_idx - 1] = waveform_initial

        # Ensure there are at least three sequence instructions
        while len(self.sequences[channel]) < 3:
            waveform_idx = self.add_single_waveform(
                channel, last_voltage * np.ones(320)
            )
            # Add extra blank segment which will automatically run to
            # the next segment (~ 70 ns offset)
            self.sequences[channel].append((waveform_idx, 1, 0, "final_filler_pulse"))

        # Ensure total waveform points are less than memory limit
        total_waveform_points = sum(
            len(waveform) for waveform in self.waveforms[channel]
        )
        if total_waveform_points > self.instrument.waveform_max_length:
            raise RuntimeError(
                f"Total waveform points {total_waveform_points} exceeds "
                f"limit of 81180A ({self.instrument.waveform_max_length})"
            )

        # Sequence all loaded waveforms
        # breakpoint()
        waveform_idx_mapping = instrument_channel.upload_waveforms(
            self.waveforms[channel], allow_existing=True
        )
        # breakpoint()
        # Update waveform indices since they may correspond to pre-existing waveforms
        self.sequences[channel] = [
            (waveform_idx_mapping[idx], *instructions)
            for idx, *instructions in self.sequences[channel]
        ]
        instrument_channel.set_sequence(self.sequences[channel])

        # Check that the sample point offsets do not exceed limit
        self.max_point_offsets[channel] = max(np.abs(self.point_offsets[channel]))
        if self.max_point_offsets[channel] > self.point_offset_limit:
            logger.warning(
                f"81180A maximum sample point offset exceeds limit {self.point_offset_limit}. "
                f"Current maximum: {self.max_point_offsets}"
            )
        else:
            logger.debug(
                f"81180A sample point maximum offset: {self.max_point_offsets}"
            )

    def start(self):
        """Turn all active instrument channels on"""
        for ch_name in self.active_channels():
            instrument_channel = getattr(self.instrument, ch_name)
            instrument_channel.on()

    def stop(self):
        """Turn both instrument channels off"""
        self.instrument.ch1.off()
        self.instrument.ch2.off()

    def verify_waveforms(self, channel, plot=True):
        AWG_channel = self.instrument.channels[channel]
        sample_rate = AWG_channel.sample_rate()

        waveforms = AWG_channel.uploaded_waveforms()
        # Each sequence step has form
        # (waveform_idx, repetitions, jump flag (always 0), label)
        sequence = AWG_channel.uploaded_sequence()

        # Concatenate all waveforms
        waveforms_list = []
        for step in sequence:
            waveform_idx, repetitions, _, _ = step
            waveform_idx -= 1
            waveforms_list += [waveforms[waveform_idx]] * repetitions

        waveform_pts = sum(len(waveform) for waveform in waveforms_list)
        assert waveform_pts < 200e6

        # Create single long array
        single_waveform = np.hstack(waveforms_list)
        print(f"Waveform points: {waveform_pts/1e6:.2f}M")

        # Generate target single waveform from pulse sequence
        channel_pulses = self.pulse_sequence.get_pulses(output_channel=channel)
        target_single_waveform = np.zeros(len(single_waveform))
        for pulse in channel_pulses:
            start_idx = int(np.ceil(pulse.t_start * sample_rate))
            stop_idx = int(np.floor(pulse.t_stop * sample_rate))
            t_list = np.arange(start_idx, stop_idx) / sample_rate
            voltages = pulse.get_voltage(t_list)
            target_single_waveform[start_idx:stop_idx] = voltages

        waveform_difference = single_waveform - target_single_waveform

        print(f'Maximum voltage difference: {np.max(np.abs(waveform_difference)):.2f} V')

        results = {
            "uploaded_single_waveform": single_waveform,
            "target_single_waveform": target_single_waveform,
            "waveform_difference": waveform_difference,
            "pulses": channel_pulses,
            "channel": channel
        }

        if plot:
            step = int(len(waveform_difference) / 200e3) # Plot around 200k points
            self.plot_waveform_verification(**results, step=step)

        return results

    def plot_waveform_verification(
        self,
        uploaded_single_waveform,
        target_single_waveform,
        waveform_difference,
        channel,
        step=1,
        t_start=None,
        t_stop=None,
        **kwargs,
    ):
        sample_rate = self.instrument.channels[channel].sample_rate()

        start_idx = int(t_start * sample_rate) if t_start is not None else None
        stop_idx = int(t_stop * sample_rate) if t_stop is not None else None
        idxs = slice(start_idx, stop_idx, step)

        arrays = [
            uploaded_single_waveform[idxs],
            target_single_waveform[idxs],
            waveform_difference[idxs],
        ]

        points = len(arrays[0])
        assert points < 2e6, (
            f"Points {points:.0f} exceeds max 2M, "
            f"please increase step to at least {2e6 // points+1}"
        )
        if start_idx is None:
            start_idx = 0

        t_list = (start_idx + np.arange(points) * step) / sample_rate
        plot = MatPlot(arrays, x=t_list * 1e3)
        plot[0].legend(["Uploaded", "Target", "verification"])
        plot[0].set_xlabel("Time (ms)")
        plot[0].set_ylabel("Amplitude (V)")


class DCPulseImplementation(PulseImplementation):
    pulse_class = DCPulse

    @staticmethod
    def implement(pulse: DCPulse, sample_rate: float, max_points: int = 6000) -> dict:
        # TODO shouldn't the properties of self be from self.pulse instead?

        # Number of points must be a multiple of 32
        N = 32 * np.floor(pulse.duration * sample_rate / 32)

        # Add a waveform_initial if this pulse is the start of the pulse_sequence
        # and has a long enough duration. The first point of waveform_initial will
        # later on be set to the last point of the last waveform, ensuring that
        # any pulse_sequence.final_delay remains at the last voltage
        if pulse.t_start == 0 and N > 640:
            N -= 320
            waveform_initial = pulse.amplitude * np.ones(320)
            logger.debug("adding waveform_initial")
        else:
            waveform_initial = None

        if N < 320:
            raise RuntimeError(
                f"Cannot add pulse because the number of waveform points {N} "
                f"is less than the minimum 320. {pulse}"
            )

        # Find an approximate divisor of the number of points N, allowing us
        # to shrink the waveform
        approximate_divisor = find_approximate_divisor(
            N=N,
            max_cycles=1000000,
            points_multiple=32,  # waveforms must be multiple of 32
            min_points=320,  # waveform must have at least 320 points
            max_points=max_points,
            max_remaining_points=1000,
            min_remaining_points=320,
        )

        if approximate_divisor is None:
            raise RuntimeError(
                f"Could not add DC waveform because no divisor "
                f"could be found for the number of points {N}"
            )

        # Add waveform(s) and sequence steps
        waveform = pulse.amplitude * np.ones(approximate_divisor["points"])

        # Add separate waveform if there are remaining points left after division
        if approximate_divisor["remaining_points"]:
            waveform_tail = pulse.amplitude * np.ones(
                approximate_divisor["remaining_points"]
            )
        else:
            waveform_tail = None

        return {
            "waveform": waveform,
            "loops": approximate_divisor["cycles"],
            "waveform_initial": waveform_initial,
            "waveform_tail": waveform_tail,
        }


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    settings = {
        "max_points": 50e3,
        "frequency_threshold": 30,
    }

    def implement(self, sample_rate, plot=False, **kwargs):
        # If frequency is zero, use DC pulses instead
        if self.pulse.frequency == 0:
            DC_pulse = DCPulse(
                "DC_sine",
                t_start=self.pulse.t_start,
                t_stop=self.pulse.t_stop,
                amplitude=self.pulse.get_voltage(self.pulse.t_start),
            )
            return DCPulseImplementation.implement(
                pulse=DC_pulse, sample_rate=sample_rate
            )
        settings = copy(self.settings)
        settings.update(**config.properties.get("sine_waveform_settings", {}))
        settings.update(
            **config.properties.get("keysight_81180A_sine_waveform_settings", {})
        )
        settings.update(**kwargs)

        # Do not approximate frequency if the pulse is sufficiently short
        max_points_exact = settings.pop("max_points_exact", 4000)
        points = int(self.pulse.duration * sample_rate)
        if points > max_points_exact:
            self.results = pulse_to_waveform_sequence(
                frequency=self.pulse.frequency,
                sampling_rate=sample_rate,
                total_duration=self.pulse.duration,
                min_points=320,
                sample_points_multiple=32,
                plot=plot,
                **settings,
            )
        else:
            self.results = None

        if self.results is None:
            t_list = np.arange(self.pulse.t_start, self.pulse.t_stop, 1 / sample_rate)
            t_list = t_list[: len(t_list) // 32 * 32]  # Ensure pulse is multiple of 32
            if len(t_list) < 320:
                raise RuntimeError(
                    f"Sine waveform points {len(t_list)} is below "
                    f"minimum of 320. Increase pulse duration or "
                    f"sample rate. {self.pulse}"
                )

            return {
                "waveform": self.pulse.get_voltage(t_list),
                "loops": 1,
                "waveform_initial": None,
                "waveform_tail": None,
            }

        optimum = self.results["optimum"]
        waveform_loops = max(optimum["repetitions"], 1)

        # Temporarily modify pulse frequency to ensure waveforms have full period
        original_frequency = self.pulse.frequency
        self.pulse.frequency = optimum["modified_frequency"]

        waveform_pts = optimum["points"]
        # Potentially include a waveform tail
        waveform_tail_pts = int(round(optimum["final_delay"] * sample_rate))
        # Waveform must be multiple of 32, if number of points is less than
        # this, there is no point in adding the waveform
        if waveform_tail_pts < 32:
            waveform_tail_array = None
        elif waveform_loops == 1:
            # Add waveform tail to main waveform
            waveform_pts += waveform_tail_pts
            waveform_pts = 32 * (waveform_pts // 32)
            waveform_tail_pts = 0
            waveform_tail_array = None
        else:
            if waveform_tail_pts < 320:  # Waveform must be at least 320 points
                # Find minimum number of loops of main waveform that are needed
                # to increase tail to be at least 320 points long
                subtract_loops = int(
                    np.ceil((320 - waveform_tail_pts) / optimum["points"])
                )
            else:
                subtract_loops = 0

            if waveform_loops - subtract_loops > 0:
                # Safe to subtract loops from the main waveform, add to this one
                waveform_loops -= subtract_loops
                waveform_tail_pts += subtract_loops * optimum["points"]

                t_list_tail = np.arange(
                    self.pulse.t_start + optimum["duration"] * waveform_loops,
                    self.pulse.t_stop,
                    1 / sample_rate,
                )
                t_list_tail = t_list_tail[: 32 * (len(t_list_tail) // 32)]
                waveform_tail_array = self.pulse.get_voltage(t_list_tail)
            else:
                # Cannot subtract loops from the main waveform because then
                # the main waveform would not have any loops remaining
                waveform_tail_array = None

        # Get waveform points for repeated segment
        t_list = self.pulse.t_start + np.arange(waveform_pts) / sample_rate
        waveform_array = self.pulse.get_voltage(t_list)

        # Reset pulse frequency to original
        self.pulse.frequency = original_frequency

        if plot:
            plot = MatPlot(subplots=(2, 1), figsize=(10, 6), sharex=True)

            ax = plot[0]
            t_list = np.arange(self.pulse.t_start, self.pulse.t_stop, 1 / sample_rate)
            voltages = self.pulse.get_voltage(t_list)
            ax.add(t_list, voltages, color="C0")

            # Add recreated sine pulse
            wf_voltages_main = np.tile(waveform_array, waveform_loops)
            wf_voltages_tail = waveform_tail_array
            wf_voltages = np.hstack((wf_voltages_main, wf_voltages_tail))
            t_stop = self.pulse.t_start + len(wf_voltages) / sample_rate
            wf_t_list = self.pulse.t_start + np.arange(len(wf_voltages)) / sample_rate
            ax.add(wf_t_list, wf_voltages, marker="o", ms=2, color="C1")

            # Add remaining marker values
            new_wf_idxs = np.arange(waveform_loops) * len(waveform_array)
            ax.plot(
                wf_t_list[new_wf_idxs], wf_voltages[new_wf_idxs], "o", color="C2", ms=4
            )
            wf_tail_idx = len(waveform_array) * waveform_loops
            ax.plot(
                wf_t_list[wf_tail_idx], wf_voltages[wf_tail_idx], "o", color="C3", ms=4
            )
            ax.set_ylabel("Amplitude (V)")

            ax = plot[1]
            ax.add(wf_t_list, wf_voltages - voltages)
            ax.set_ylabel("Amplitude error (V)")
            ax.set_xlabel("Time (s)")

            plot.tight_layout()

        return {
            "waveform": waveform_array,
            "loops": waveform_loops,
            "waveform_initial": None,
            "waveform_tail": waveform_tail_array,
        }


class FrequencyRampPulseImplementation(PulseImplementation):
    pulse_class = FrequencyRampPulse

    def implement(self, sample_rate, plot=False, **kwargs):
        if self.pulse.frequency_deviation == 0:
            raise RuntimeError(f"{self.pulse} has no frequency_deviation")

        # Convert t_start and t_stop to int to get rid of floating point errors
        start_idx = int(round(self.pulse.t_start * sample_rate))
        stop_idx = int(round(self.pulse.t_stop * sample_rate))
        # Add half of 32 points to ensure good rounding during floor division
        t_list = np.arange(start_idx, stop_idx + 16)
        # Ensure number of points is multiple of 32
        t_list = t_list[: len(t_list) // 32 * 32]
        t_list = t_list / sample_rate
        if len(t_list) < 320:
            raise RuntimeError("Waveform has fewer than minimum 320 points")

        waveform_array = self.pulse.get_voltage(t_list)

        return {
            "waveform": waveform_array,
            "loops": 1,
            "waveform_initial": None,
            "waveform_tail": None,
        }
