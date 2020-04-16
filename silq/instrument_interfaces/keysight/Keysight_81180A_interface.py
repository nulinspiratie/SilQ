import numpy as np
import logging
from typing import List, Union
from copy import copy

from silq import config
from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import (
    DCPulse,
    TriggerPulse,
    SinePulse,
    FrequencyRampPulse,
    PulseImplementation,
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
        assert max_amplitude <= 1.5

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
            "trigger_in_duration",
            parameter_class=ManualParameter,
            unit="s",
            initial_value=1e-6,
        )
        self.add_parameter(
            "active_channels",
            parameter_class=ManualParameter,
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

    def generate_waveform_sequences(self):
        self.waveforms = {ch: [] for ch in self.active_channels()}
        self.sequences = {ch: [] for ch in self.active_channels()}
        self.point = {ch: 0 for ch in self.active_channels()}
        self.point_offsets = {ch: [] for ch in self.active_channels()}

        for ch in self.active_channels():
            instrument_channel = self.instrument.channels[ch]
            sample_rate = instrument_channel.sample_rate()
            # Set start time t=0
            t_pulse = 0
            self.waveforms_initial[ch] = None

            # A waveform must have at least 320 points
            min_waveform_duration = 320 / sample_rate

            # Always begin by waiting for a trigger/event pulse
            # Add empty waveform (0V DC), with minimum points (320)
            self.add_single_waveform(ch, waveform_array=np.zeros(320))

            pulses = self.pulse_sequence.get_pulses(output_channel=ch)

            for pulse in pulses:
                # Check if there is a gap between next pulse and current time t_pulse
                if pulse.t_start + 1e-11 < t_pulse:
                    raise SyntaxError(
                        f"Trying to add pulse {pulse} which starts before current "
                        f"time position in waveform {t_pulse}"
                    )
                elif 1e-11 < pulse.t_start - t_pulse < min_waveform_duration + 1e-11:
                    # The gap between pulses is smaller than the minimum waveform
                    # duration. Cannot create DC waveform to bridge the gap
                    raise SyntaxError(
                        f"Delay between pulse {pulse} start {pulse.t_start} s "
                        f"and current time {t_pulse} s is less than minimum "
                        f"waveform duration. cannot add 0V DC pulse to bridge gap"
                    )
                elif pulse.t_start - t_pulse >= min_waveform_duration + 1e-11:
                    # Add 0V DC pulse to bridge the gap between pulses
                    self.sequences[ch] += self._add_DC_waveform(
                        channel_name=ch,
                        t_start=t_pulse,
                        t_stop=pulse.t_start,
                        amplitude=0,
                        sample_rate=sample_rate,
                        pulse_name="DC",
                    )

                # Get waveform of current pulse
                waveform = pulse.implementation.implement(sample_rate=sample_rate,)

                # Add waveform and sequence steps
                sequence_steps = self.add_pulse_waveforms(
                    ch,
                    **waveform,
                    t_stop=pulse.t_stop,
                    sample_rate=sample_rate,
                    pulse_name=pulse.name,
                )
                self.sequences[ch] += sequence_steps

                # Set current time to pulse.t_stop
                t_pulse = pulse.t_stop

            # Add 0V pulse if last pulse does not stop at pulse_sequence.duration
            if self.pulse_sequence.duration - t_pulse >= min_waveform_duration + 1e-11:
                self.sequences[ch] += self._add_DC_waveform(
                    channel_name=ch,
                    t_start=t_pulse,
                    t_stop=self.pulse_sequence.duration,
                    amplitude=0,
                    sample_rate=sample_rate,
                    pulse_name="final_DC",
                )

            last_waveform_idx = self.sequences[ch][-1][0]
            last_waveform = self.waveforms[ch][last_waveform_idx - 1]
            last_voltage = last_waveform[-1]

            if self.waveforms_initial[ch] is not None:
                waveform_initial_idx, waveform_initial = self.waveforms_initial[ch]

                logger.debug(
                    f"Changing first point of first waveform to {last_voltage}"
                )

                waveform_initial[0] = last_voltage
                self.waveforms[ch][waveform_initial_idx - 1] = waveform_initial

            # Ensure there are at least three sequence instructions
            while len(self.sequences[ch]) < 3:
                waveform_idx = self.add_single_waveform(ch, last_voltage * np.ones(320))
                # Add extra blank segment which will automatically run to
                # the next segment (~ 70 ns offset)
                self.sequences[ch].append((waveform_idx, 1, 0, "final_filler_pulse"))

            # Ensure total waveform points are less than memory limit
            total_waveform_points = sum(
                len(waveform) for waveform in self.waveforms["ch1"]
            )
            if total_waveform_points > self.instrument.waveform_max_length:
                raise RuntimeError(
                    f"Total waveform points {total_waveform_points} exceeds "
                    f"limit of 81180A ({self.instrument.waveform_max_length})"
                )

            # Sequence all loaded waveforms
            waveform_idx_mapping = instrument_channel.upload_waveforms(
                self.waveforms[ch], allow_existing=True
            )
            # Update waveform indices since they may correspond to pre-existing waveforms
            self.sequences[ch] = [
                (waveform_idx_mapping[idx], *instructions)
                for idx, *instructions in self.sequences[ch]
            ]
            instrument_channel.set_sequence(self.sequences[ch])

            # Check that the sample point offsets do not exceed limit
            self.max_point_offsets[ch] = max(np.abs(self.point_offsets[ch]))
            if self.max_point_offsets[ch] > self.point_offset_limit:
                logger.warning(
                    f"81180A maximum sample point offset exceeds limit {self.point_offset_limit}. "
                    f"Current maximum: {self.max_point_offsets}"
                )
            else:
                logger.debug(
                    f"81180A sample point maximum offset: {self.max_point_offsets}"
                )

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
        t_stop: float,
        sample_rate: float,
        pulse_name=None,
    ):
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

    def start(self):
        """Turn all active instrument channels on"""
        for ch_name in self.active_channels():
            instrument_channel = getattr(self.instrument, ch_name)
            instrument_channel.on()

    def stop(self):
        """Turn both instrument channels off"""
        self.instrument.ch1.off()
        self.instrument.ch2.off()


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
                f'Cannot add pulse because the number of waveform points {N} '
                f'is less than the minimum 320. {pulse}'
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
        max_points_exact = settings.pop('max_points_exact', 4000)
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

        # Get waveform points for repeated segment
        t_list = self.pulse.t_start + np.arange(optimum["points"]) / sample_rate
        waveform_array = self.pulse.get_voltage(t_list)

        # Potentially include a waveform tail
        waveform_tail_pts = int(optimum["final_delay"] * sample_rate)
        # Waveform must be multiple of 32, if number of points is less than
        # this, there is no point in adding the waveform
        if waveform_tail_pts >= 32:
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
        else:
            waveform_tail_array = None

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
