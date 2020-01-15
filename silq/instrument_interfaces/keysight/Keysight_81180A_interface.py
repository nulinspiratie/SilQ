import numpy as np
import logging
from typing import List
from warnings import warn

from silq import config
from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import (
    DCPulse,
    TriggerPulse,
    SinePulse,
    MarkerPulse,
    PulseImplementation,
)
from silq.pulses import PulseSequence
from silq.tools.general_tools import arreqclose_in_list, find_approximate_divisor
from silq.tools.pulse_tools import pulse_to_waveform_sequence


from qcodes import ManualParameter, ParameterNode, MatPlot
from qcodes import validators as vals


logger = logging.getLogger(__name__)


class Keysight81180AInterface(InstrumentInterface):
    """

    Notes:
        - When the output is turned on, there is a certain ramping time of a few
          milliseconds. This negatively impacts the first repetition of a
          pulse sequence
        - see ``interface.additional_settings`` for instrument settings that should
          be set manually
    """

    def __init__(self, instrument_name, **kwargs):
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
                    ("amplitude", {"max": 1.5}),
                    ("duration", {"min": 100e-9}),
                ]
            ),
            SinePulseImplementation(
                pulse_requirements=[
                    ("frequency", {"min": 0, "max": 1.5e9}),
                    ("amplitude", {"max": 0.5}),
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

        self.add_parameter(
            'start_at_0V',
            initial_value=False,
            set_cmd=None,
            vals=vals.Bool(),
            docstring='Optionally start sequence with a pulse at 0V. '
                      'This ensures that the first shot of the pulse sequence '
                      'starts at 0V instead of the first voltage of the first pulse. '
                      'Note however that any pulse_sequence.final_delay is then '
                      'also at 0V. If start_at_0V is False, final_delay is fixed '
                      'at the final voltage of the last pulse (default)'
        )

        self.waveforms = {}
        self.sequences = {}

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
        additional_pulses = []

        t_start = np.min(self.pulse_sequence.t_start_list)

        # Request a single trigger at the start
        logger.info(f"Creating trigger for Keysight 81180A: {self.name}")
        return [
            TriggerPulse(
                name=self.name + "_trigger",
                t_start=t_start,
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

            instrument_channel.clear_waveforms()

            instrument_channel.continuous_run_mode(False)
            instrument_channel.function_mode("sequenced")

            # TODO Are these needed? Or set via sequence?
            # instrument_channel.power(5)  # If coupling is AC
            # instrument_channel.voltage_DAC(voltage)  # If coupling is DAC (max 0.5)
            # instrument_channel.voltage_DC(voltage)  # If coupling is DC (max 2)

            instrument_channel.voltage_offset(0)
            instrument_channel.output_modulation('off')

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

        if self.instrument.is_idle() != '1':
            warn('Not idle')

        self.instrument.ensure_idle = True
        self.generate_waveform_sequences()
        self.instrument.ensure_idle = False

    def generate_waveform_sequences(self):
        self.waveforms = {ch: [] for ch in self.active_channels()}
        self.sequences = {ch: [] for ch in self.active_channels()}


        for ch in self.active_channels():
            instrument_channel = self.instrument.channels[ch]
            # Set time t_pulse to start of first pulse in sequence.
            # This will increase as we iterate over pulses, and is used to ensure
            # that there are no gaps between pulses
            t_pulse = min(self.pulse_sequence.t_start_list)

            # A waveform must have at least 320 points
            min_waveform_duration = 320 / instrument_channel.sample_rate()

            pulse_sequence = PulseSequence(
                self.pulse_sequence.get_pulses(output_channel=ch)
            )

            # Always begin by waiting for a trigger/event pulse
            # Add empty waveform (0V DC), with minimum points (320)
            empty_idx = self.add_waveform(ch, waveform_array=np.zeros(320))

            # Optionally start sequence with a pulse at 0V.
            # This ensures that the first shot of the pulse sequence starts at 0V
            # instead of the first voltage of the first pulse.
            # Note however that any pulse_sequence.final_delay is then also at 0V
            # If start_at_0V is False, final_delay is fixed at the final voltage
            # of the last pulse
            if self.start_at_0V():
                self.sequences[ch].append((empty_idx, 1, 0, 'initial_pulse'))

            for pulse in pulse_sequence.get_pulses():
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
                    DC_sequence_steps = self._add_DC_waveform(
                        channel_name=ch,
                        amplitude=0,
                        duration=pulse.t_start - t_pulse,
                        sample_rate=instrument_channel.sample_rate(),
                    )
                    self.sequences[ch] += DC_sequence_steps

                # Get waveform of current pulse
                waveform = pulse.implementation.implement(
                    sample_rate=instrument_channel.sample_rate(),
                    trigger_duration=self.trigger_in_duration(),
                )

                waveform_idx = self.add_waveform(ch, waveform["waveform"])

                # Add sequence step (waveform_idx, loops, jump_event)
                self.sequences[ch].append((waveform_idx, waveform["loops"], 0, pulse.name))

                # Also add waveform tail if needed
                if waveform["waveform_tail"] is not None:
                    waveform_tail_idx = self.add_waveform(ch, waveform["waveform_tail"])
                    self.sequences[ch].append((waveform_tail_idx, 1, 0, f'{pulse.name}_tail'))

                # Set current time to pulse.t_stop
                t_pulse = pulse.t_stop

            # Add 0V pulse if last pulse does not stop at pulse_sequence.duration
            if self.pulse_sequence.duration - t_pulse >= min_waveform_duration + 1e-11:
                final_DC_sequence_steps = self._add_DC_waveform(
                    channel_name=ch,
                    amplitude=0,
                    duration=self.pulse_sequence.duration - t_pulse,
                    sample_rate=instrument_channel.sample_rate(),
                )
                self.sequences[ch] += final_DC_sequence_steps

            # Ensure there are at least three sequence instructions
            while len(self.sequences[ch]) < 3:
                # Add extra blank segment which will automatically run to
                # the next segment (~ 70 ns offset)
                self.sequences[ch].append((empty_idx, 1, 0, 'final_filler_pulse'))

            # Sequence all loaded waveforms
            instrument_channel.set_sequence(self.sequences[ch])

    def _add_DC_waveform(
        self, channel_name: str, amplitude: float, duration: float, sample_rate: float
    ) -> List:
        # Create waveform from DCPulseImplementation
        waveform = DCPulseImplementation.implement(
            self=None, sample_rate=sample_rate, amplitude=amplitude, duration=duration
        )
        # Add waveform(s) and sequence steps
        waveform_idx = self.add_waveform(
            channel_name, waveform_array=waveform["waveform"]
        )

        sequence_steps = [(waveform_idx, waveform["loops"], 0, 'DC_pulse')]

        # Add separate waveform if there are remaining points left after division
        if waveform["waveform_tail"] is not None:
            waveform_idx = self.add_waveform(
                channel_name, waveform_array=waveform["waveform_tail"]
            )
            sequence_steps.append((waveform_idx, 1, 0, 'DC_tail'))

        return sequence_steps

    def add_waveform(
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
                waveform_array, self.waveforms[channel_name]
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

            # Upload waveform to instrument
            logger.debug(
                f"Uploading new waveform with {len(waveform_array)} points and "
                f"index {waveform_idx}"
            )
            instrument_channel = self.instrument.channels[channel_name]
            instrument_channel.add_waveform(waveform_array, waveform_idx)

        return waveform_idx

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

    def implement(
        self,
        sample_rate: float,
        max_points: int = 6000,
        trigger_duration: float = None,
        amplitude: float = None,
        duration: float = None,
    ) -> dict:
        if amplitude is None:
            amplitude = self.pulse.amplitude
        if duration is None:
            duration = self.pulse.duration

        # Number of points must be a multiple of 32
        N = 32 * np.floor(duration * sample_rate / 32)

        # Find an approximate divisor of the number of points N, allowing us
        # to shrink the waveform
        approximate_divisor = find_approximate_divisor(
            N=N,
            max_cycles=1000000,
            points_multiple=32,
            min_points=320,
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
        waveform = amplitude * np.ones(approximate_divisor["points"])

        # Add separate waveform if there are remaining points left after division
        if approximate_divisor["remaining_points"]:
            waveform_tail = amplitude * np.ones(approximate_divisor["remaining_points"])
        else:
            waveform_tail = None

        return {
            "waveform": waveform,
            "loops": approximate_divisor["cycles"],
            "waveform_tail": waveform_tail,
        }


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def implement(self, sample_rate, trigger_duration, plot=False, **kwargs):
        settings = {"max_points": 50e3, "frequency_threshold": 30}
        settings.update(**config.properties.get("sine_waveform_settings", {}))
        settings.update(**kwargs)

        self.results = pulse_to_waveform_sequence(
            frequency=self.pulse.frequency,
            sampling_rate=sample_rate,
            total_duration=self.pulse.duration,
            min_points=320,
            sample_points_multiple=32,
            plot=plot,
            **settings,
        )
        optimum = self.results["optimum"]
        waveform_loops = max(optimum["repetitions"], 1)

        # Temporarily modify pulse frequency to ensure waveforms have full period
        original_frequency = self.pulse.frequency
        self.pulse.frequency = optimum['modified_frequency']

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
                subtract_loops = int(np.ceil((320 - waveform_tail_pts) / optimum["points"]))
            else:
                subtract_loops = 0

            if waveform_loops - subtract_loops > 0:
                # Safe to subtract loops from the main waveform, add to this one
                waveform_loops -= subtract_loops
                waveform_tail_pts += subtract_loops * optimum['points']

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
                wf_t_list[new_wf_idxs], wf_voltages[new_wf_idxs], "o",
                color="C2", ms=4
            )
            wf_tail_idx = len(waveform_array) * waveform_loops
            ax.plot(
                wf_t_list[wf_tail_idx], wf_voltages[wf_tail_idx], "o",
                color="C3", ms=4
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
            "waveform_tail": waveform_tail_array,
        }
