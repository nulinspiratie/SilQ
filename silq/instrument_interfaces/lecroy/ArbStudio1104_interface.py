import numpy as np
import logging
from typing import List
import time

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import Pulse, DCPulse, DCRampPulse, TriggerPulse, SinePulse, \
    MultiSinePulse, FrequencyRampPulse, PulseImplementation, MarkerPulse
from silq import config
from silq.tools.pulse_tools import pulse_to_waveform_sequence

from qcodes.utils.helpers import arreqclose_in_list


logger = logging.getLogger(__name__)


class ArbStudio1104Interface(InstrumentInterface):
    """ Interface for the LeCroy Arbstudio 1104

    When a `PulseSequence` is targeted in the `Layout`, the
    pulses are directed to the appropriate interface. Each interface is
    responsible for translating all pulses directed to it into instrument
    commands. During the actual measurement, the instrument's operations will
    correspond to that required by the pulse sequence.

    The interface also contains a list of all available channels in the
    instrument.

    Args:
        instrument_name: name of instrument for which this is an interface

    Note:
        For a given instrument, its associated interface can be found using
            `get_instrument_interface`

    Todo:
        * Add modes other than stepped.
        * Add use as primary instrument.


    # Arbstudio interface information
    The interface is programmed for stepped mode, meaning that the arbstudio
    needs a triger after every pulse. Consequently, the interface is not
    programmed to be the primary instrument.
    The arbstudio needs a trigger at the end of the sequence to restart.
    Note that this means that during any final delay, the arbstudio will
    already have restarted its sequence. Not sure how to best deal with this.

    """
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, output=True)
        for k in [1, 2, 3, 4]}

        self._channels = {**self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True),
            'trig_out': Channel(instrument_name=self.instrument_name(),
                                name='trig_out', output_TTL=(0, 3.3))}
        # TODO check Arbstudio output TTL high voltage

        self.pulse_implementations = [SinePulseImplementation(
            pulse_requirements=[('frequency', {'min': 1e2, 'max': 125e6})]),
            MultiSinePulseImplementation(),
            FrequencyRampPulseImplementation(),
            DCPulseImplementation(),
            DCRampPulseImplementation(),
            MarkerPulseImplementation(),
            TriggerPulseImplementation()]

        self.add_parameter('trigger_in_duration',
                           set_cmd=None,
                           unit='s',
                           initial_value=100e-9)
        self.add_parameter('pulse_final_delay',
                           set_cmd=None,
                           unit='s', initial_value=1e-6,
                           docstring='Final delay up to the end of pulses that '
                                     'have full waveforms, to ensure that the '
                                     'waveform is finished before the next '
                                     'trigger arrives. This does not count for '
                                     'pulses such as DCPulse, which only have '
                                     'four points')

        self.add_parameter('active_channels', get_cmd=self._get_active_channels)

    def _get_active_channels(self) -> List[str]:
        """Get all active channels used in pulses"""
        active_channels = [pulse.connection.output['channel'].name for pulse in
                           self.pulse_sequence]
        # Transform into set to ensure that elements are unique
        return list({pulse.connection.output['channel'].name
                     for pulse in self.pulse_sequence})

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses.
        """
        # Return empty list if no pulses are in the pulse sequence
        if not self.pulse_sequence or self.is_primary():
            return []

        additional_pulses = []

        # Get current trigger pulse times. Even though there probably isn't a
        # trigger pulse in the input_pulse_sequence yet, add these anyway
        t_trigger_pulses = [pulse.t_start for pulse in self.input_pulse_sequence
                            if isinstance(pulse, TriggerPulse)]
        for t_start in self.pulse_sequence.t_start_list:
            if t_start == 0 or t_start in t_trigger_pulses:
                # No trigger at t=0 since the first waveform plays immediately
                # Also no trigger if it already exists
                continue
            else:
                additional_pulses.append(self._get_trigger_pulse(t_start))
                t_trigger_pulses.append(t_start)

        # Loop over channels ensuring that all channels are programmed for each
        # trigger segment
        for ch in self.active_channels():
            pulses = self.pulse_sequence.get_pulses(output_channel=ch)
            connection = self.pulse_sequence.get_connection(output_channel=ch)
            t = 0
            while t < self.pulse_sequence.duration:
                remaining_pulses = [pulse for pulse in pulses if
                                    pulse.t_start >= t]
                if not remaining_pulses:
                    # Add final DC pulse at amplitude zero
                    dc_pulse = self.get_pulse_implementation(
                        DCPulse(
                            'final_pulse',
                            t_start=t,
                            t_stop=self.pulse_sequence.duration,
                            amplitude=0,
                            connection=connection)
                    )
                    self.pulse_sequence.add(dc_pulse)

                    # Check if trigger pulse is necessary.
                    if t > 0 and t not in t_trigger_pulses:
                        additional_pulses.append(self._get_trigger_pulse(t))
                        t_trigger_pulses.append(t)

                    t = self.pulse_sequence.duration
                else:
                    # Determine start time of next pulse
                    t_next = min(pulse.t_start for pulse in remaining_pulses)
                    pulse_next = [pulse for pulse in remaining_pulses if
                                  pulse.t_start == t_next][0]
                    if t_next > t:
                        # Add DC pulse at amplitude zero
                        dc_pulse = self.get_pulse_implementation(
                            DCPulse(t_start=t, t_stop=t_next, amplitude=0,
                                    connection=connection))
                        self.pulse_sequence.add(dc_pulse)

                        # Check if trigger pulse is necessary.
                        if t > 0 and t not in t_trigger_pulses:
                            additional_pulses.append(self._get_trigger_pulse(t))
                            t_trigger_pulses.append(t)

                    # Set time to t_stop of next pulse
                    t = pulse_next.t_stop

        # Add a trigger pulse at the end if it does not yet exist
        if self.pulse_sequence.duration not in t_trigger_pulses:
            trigger_pulse = self._get_trigger_pulse(self.pulse_sequence.duration)
            additional_pulses.append(trigger_pulse)
            t_trigger_pulses.append(self.pulse_sequence.duration)

        return additional_pulses

    def setup(self, **kwargs):
        """Set up instrument after layout has been targeted by pulse sequence.

        Args:
            **kwargs: Unused setup kwargs passed by `Layout`
        """
        # TODO implement setup for modes other than stepped

        assert not self.is_primary(), 'Arbstudio is currently not programmed ' \
                                      'to function as primary instrument'

        # Clear waveforms and sequences
        for ch in self._output_channels.values():
            self.instrument._waveforms = [[] for k in range(4)]
            exec(f'self.instrument.ch{ch.id}_sequence([])')

        # Generate waveforms and sequences
        self.generate_waveforms_sequences()

        for ch in self.active_channels():

            self.instrument.parameters[f'{ch}_trigger_source']('fp_trigger_in')
            self.instrument.functions[f'{ch}_clear_waveforms']()

            if self.pulse_sequence.get_pulses(output_channel=ch,
                                              pulse_class=SinePulse):
                # TODO better check for when to use burst or stepped mode
                self.instrument.parameters[f'{ch}_trigger_mode']('burst')
            else:
                self.instrument.parameters[f'{ch}_trigger_mode']('stepped')

            # Add waveforms to channel
            for waveform in self.waveforms[ch]:
                self.instrument.functions[f'{ch}_add_waveform'](waveform)

            # Add sequence to channel
            self.instrument.parameters[f'{ch}_sequence'](self.sequences[ch])

        active_channels_id = [self._channels[channel].id for channel in
                              self.active_channels()]
        self.instrument.load_waveforms(channels=active_channels_id)
        self.instrument.load_sequence(channels=active_channels_id)

    def start(self):
        """Start instrument"""
        try:
            self.instrument.run(channels=[self._channels[channel].id for channel in
                                          self.active_channels()])
        except AssertionError:
            logger.error('Driver connection error when starting arbstudio, retrying.')
            time.sleep(3)
            self.instrument.run(channels=[self._channels[channel].id for channel in
                                          self.active_channels()])

    def stop(self):
        """Stop instrument"""
        self.instrument.stop()

    def _get_trigger_pulse(self, t: float) -> TriggerPulse:
        """Create input trigger pulse

        Args:
            t: trigger start time

        Returns:
            Trigger pulse with specified start time
        """
        trigger_pulse = TriggerPulse(t_start=t,
            duration=self.trigger_in_duration(),
            connection_requirements={'input_instrument': self.instrument_name(),
                'trigger': True})
        return trigger_pulse

    def generate_waveforms_sequences(self) -> List:
        """Generate waveforms and sequence

        Updates self.waveforms and self.sequence.

        Returns:
            List of waveforms
        """
        # Determine sampling rates
        sampling_rates = {ch: 250e6 / self.instrument.parameters[
            ch + '_sampling_rate_prescaler']() for ch in self.active_channels()}

        # Set time t_pulse to zero for each channel
        # This will increase as we iterate over pulses, and is used to ensure
        # that there are no times between pulses
        t_pulse = {ch: 0 for ch in self.active_channels()}

        self.waveforms = {ch: [] for ch in self.active_channels()}
        self.sequences = {ch: [] for ch in self.active_channels()}
        for pulse in self.pulse_sequence:

            # For each channel, obtain list of waverforms, and the sequence
            # in which to perform the waveforms
            channels_waveforms, channels_sequence = \
                pulse.implementation.implement(
                    sampling_rates=sampling_rates,
                    input_pulse_sequence=self.input_pulse_sequence)

            for ch in channels_waveforms:
                # Ensure that the start of this pulse corresponds to the end of
                # the previous pulse for each channel
                assert abs(pulse.t_start - t_pulse[
                    ch]) < 1e-11, f"Pulse {pulse}: pulses.t_start = " \
                                  f"{pulse.t_start} does not match {t_pulse[ch]}"

                channel_waveforms = channels_waveforms[ch]
                channel_sequence = channels_sequence[ch]

                # Only add waveforms that don't already exist
                for waveform in channel_waveforms:
                    waveform_idx = arreqclose_in_list(waveform, self.waveforms[ch],
                                                      rtol=1e-4, atol=1e-5)
                    if waveform_idx is None:
                        self.waveforms[ch].append(waveform)

                # Finding the correct sequence idx for each item in
                # channel_sequence. By first adding waveforms and then
                # finding the correct sequence indices, we ensure that there
                # are no duplicate waveforms and each sequence idx is correct.
                for k, sequence_idx in enumerate(channel_sequence):
                    waveform = channel_waveforms[sequence_idx]
                    waveform_idx = arreqclose_in_list(waveform,
                                                      self.waveforms[ch],
                                                      rtol=1e-4, atol=1e-4)
                    # Update channel_sequence item to correct index
                    channel_sequence[k] = waveform_idx
                self.sequences[ch].extend(channel_sequence)

                # Increase t_pulse to match start of next pulses
                t_pulse[ch] += pulse.duration

        # Ensure that the start of this pulse corresponds to the end of
        # the previous pulse for each channel
        for ch in self.active_channels():
            assert abs(t_pulse[ch] - self.pulse_sequence.duration) < 1e-11, \
                f"Final pulse of channel {ch} ends at {t_pulse[ch]} " \
                f"instead of {self.pulse_sequence.duration}"
        return self.waveforms


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse
    max_waveform_points = 300e3

    def target_pulse(self, pulse, interface, **kwargs):
        targeted_pulse = super().target_pulse(pulse, interface, **kwargs)

        # Set final delay from interface parameter
        targeted_pulse.implementation.final_delay = interface.pulse_final_delay()
        return targeted_pulse

    def implement(self, sampling_rates, input_pulse_sequence, plot=False, **kwargs):
        """
        Implements the sine pulse for the ArbStudio for SingleConnection.
        Args:

        Returns:
            waveforms: {output_channel: waveforms} dictionary for each output
                channel, where each element in waveforms is a list
                containing the voltage levels of the waveform
            waveforms: {output_channel: sequence} dictionary for each
                output channel, where each element in sequence indicates the
                waveform that must be played after the trigger
        """
        # Find all trigger pulses occuring within this pulse
        trigger_pulses = input_pulse_sequence.get_pulses(
            t_start=('>', self.pulse.t_start), t_stop=('<', self.pulse.t_stop),
            trigger=True)
        assert len(trigger_pulses) == 0, \
            "Cannot implement sine pulse if the arbstudio receives intermediary triggers"

        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception(f"No implementation for connection {self.pulse.connection}")

        # assert self.pulse.frequency < min(sampling_rates[ch] for ch in channels)/2, \
        #     'Sine frequency is higher than the Nyquist limit for channels {channels}'

        assert self.pulse.frequency > 0, "Pulse frequency must be larger than zero."

        waveforms, sequences = {}, {}
        for ch in channels:
            sample_rate = sampling_rates[ch]
            points_per_period = int(sample_rate / self.pulse.frequency)
            points_per_period = points_per_period - points_per_period % 2

            if points_per_period > self.max_waveform_points:
                raise RuntimeError(
                    f"Sine waveform points {points_per_period} is above maximum "
                    f"{self.max_waveform_points}. Could not segment sine waveform"
                )
            elif points_per_period > 1000:
                # Temporarily modify pulse frequency to ensure waveforms have full period
                original_frequency = self.pulse.frequency
                modified_frequency = sample_rate / points_per_period
                self.pulse.frequency = modified_frequency

                t_list = self.pulse.t_start + np.arange(points_per_period) / sample_rate
                voltages = self.pulse.get_voltage(t_list)

                self.pulse.frequency = original_frequency
            else:
                periods = 50000 // points_per_period
                waveform_points = int(periods * points_per_period)
                t_list = self.pulse.t_start + np.arange(waveform_points) / sample_rate
                voltages = self.pulse.get_voltage(t_list)

            waveforms[ch] = [voltages]
            sequences[ch] = np.zeros(1, dtype=int)

        return waveforms, sequences


class MultiSinePulseImplementation(PulseImplementation):
    pulse_class = MultiSinePulse

    def target_pulse(self, pulse, interface, **kwargs):
        targeted_pulse = super().target_pulse(pulse, interface, **kwargs)

        # Set final delay from interface parameter
        targeted_pulse.implementation.final_delay = interface.pulse_final_delay()
        return targeted_pulse

    def implement(self, sampling_rates, input_pulse_sequence, plot=False, **kwargs):
        """
        Implements the multi sine pulse for the ArbStudio for SingleConnection.
        Args:

        Returns:
            waveforms: {output_channel: waveforms} dictionary for each output
                channel, where each element in waveforms is a list
                containing the voltage levels of the waveform
            waveforms: {output_channel: sequence} dictionary for each
                output channel, where each element in sequence indicates the
                waveform that must be played after the trigger
        """
        # Find all trigger pulses occurring within this pulse
        trigger_pulses = input_pulse_sequence.get_pulses(
            t_start=('>', self.pulse.t_start),
            t_stop=('<', self.pulse.t_stop),
            trigger=True)
        assert len(trigger_pulses) == 0, \
            "Cannot implement multi sine pulse if the Arbstudio receives intermediary triggers"

        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception(f"No implementation for connection {self.pulse.connection}")

        waveforms, sequences = {}, {}
        for ch in channels:
            total_points = self.pulse.duration * sampling_rates[ch]
            final_points = self.final_delay * sampling_rates[ch]
            # Waveform points subtract the final waveform delay
            waveform_points = int(round(total_points - final_points))

            # All waveforms must have an even number of points
            if waveform_points % 2:
                waveform_points -= 1

            t_list = self.pulse.t_start + np.arange(waveform_points) / sampling_rates[ch]
            voltages = self.pulse.get_voltage(t_list)

            waveforms[ch] = [voltages]
            sequences[ch] = np.zeros(1, dtype=int)

        return waveforms, sequences


class FrequencyRampPulseImplementation(PulseImplementation):
    pulse_class = FrequencyRampPulse

    def target_pulse(self, pulse, interface, **kwargs):
        targeted_pulse = super().target_pulse(pulse, interface, **kwargs)

        # Set final delay from interface parameter
        targeted_pulse.implementation.final_delay = interface.pulse_final_delay()
        return targeted_pulse

    def implement(self, sampling_rates, input_pulse_sequence, plot=False, **kwargs):
        """
        Implements the frequency ramp pulse for the ArbStudio for SingleConnection.
        Args:

        Returns:
            waveforms: {output_channel: waveforms} dictionary for each output
                channel, where each element in waveforms is a list
                containing the voltage levels of the waveform
            waveforms: {output_channel: sequence} dictionary for each
                output channel, where each element in sequence indicates the
                waveform that must be played after the trigger
        """
        # Find all trigger pulses occuring within this pulse
        trigger_pulses = input_pulse_sequence.get_pulses(
            t_start=('>', self.pulse.t_start),
            t_stop=('<', self.pulse.t_stop),
            trigger=True)
        assert len(trigger_pulses) == 0, \
            "Cannot implement frequency ramp pulse if the arbstudio receives intermediary triggers"

        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception(f"No implementation for connection {self.pulse.connection}")

        assert self.pulse.frequency_start > 0 and \
               self.pulse.frequency_stop > 0, "Start and stop frequencies in pulse must be larger than zero."

        waveforms, sequences = {}, {}
        for ch in channels:
            total_points = self.pulse.duration * sampling_rates[ch]
            final_points = self.final_delay * sampling_rates[ch]
            # Waveform points subtract the final waveform delay
            waveform_points = int(round(total_points - final_points))

            # All waveforms must have an even number of points
            if waveform_points % 2:
                waveform_points -= 1

            t_list = self.pulse.t_start + np.arange(waveform_points) / sampling_rates[ch]
            voltages = self.pulse.get_voltage(t_list)

            waveforms[ch] = [voltages]
            sequences[ch] = np.zeros(1, dtype=int)

        return waveforms, sequences


class DCPulseImplementation(PulseImplementation):
    # Number of points in a waveform (must be at least 4)
    pulse_class = DCPulse
    pts = 4

    def implement(self, input_pulse_sequence, **kwargs):
        """
        Implements the DC pulses for the ArbStudio for SingleConnection. If
        the input pulse sequence contains triggers in between the DC pulse,
        the output sequence will repeat the waveform multiple times

        Args:
            input_pulse_sequence: Arbstudio input pulsesfrom which the
                triggering determines how how often the sequence repeats the
                waveform

        Returns:
            waveforms: {output_channel: waveforms} dictionary for each output
                channel, where each element in waveforms is a list
                containing the voltage levels of the waveform.
            waveforms: {output_channel: sequence} dictionary for each
                output channel, where each element in sequence indicates the
                waveform that must be played after the trigger
        """

        # Find all trigger pulses occuring within this pulse
        trigger_pulses = input_pulse_sequence.get_pulses(
            t_start=('>', self.pulse.t_start),
            t_stop=('<', self.pulse.t_stop),
            trigger=True)

        # Arbstudio requires a minimum of four points to be returned
        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception(f"No implementation for connection "
                            f"{self.pulse.connection}")

        waveforms = {ch: [np.ones(self.pts) * self.pulse.amplitude]
                     for ch in channels}
        sequences = {ch: np.zeros(len(trigger_pulses) + 1, dtype=int)
                     for ch in channels}

        return waveforms, sequences


class DCRampPulseImplementation(PulseImplementation):
    pulse_class = DCRampPulse

    def target_pulse(self, pulse, interface, **kwargs):
        targeted_pulse = super().target_pulse(pulse, interface, **kwargs)

        # Set final delay from interface parameter
        targeted_pulse.implementation.final_delay = interface.pulse_final_delay()
        return targeted_pulse

    def implement(self, sampling_rates, input_pulse_sequence, **kwargs):
        """
        Implements the DC pulses for the ArbStudio for SingleConnection and
        CombinedConnection. For a CombinedConnection, it weighs the DC pulses
        amplitude by the corresponding channel scaling factor (default 1).
        Args:

        Returns:
            waveforms: {output_channel: waveforms} dictionary for each output
                channel, where each element in waveforms is a list
                containing the voltage levels of the waveform
            waveforms: {output_channel: sequence} dictionary for each
                output channel, where each element in sequence indicates the
                waveform that must be played after the trigger
        """
        # Find all trigger pulses occuring within this pulse
        trigger_pulses = input_pulse_sequence.get_pulses(
            t_start=('>', self.pulse.t_start),
            t_stop=('<', self.pulse.t_stop),
            trigger=True)
        assert len(trigger_pulses) == 0, \
            "Cannot implement DC ramp pulse if the arbstudio receives " \
            "intermediary triggers"

        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception(f"No implementation for connection {self.pulse.connection}")

        waveforms, sequences = {}, {}
        for ch in channels:
            total_points = self.pulse.duration * sampling_rates[ch]
            final_points = self.final_delay * sampling_rates[ch]
            # Waveform points subtract the final waveform delay
            waveform_points = int(round(total_points - final_points))

            # All waveforms must have an even number of points
            if waveform_points % 2:
                waveform_points -= 1

            waveforms[ch] = [np.linspace(self.pulse.amplitude_start,
                                         self.pulse.amplitude_stop,
                                         waveform_points)]
            sequences[ch] = np.zeros(1, dtype=int)

        return waveforms, sequences


class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse
    pts = 4

    def implement(self, sampling_rates, input_pulse_sequence, **kwargs):
        """
        Implements the DC pulses for the ArbStudio for SingleConnection. If
        the input pulse sequence contains triggers in between the DC pulse,
        the output sequence will repeat the waveform multiple times

        Args:
            input_pulse_sequence: Arbstudio input pulsesfrom which the
                triggering determines how how often the sequence repeats the
                waveform

        Returns:
            waveforms: {output_channel: waveforms} dictionary for each output
                channel, where each element in waveforms is a list
                containing the voltage levels of the waveform
            waveforms: {output_channel: sequence} dictionary for each
            output channel, where each element in sequence indicates the
            waveform that must be played after the trigger
        """

        # Find all trigger pulses occuring within this pulse
        trigger_pulses = input_pulse_sequence.get_pulses(
            t_start=('>', self.pulse.t_start),
            t_stop=('<', self.pulse.t_stop),
            trigger=True)

        # Arbstudio requires a minimum of four points to be returned
        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception(f"No implementation for connection "
                            f"{self.pulse.connection}")

        waveforms = {ch: [np.ones(self.pts) * self.pulse.amplitude]
                     for ch in channels}
        sequences = {ch: np.zeros(len(trigger_pulses) + 1, dtype=int)
                     for ch in channels}

        return waveforms, sequences


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse
    # TODO add implement method
