import numpy as np
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, DCRampPulse, TriggerPulse, SinePulse, \
    PulseImplementation
from silq.tools.general_tools import arreqclose_in_list

from qcodes.instrument.parameter import ManualParameter


class ArbStudio1104Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                                      name='ch{}'.format(k), id=k, output=True)
        for k in [1, 2, 3, 4]}

        self._channels = {**self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True),
            'trig_out': Channel(instrument_name=self.instrument_name(),
                                name='trig_out', output_TTL=(0, 3.3))}
        # TODO check Arbstudio output TTL high

        self.pulse_implementations = [SinePulseImplementation(
            pulse_requirements=[('frequency', {'min': 1e6, 'max': 125e6})]),
            DCPulseImplementation(pulse_requirements=[]),
            DCRampPulseImplementation(pulse_requirements=[]),
            TriggerPulseImplementation(pulse_requirements=[])]

        self.add_parameter('trigger_in_duration',
                           parameter_class=ManualParameter, unit='us',
                           initial_value=0.1)
        self.add_parameter('final_delay', parameter_class=ManualParameter,
                           unit='us', initial_value=0.2)

        self.add_parameter('active_channels', get_cmd=self._get_active_channels)

    def _get_active_channels(self):
        active_channels = [pulse.connection.output['channel'].name for pulse in
                           self.pulse_sequence]
        # Transform into set to ensure that elements are unique
        active_channels = list(set(active_channels))
        return active_channels

    def setup(self, **kwargs):
        # TODO implement setup for modes other than stepped

        # Clear waveforms and sequences
        for ch in self._output_channels.values():
            self.instrument._waveforms = [[] for k in range(4)]
            exec(f'self.instrument.ch{ch.id}_sequence([])')

        # Generate waveforms and sequences
        self.generate_waveforms_sequences()

        for ch in self.active_channels():

            self.instrument.parameters[ch + '_trigger_source']('fp_trigger_in')
            self.instrument.functions[ch + '_clear_waveforms']()

            if self.pulse_sequence.get_pulses(output_channel=ch,
                                              pulse_class=SinePulse):
                # TODO better check for when to use burst or stepped mode
                self.instrument.parameters[ch + '_trigger_mode']('burst')
            else:
                self.instrument.parameters[ch + '_trigger_mode']('stepped')

            # Add waveforms to channel
            for waveform in self.waveforms[ch]:
                self.instrument.functions[ch + '_add_waveform'](waveform)

            # Add sequence to channel
            self.instrument.parameters[ch + '_sequence'](self.sequences[ch])

        active_channels_id = [self._channels[channel].id for channel in
                              self.active_channels()]
        self.instrument.load_waveforms(channels=active_channels_id)
        self.instrument.load_sequence(channels=active_channels_id)

    def start(self):
        self.instrument.run(channels=[self._channels[channel].id for channel in
                                      self.active_channels()])

    def stop(self):
        self.instrument.stop()

    def get_additional_pulses(self, **kwargs):
        final_pulses = []

        # Return empty list if no pulses are in the pulse sequence
        if not self.pulse_sequence:
            return final_pulses

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
                        DCPulse(t_start=t, t_stop=self.pulse_sequence.duration,
                                amplitude=0, connection=connection))
                    self.pulse_sequence.add(dc_pulse)

                    # Check if trigger pulse is necessary.
                    if dc_pulse.additional_pulses:
                        trigger_pulse = dc_pulse.additional_pulses[0]
                        if trigger_pulse not in final_pulses:
                            final_pulses.append(trigger_pulse)

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

                        # Check if trigger pulse is necessary. Only the case when
                        #  no trigger pulse already exists at the same time,
                        # when t > 0 (first trigger occurs at the end)
                        if dc_pulse.additional_pulses:
                            trigger_pulse = dc_pulse.additional_pulses[0]
                            if trigger_pulse not in final_pulses:
                                final_pulses.append(trigger_pulse)

                    # Set time to t_stop of next pulse
                    t = pulse_next.t_stop

        # Add a trigger pulse at the end if it does not yet exist
        trigger_pulse = self.get_trigger_pulse(self.pulse_sequence.duration)
        if trigger_pulse not in self.input_pulse_sequence and trigger_pulse not in final_pulses:
            final_pulses.append(trigger_pulse)

        return final_pulses

    def get_trigger_pulse(self, t):
        trigger_pulse = TriggerPulse(t_start=t,
            duration=self.trigger_in_duration() * 1e-3,
            connection_requirements={'input_instrument': self.instrument_name(),
                'trigger': True})
        return trigger_pulse

    def generate_waveforms_sequences(self):

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
            channels_waveforms, channels_sequence = pulse.implement(
                sampling_rates=sampling_rates,
                input_pulse_sequence=self.input_pulse_sequence)

            for ch in channels_waveforms:
                # Ensure that the start of this pulse corresponds to the end of
                # the previous pulse for each channel
                assert abs(pulse.t_start - t_pulse[
                    ch]) < 1e-11, "Pulse {}: pulses.t_start = {} does not match {}".format(
                    pulse, pulse.t_start, t_pulse[ch])

                channel_waveforms = channels_waveforms[ch]
                channel_sequence = channels_sequence[ch]

                # Only add waveforms that don't already exist
                for waveform in channel_waveforms:
                    waveform_idx = arreqclose_in_list(waveform, self.waveforms[ch])
                    if waveform_idx is None:
                        self.waveforms[ch].append(waveform)

                # Finding the correct sequence idx for each item in
                # channel_sequence. By first adding waveforms and then
                # finding the correct sequence indices, we ensure that there
                # are no duplicate waveforms and each sequence idx is correct.
                for k, sequence_idx in enumerate(channel_sequence):
                    waveform = channel_waveforms[sequence_idx]
                    waveform_idx = arreqclose_in_list(waveform,
                                                      self.waveforms[ch])
                    # Update channel_sequence item to correct index
                    channel_sequence[k] = waveform_idx
                self.sequences[ch].extend(channel_sequence)

                # Increase t_pulse to match start of next pulses
                t_pulse[ch] += pulse.duration

        # Ensure that the start of this pulse corresponds to the end of
        # the previous pulse for each channel
        for ch in self.active_channels():
            assert abs(t_pulse[ch] - self.pulse_sequence.duration) < 1e-11, \
                "Final pulse of channel {} ends at {} instead of {}".format(
                ch, t_pulse[ch], self.pulse_sequence.duration)
        return self.waveforms


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(self, pulse,
            interface=interface, is_primary=is_primary, **kwargs)

        # Set final delay from interface parameter
        targeted_pulse.final_delay = interface.final_delay()

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence.get_pulses(
            t_start=pulse.t_start, trigger=True)

        if not (is_primary or trigger_pulses or targeted_pulse.t_start == 0):
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=targeted_pulse.t_start,
                             duration=interface.trigger_in_duration() * 1e-3,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))
        return targeted_pulse

    def implement(self, sampling_rates, input_pulse_sequence, **kwargs):
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
            t_start=('>', self.t_start), t_stop=('<', self.t_stop),
            trigger=True)
        assert len(
            trigger_pulses) == 0, "Cannot implement sine pulse if the arbstudio receives " \
                                  "intermediary triggers"

        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception(
                "No implementation for connection {}".format(self.connection))

        assert self.frequency < min(sampling_rates[ch] for ch in
                                    channels) / 2, 'Sine frequency is higher than the Nyquist limit ' \
                                                   'for channels {}'.format(
            channels)

        waveforms, sequences = {}, {}

        # If the sampling rate is too high, the waveform for the full
        # duration requires too many points. We therefore find a duration
        # that does not create too many points, and generate the waveform.
        # TODO implement full waveform if sampling rate is not too high
        self.waveform_duration = 1  # us
        assert self.waveform_duration * 1e-6 * self.frequency > 10, "sine pulse frequency {} too low, increase waveform " \
                                                                    "duration".format(
            self.frequency)
        assert self.waveform_duration * 1e-3 < self.duration, "Waveform duration too long"

        for ch in channels:
            # TODO choose number of points to minimize the phase accumulation
            # Start t_list from t_start to ensure phase is taken into account
            t_list = np.arange(self.t_start,
                               self.t_start + self.waveform_duration * 1e-3,
                               1 / sampling_rates[ch] * 1e3)  # ms
            if len(t_list) % 2:
                t_list = t_list[:-1]

            waveforms[ch] = [self.get_voltage(t_list)]
            sequences[ch] = np.zeros(1, dtype=int)

        return waveforms, sequences


class DCPulseImplementation(PulseImplementation, DCPulse):
    # Number of points in a waveform (must be at least 4)
    pulse_class = DCPulse
    pts = 4

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(self, pulse,
            interface=interface, is_primary=is_primary, **kwargs)

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence.get_pulses(
            t_start=pulse.t_start, trigger=True)

        if not (is_primary or trigger_pulses or targeted_pulse.t_start == 0):
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=interface.trigger_in_duration() * 1e-3,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))
        return targeted_pulse

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
                containing the voltage levels of the waveform
            waveforms: {output_channel: sequence} dictionary for each
            output channel, where each element in sequence indicates the
            waveform that must be played after the trigger
        """

        # Find all trigger pulses occuring within this pulse
        trigger_pulses = input_pulse_sequence.get_pulses(
            t_start=('>', self.t_start), t_stop=('<', self.t_stop),
            trigger=True)

        # Arbstudio requires a minimum of four points to be returned
        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception(
                "No implementation for connection {}".format(self.connection))

        waveforms = {ch: [np.ones(self.pts) * self.amplitude]
                     for ch in channels}
        sequences = {ch: np.zeros(len(trigger_pulses) + 1, dtype=int) for ch in
                     channels}

        return waveforms, sequences


class DCRampPulseImplementation(PulseImplementation):
    pulse_class = DCRampPulse

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(self, pulse,
            interface=interface, is_primary=is_primary, **kwargs)

        # Set final delay from interface parameter
        targeted_pulse.final_delay = interface.final_delay()

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence.get_pulses(
            t_start=pulse.t_start, trigger=True)

        if not (is_primary or trigger_pulses or targeted_pulse.t_start == 0):
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=interface.trigger_in_duration() * 1e-3,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))
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
            t_start=('>', self.t_start), t_stop=('<', self.t_stop),
            trigger=True)
        assert len(
            trigger_pulses) == 0, "Cannot implement DC ramp pulse if the " \
                                  "arbstudio receives intermediary triggers"

        t_list = {
        ch: np.arange(self.t_start, self.t_stop - self.final_delay * 1e-3,
                      1 / sampling_rates[ch] * 1e3) for ch in sampling_rates}

        # All waveforms must have an even number of points
        for ch in t_list:
            if len(t_list[ch]) % 2:
                t_list[ch] = t_list[ch][:-1]

        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception(
                "No implementation for connection {}".format(self.connection))

        waveforms = {ch: [self.get_voltage(t_list[ch])] for ch in channels}
        sequences = {ch: np.zeros(1, dtype=int) for ch in channels}
        return waveforms, sequences


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse
    # TODO add implement method

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(self, pulse,
            interface=interface, is_primary=is_primary, **kwargs)
        if not is_primary:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=interface.trigger_in_duration() * 1e-3,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))
        return targeted_pulse
