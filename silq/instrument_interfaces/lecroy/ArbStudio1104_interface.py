import numpy as np

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, DCRampPulse, TriggerPulse, PulseImplementation

from qcodes.instrument.parameter import ManualParameter

class ArbStudio1104Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'ch{}'.format(k): Channel(instrument_name=self.name,
                                      name='ch{}'.format(k), id=k,
                                      output=True) for k in [1, 2, 3, 4]}
        self._trigger_in_channel = Channel(instrument_name=self.name,
                                          name='trig_in',
                                          input_trigger=True)
        self._trigger_out_channel = Channel(instrument_name=self.name,
                                           name='trig_out',
                                           output_TTL=(0, 3.3))
        # TODO check Arbstudio output TTL high

        self._channels = {**self._output_channels,
                         'trig_in': self._trigger_in_channel,
                         'trig_out': self._trigger_out_channel}

        self.pulse_implementations = [
            # TODO implement SinePulse
            # SinePulse(
            #     pulse_requirements=('frequency', {'min':1e6, 'max':50e6})
            # ),
            DCPulseImplementation(
                pulse_requirements=[('amplitude', {'min': -2.5, 'max': 2.5})]
            ),
            DCRampPulseImplementation(
                pulse_requirements=[('amplitude_start', {'min': -2.5, 'max': 2.5}),
                                    ('amplitude_stop',
                                     {'min': -2.5, 'max': 2.5})]
            ),
            TriggerPulseImplementation(
                pulse_requirements=[]
            )
        ]

        self.add_parameter('trigger_in_duration',
                           parameter_class=ManualParameter,
                           units='us',
                           initial_value=0.1)

        self.add_parameter('active_channels',
                           get_cmd=self._get_active_channels)

    def _get_active_channels(self):
        active_channels = [pulse.connection.output['channel'].name
                           for pulse in self.pulse_sequence()]
        # Transform into set to ensure that elements are unique
        active_channels = list(set(active_channels))
        return active_channels

    def setup(self, **kwargs):
        # TODO implement setup for modes other than stepped
        # Generate waveforms and sequences
        self.generate_waveforms()
        self.generate_sequences()

        for ch in self.active_channels():

            self.instrument.parameters[ch + '_trigger_source']('fp_trigger_in')
            self.instrument.parameters[ch + '_trigger_mode']('stepped')
            self.instrument.functions[ch + '_clear_waveforms']()

            # Add waveforms to channel
            for waveform in self.waveforms[ch]:
                self.instrument.functions[ch + '_add_waveform'](waveform)

            # Add sequence to channel
            self.instrument.parameters[ch + '_sequence'](self.sequences[ch])

        active_channels_id = [self._channels[channel].id
                              for channel in self.active_channels()]
        self.instrument.load_waveforms(channels=active_channels_id)
        self.instrument.load_sequence(channels=active_channels_id)

    def start(self):
        self.instrument.run(channels=[self._channels[channel].id
                                      for channel in self.active_channels()])

    def stop(self):
        self.instrument.stop()

    def get_final_additional_pulses(self):
        final_pulses = []

        # Loop over channels ensuring that all channels are programmed for each
        # trigger segment
        t = 0
        while t < self._pulse_sequence.duration:
            # Determine next moment in time
            t_next = min(t_val for t_val in self._pulse_sequence.t_list
                         if t_val > t)
            for channel in self.active_channels():
                # Check if there is a pulse that is active between t and t_next
                active_pulse = self._pulse_sequence.get_pulse(
                    t_start=('<=', t), t_stop=('>=', t_next),
                    output_channel=channel)

                if active_pulse is None:
                    # Add DC pulse at amplitude zero
                    dc_pulse = self.get_pulse_implementation(
                        DCPulse(t_start=t, t_stop=t_next, amplitude=0))
                    connection = self._pulse_sequence.get_connection(
                        output_channel=channel)
                    dc_pulse.connection = connection
                    self._pulse_sequence.add(dc_pulse)

                    # Check if trigger pulse is necessary. Only the case when
                    #  no trigger pulse already exists at the same time,
                    # when t > 0 (first trigger occurs at the end)
                    trigger_pulse = self.get_trigger_pulse(t)
                    if trigger_pulse not in self._input_pulse_sequence \
                            and t > 0 and trigger_pulse not in final_pulses:
                        final_pulses.append(trigger_pulse)

            t = t_next

        # Add a trigger pulse at the end if it does not yet exist
        trigger_pulse = self.get_trigger_pulse(self._pulse_sequence.duration)
        if trigger_pulse not in self._input_pulse_sequence \
                and trigger_pulse not in final_pulses:
            final_pulses.append(trigger_pulse)

        return final_pulses

    def get_trigger_pulse(self, t):
        trigger_pulse = TriggerPulse(
            t_start=t,
            duration=self.trigger_in_duration()*1e-3,
            connection_requirements={
               'input_instrument':
                   self.instrument_name(),
                'trigger': True}
            )
        return trigger_pulse


    def generate_waveforms(self):

        # Determine sampling rates
        sampling_rates = {ch:
            250e6 / self.instrument.parameters[ch+'_sampling_rate_prescaler']()
                          for ch in self.active_channels()}

        # Set time t_pulse to zero for each channel
        # This will increase as we iterate over pulses, and is used to ensure
        # that there are no times between pulses
        t_pulse = {ch: 0 for ch in self.active_channels()}

        self.waveforms = {ch: [] for ch in self.active_channels()}
        for pulse in self.pulse_sequence():

            channels_waveform = pulse.implement(sampling_rates=sampling_rates)

            for ch, waveform in channels_waveform.items():
                assert pulse.t_start == t_pulse[ch], \
                    "Pulse {}: pulses.t_start = {} does not match {}".format(
                        pulse, pulse.t_start, t_pulse[ch])

                self.waveforms[ch].append(waveform)
                # Increase t_pulse to match start of next pulses
                t_pulse[ch] += pulse.duration
        return self.waveforms

    def generate_sequences(self):
        self.sequences = {ch: list(range(len(self.waveforms[ch])))
                          for ch in self.active_channels()}
        return self.sequences


class DCPulseImplementation(PulseImplementation, DCPulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=DCPulse, **kwargs)

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, is_primary=is_primary, **kwargs)

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence().get_pulses(
            t_start=pulse.t_start, trigger=True
        )

        if not is_primary and not trigger_pulses and \
                not targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=interface.trigger_in_duration()*1e-3,
                             connection_requirements={
                                 'input_instrument':
                                     interface.instrument_name(),
                                 'trigger': True}
                             )
            )
        return targeted_pulse

    def implement(self, **kwargs):
        """
        Implements the DC pulses for the ArbStudio for SingleConnection and
        CombinedConnection. For a CombinedConnection, it weighs the DC pulses
        amplitude by the corresponding channel scaling factor (default 1).
        Args:

        Returns:
            {output_channel: pulses arr} dictionary for each output channel
        """
        # Arbstudio requires a minimum of four points to be returned
        if isinstance(self.connection, SingleConnection):
            return {self.connection.output['channel'].name:
                        [self.amplitude] * 4}
        elif isinstance(self.connection, CombinedConnection):
            return {ch.name: [self.amplitude] * 4
                    for ch in self.connection.output['channels']}
        else:
            raise Exception("No implementation for connection {}".format(
                self.connection))


class DCRampPulseImplementation(PulseImplementation, DCRampPulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=DCRampPulse, **kwargs)

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, is_primary=is_primary, **kwargs)

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence().get_pulses(
            t_start=pulse.t_start, trigger=True
        )

        if not is_primary and not trigger_pulses and \
                not targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=interface.trigger_in_duration()*1e-3,
                             connection_requirements={
                                 'input_instrument':
                                     interface.instrument_name(),
                                 'trigger': True}
                             )
            )
        return targeted_pulse

    def implement(self, sampling_rates, **kwargs):
        """
        Implements the DC pulses for the ArbStudio for SingleConnection and
        CombinedConnection. For a CombinedConnection, it weighs the DC pulses
        amplitude by the corresponding channel scaling factor (default 1).
        Args:

        Returns:
            {output_channel: pulses arr} dictionary for each output channel
        """
        # TODO Take sampling rate prescaler into account
        output = {}

        t_list = {ch: np.arange(self.t_start, self.t_stop,
                                1 / sampling_rates[ch] * 1e3)
                  for ch in
                  sampling_rates}

        # All waveforms must have an even number of points
        for ch in t_list:
            if len(t_list[ch]) % 2:
                t_list[ch] = t_list[ch][:-1]

        if isinstance(self.connection, SingleConnection):
            channel = self.connection.output['channel']
            signal = self.get_voltage(t_list[ch])
            output[channel.name] = signal
        elif isinstance(self.connection, CombinedConnection):
            for channel in self.connection.output['channels']:
                signal = self.get_voltage(t_list[ch])
                output[channel.name] = signal
        else:
            raise Exception("No implementation for connection {}".format(
                self.connection))
        return output

class TriggerPulseImplementation(TriggerPulse, PulseImplementation):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=TriggerPulse, **kwargs)

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, is_primary=is_primary, **kwargs)
        if not is_primary:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=interface.trigger_in_duration()*1e-3,
                             connection_requirements={
                                 'input_instrument':
                                     interface.instrument_name(),
                                 'trigger': True}
                             )
            )
        return targeted_pulse