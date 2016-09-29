from silq.meta_instruments.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, PulseImplementation


class ArbStudio1104_Interface(InstrumentInterface):
    def __init__(self, instrument):
        super().__init__(instrument)

        self.output_channels = [Channel(self, name='ch{}'.format(k), id=k,
                                        output=True) for k in [1, 2, 3, 4]]
        self.trigger_in_channel = Channel(self, name='trig_in',
                                          input_trigger=True)
        self.trigger_out_channel = Channel(self, name='trig_out',
                                           output_trigger=True)

        self.pulse_implementations = [
            # TODO implement SinePulse
            # pulses.SinePulse.create_implementation(
            #     pulse_conditions=('frequency', {'min':1e6, 'max':50e6})
            # ),
            DCPulse.create_implementation(
                DCPulseImplementation,
                pulse_conditions=[('amplitude', {'min': 0, 'max': 2.5})]
            ),
            TriggerPulse.create_implementation(
                TriggerPulseImplementation,
                pulse_conditions=[]
            )
        ]

    def setup(self):
        # TODO implement setup for modes other than stepped
        # Transform into set to ensure that elements are unique
        self.active_channels = set([pulse.connection.output_channel
                        for pulse in self.pulse_sequence])

        # # Find sampling rates (these may be different for different channels)
        # sampling_rates = [self.instrument.sampling_rate /
        #                   eval("self.instrument.ch{}_sampling_rate_prescaler()"
        #                        "".format(ch)) for ch in self.active_channels]

        # Generate waveforms and sequences
        self.generate_waveforms()
        self.generate_sequences()

        for channel in self.active_channels:

            eval("self.instrument.{ch}_trigger_source('fp_trigger_in')".format(
                ch=channel.name))
            eval("self.instrument.{ch}_trigger_mode('stepped')".format(
                ch=channel.name))
            eval('self.instrument.{ch}_clear_waveforms()'.format(
                ch=channel.name))

            # Add waveforms to channel
            for waveform in self.waveforms[channel.name]:
                eval('self.instrument.{ch}_add_waveform({waveform}'.format(
                    ch=channel.name, waveform=self.waveform))

            # Add sequence to channel
            eval('self.instrument.{ch}_sequence({sequence}'.format(
                ch=channel.name, sequence=self.sequences[channel.name]))

        active_channels_id = [channel.id for channel in self.active_channels]
        self.instrument.load_waveforms(channels=active_channels_id)
        self.instrument.load_sequence(channels=active_channels_id)

    def generate_waveforms(self):
        # Set time t_pulse to zero, will increase as we iterate over pulses
        t_pulse = 0

        self.waveforms = {ch: [] for ch in self.active_channels}
        for pulse in self.pulse_sequence:
            assert pulse.t_start == t_pulse, \
                "Pulse {}: pulses.t_start = {} does not match {}".format(
                    pulse, pulse.t_start, t_pulse)

            pulse_implementation = self.get_pulse_implementation(pulse)
            channels_waveform = pulse_implementation.implement_pulse(pulse)

            for ch in self.active_channels:
                self.waveforms[ch].append(channels_waveform[ch])

            # Increase t_pulse to match start of next pulses
            t_pulse += pulse.duration
        return self.waveforms

    def generate_sequences(self):
        self.sequences = {ch: range(len(self.waveforms[ch]))
                          for ch in self.active_channels}
        return self.sequences



class DCPulseImplementation(PulseImplementation):
    def __init__(self, pulse_class, **kwargs):
        super().__init__(pulse_class, **kwargs)

    def implement_pulse(self, DC_pulse):
        """
        Implements the DC pulses for the ArbStudio for SingleConnection and
        CombinedConnection. For a CombinedConnection, it weighs the DC pulses
        amplitude by the corresponding channel scaling factor (default 1).
        Args:
            DC_pulse: DC pulses to implement

        Returns:
            {output_channel: pulses arr} dictionary for each output channel
        """
        # Arbstudio requires a minimum of four points to be returned
        if isinstance(DC_pulse.connection, SingleConnection):
            return {DC_pulse.connection.output['channel']:
                        [DC_pulse.amplitude] * 4}
        elif isinstance(DC_pulse.connection, CombinedConnection):
            return {ch: [DC_pulse.amplitude] * 4
                    for ch in DC_pulse.connection.output['channels']}
        else:
            raise Exception("No implementation for connection {}".format(
                DC_pulse.connection))


class TriggerPulseImplementation(PulseImplementation):
    def __init__(self, pulse_class, **kwargs):
        super().__init__(pulse_class, **kwargs)

    def implement_pulse(self, trigger_pulse):
        if isinstance(trigger_pulse.connection, SingleConnection):
            pass
        elif isinstance(trigger_pulse.connection, CombinedConnection):
            pass