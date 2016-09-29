from silq.meta_instruments.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.PulseSequence import PulseSequence
from silq.meta_instruments import pulses, layout


class ArbStudio1104_Interface(InstrumentInterface):
    def __init__(self, instrument):
        super().__init__(instrument)

        self.output_channels = [Channel(self, name='ch{}'.format(k),
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
            pulses.DCPulse.create_implementation(DCPulseImplementation,
                pulse_conditions=('amplitude', {'min': 0, 'max': 2.5})
            ),
            pulses.TriggerPulse.create_implementation(
                pulse_conditions=[]
            )
        ]

    def setup(self):
        # TODO implement setup for modes other than stepped
        # Transform into set to ensure that elements are unique
        self.active_channels = set([pulse.connection.output_channel
                        for pulse in self.pulse_sequence])

        # Find sampling rates (these may be different for different channels)
        sampling_rates = [self.instrument.sampling_rate /
                          eval("self.instrument.ch{}_sampling_rate_prescaler()"
                               "".format(ch)) for ch in self.active_channels]

        self.generate_waveforms()
        self.generate_sequences()

        for ch in self.active_channels:
            eval("self.instrument.{ch}_trigger_source('fp_trigger_in')".format(
                ch=ch))
            eval("self.instrument.{ch}_trigger_mode('stepped')".format(ch=ch))
            eval('self.instrument.{ch}_clear_waveforms()'.format(ch=ch))
            # Add waveforms to channel
            for waveform in self.waveforms[ch]:
                eval('self.instrument.{ch}_add_waveform({waveform}'.format(
                    ch=ch, waveform=self.waveform))
            # Add sequence to channel
            eval('self.instrument.{ch}_sequence({sequence}'.format(
                ch=ch, sequence=self.sequences[ch]))
        self.instrument.load_waveforms(channels=self.arbstudio_channels())
        self.instrument.load_sequence(channels=self.arbstudio_channels())

    def generate_waveforms(self):
        # Set time t_pulse to zero, will increase as we iterate over pulses
        t_pulse = 0

        self.waveforms = {ch: [] for ch in self.active_channels}
        for pulse in self.pulse_sequence:
            assert pulse.t_start == t_pulse, \
                "Pulse {}: pulse.t_start = {} does not match {}".format(
                    pulse, pulse.t_start, t_pulse)

            pulse_implementation = self.get_pulse_implementation(pulse)
            channels_waveform = pulse_implementation.implement_pulse(pulse)

            for ch in self.active_channels:
                self.waveforms[ch].append(channels_waveform[ch])

            # Increase t_pulse to match start of next pulse
            t_pulse += pulse.duration
        return self.waveforms

    def generate_sequences(self):
        self.sequences = {ch: range(len(self.waveforms[ch]))
                          for ch in self.active_channels}
        return self.sequences



class DCPulseImplementation(pulses.PulseImplementation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def implement_pulse(self, DC_pulse):
        """
        Implements the DC pulse for the ArbStudio for SingleConnection and
        CombinedConnection. For a CombinedConnection, it weighs the DC pulse
        amplitude by the corresponding channel scaling factor (default 1).
        Args:
            DC_pulse: DC pulse to implement

        Returns:
            {output_channel: pulse arr} dictionary for each output channel
        """
        # Arbstudio requires a minimum of four points to be returned
        if isinstance(DC_pulse.connection, layout.SingleConnection):
            return {DC_pulse.connection.output['channel']:
                        [DC_pulse.amplitude] * 4}
        elif isinstance(DC_pulse.connection, layout.CombinedConnection):
            return {ch: [DC_pulse.amplitude] * 4
                    for ch in DC_pulse.connection.output['channels']}
        else:
            raise Exception("No implementation for connection {}".format(
                DC_pulse.connection))


class TriggerPulseImplementation(pulses.PulseImplementation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def implement_pulse(self, trigger_pulse):
        if isinstance(trigger_pulse.connection, layout.SingleConnection):
            pass
        elif isinstance(trigger_pulse.connection, layout.CombinedConnection):
            pass