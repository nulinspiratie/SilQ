from silq.meta_instruments.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.PulseSequence import PulseSequence
from silq.meta_instruments import pulses


class ArbStudio1104_Interface(InstrumentInterface):
    def __init__(self, instrument):
        super().__init__(instrument)

        self.output_channels = [Channel(name='ch{}'.format(k), output=True)
                                for k in [1, 2, 3, 4]]
        self.trigger_in_channel = Channel(name='trig_in', input_trigger=True)
        self.trigger_out_channel = Channel(name='trig_out', output_trigger=True)

        self.pulse_implementations = [
            # TODO implement SinePulse
            # pulses.SinePulse.create_implementation(
            #     pulse_conditions=('frequency', {'min':1e6, 'max':50e6})
            # ),
            pulses.DCPulse.create_implementation(
                pulse_conditions=('amplitude', {'min': 0, 'max': 2.5})
            ),
            pulses.TriggerPulse.create_implementation(
                pulse_conditions=[]
            )
        ]

    def setup(self):
        # TODO implement setup for modes other than stepped
        self.active_channels = set([pulse.connection.output_channel
                        for pulse in self.pulse_sequence])

        # Find sampling rates (these may be different for different channels)
        sampling_rates = [self.instrument.sampling_rate /
                          eval("self.instrument.ch{}_sampling_rate_prescaler()"
                               "".format(ch)) for ch in self.active_channels]

        for ch in self.active_channels:
            eval("self.instrument.ch{}_trigger_source('fp_trigger_in')".format(
                ch))
            eval("self.instrument.ch{}_trigger_mode('stepped')".format(ch))
            eval('self.instrument.ch{}_clear_waveforms()'.format(ch))

    def generate_waveforms(self):
        # Set time t_pulse to zero, will increase as we iterate over pulses
        t_pulse = 0

        for pulse in self.pulse_sequence:
            assert pulse.t_start == t_pulse, \
                "Pulse {}: pulse.t_start = {} does not match {}".format(
                    pulse, pulse.t_start, t_pulse)

            pulse_implementation = self.get_pulse_implementation(pulse)


            # Increase t_pulse to match start of next pulse
            t_pulse += pulse.duration


class DCPulseImplementation(pulses.PulseImplementation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def implement_pulse(self, DC_pulse):
