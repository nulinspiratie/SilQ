from silq.meta_instruments.instrument_interfaces \
    import InstrumentInterface, Channel, PulseImplementation
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
            PulseImplementation(
                pulses.SinePulse,
                pulse_conditions=('frequency', {'min':1e6, 'max':50e6})
            ),
            PulseImplementation(
                pulses.DCPulse,
                pulse_conditions=('amplitude', {'min': 0, 'max': 2.5})
            ),
            PulseImplementation(
                pulses.TriggerPulse,
                pulse_conditions=[]
            )
        ]


class ArbStudio1104_PulseSequence(PulseSequence):
    def __init__(**kwargs):
        super().__init__(**kwargs)

    def add(self, pulse):
        pass