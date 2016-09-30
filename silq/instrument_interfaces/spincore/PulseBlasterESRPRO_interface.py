from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, PulseImplementation


class PulseBlasterESRPROInterface(InstrumentInterface):
    def __init__(self, instrument):
        super().__init__(instrument)

        self.output_channels = {
            'ch{}'.format(k): Channel(self, name='ch{}'.format(k), id=k,
                                      output_trigger=True)
            for k in [1, 2, 3, 4]}
        self.channels = {**self.output_channels}

        self.pulse_implementations = [
            TriggerPulse.create_implementation(
                TriggerPulseImplementation,
                pulse_conditions=[]
            )
        ]


class TriggerPulseImplementation(PulseImplementation):
    def __init__(self, pulse_class, **kwargs):
        super().__init__(pulse_class, **kwargs)

    def implement_pulse(self, trigger_pulse):
        pass