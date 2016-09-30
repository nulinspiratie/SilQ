from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, PulseImplementation


class PulseBlasterESRPROInterface(InstrumentInterface):
    def __init__(self, instrument):
        super().__init__(instrument)