from .mock_interface import MockInterface
from silq.instrument_interfaces import Channel
from silq.pulses import TriggerPulse, PulseImplementation


class MockTriggerInterface(MockInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        # Define instrument channels
        # - Two outputchannels (ch1 and ch2)
        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, input=True)
            for k in [1,2]
        }
        self._channels = self._output_channels

        self.pulse_implementations = [
            TriggerPulseImplementation(),
        ]

    def setup(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse