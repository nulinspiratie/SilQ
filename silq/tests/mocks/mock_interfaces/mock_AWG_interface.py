from .mock_interface import MockInterface
from silq.instrument_interfaces import Channel
from silq.pulses import DCPulse, SinePulse, PulseImplementation


class MockAWGInterface(MockInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        # Define instrument channels
        # - Two output channels (ch1 and ch2)
        # - One input trigger channel (trig_in)
        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, output=True)
            for k in [1,2]
        }
        self._channels = {
            **self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True)
        }

        self.pulse_implementations = [
            DCPulseImplementation(pulse_requirements=[('amplitude', {'min': -1,
                                                                     'max': 1})]),
            SinePulseImplementation(pulse_requirements=[('frequency', {'max': 500e6}),
                                                        ('amplitude', {'min': -1,
                                                                       'max': 1})])
        ]

    def setup(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

# Define pulse implementations

class DCPulseImplementation(PulseImplementation):
    pulse_class = DCPulse


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse
