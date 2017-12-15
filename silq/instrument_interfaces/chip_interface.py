from silq.instrument_interfaces.interface import InstrumentInterface, Channel
from silq.pulses.pulse_modules import  PulseImplementation
from silq.pulses.pulse_types import MeasurementPulse



class ChipInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'output': Channel(instrument_name=self.instrument_name(),
                              name='output',
                              output=True)}
        self._input_channels = {
            channel_name: Channel(instrument_name=self.instrument_name(),
                                  name=channel_name,
                                  input=True)
                               for channel_name in self.instrument.channels()}
        self._channels = {**self._input_channels, **self._output_channels}

        self.pulse_implementations = [
            MeasurementPulseImplementation(
                pulse_requirements=[]
            )
        ]

    def setup(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

class MeasurementPulseImplementation(PulseImplementation):
    pulse_class = MeasurementPulse


    def implement(self, instrument, sampling_rates, threshold):
        pass

