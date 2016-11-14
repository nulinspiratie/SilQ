from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection


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
                               for channel_name in ['TGAC', 'DF', 'ESR']}
        self._channels = {**self._input_channels, **self._output_channels}

    def setup(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def get_final_additional_pulses(self, **kwargs):
        return []
