from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection


class ChipInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self.output_channels = {
            'output': Channel(instrument_name=self.name,
                              name='output',
                              output=True)}
        self.input_channels = {
            channel_name: Channel(instrument_name=self.name,
                                  name=channel_name,
                                  input=True)
                               for channel_name in ['TGAC', 'DF']}
        self.channels = {**self.input_channels, **self.output_channels}

    def setup(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def get_final_additional_pulses(self):
        return []
