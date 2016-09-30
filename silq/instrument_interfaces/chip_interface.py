from silq.instrument_interfaces \
    import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection


class ChipInterface(InstrumentInterface):
    def __init__(self, instrument):
        super().__init__(instrument)

        self.output_channels = {
            'source_drain': Channel(self, name='source_drain', output=True)}
        self.input_channels = {
            channel_name: Channel(self, name=channel_name, input=True)
                               for channel_name in ['TGAC', 'DF']}
        self.channels = {**self.input_channels, **self.output_channels}