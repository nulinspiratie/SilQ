from silq.meta_instruments.instrument_interfaces \
    import InstrumentInterface, Channel


class ArbStudio1104_Interface(InstrumentInterface):
    def __init__(self, instrument):
        super().__init__(instrument)

        self.output_channels = [Channel(name='ch{}'.format(k), output=True)
                                for k in [1, 2, 3, 4]]
        self.trigger_in_channel = Channel(name='trig_in', input_trigger=True)
        self.trigger_out_channel = Channel(name='trig_out', output_trigger=True)
