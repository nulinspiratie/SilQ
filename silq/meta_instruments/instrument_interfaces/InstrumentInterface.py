

class InstrumentInterface():
    def __init__(self, instrument):
        self.instrument = instrument

        self.input_channels = []
        self.output_channels = []


class Channel:
    def __init__(self, name, input=False, output=False, input_trigger=False,
                 output_trigger=False):
        self.name = name

        self.input = input
        self.output = output

        self.input_trigger = input_trigger
        self.output_trigger = output_trigger
