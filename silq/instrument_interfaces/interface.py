from silq.pulses import PulseSequence

class InstrumentInterface():
    def __init__(self, instrument):
        self.instrument = instrument
        self.name = instrument.name

        self.input_channels = {}
        self.output_channels = {}

        self.channels = {}

        # Connection with instrument that triggers this instrument
        self.trigger = None

        self.pulse_sequence = PulseSequence()

        self.pulse_implementations = []

    def __repr__(self):
        return '{} interface'.format(self.name)

    def get_pulse_implementation(self, pulse):
        for pulse_implementation in self.pulse_implementations:
            if pulse_implementation.is_implementation(pulse):
                return pulse_implementation
        else:
            return None

    def setup(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class Channel:
    def __init__(self, instrument, name, id=None, input=False, output=False,
                 input_trigger=False, output_trigger=False):
        self.instrument = instrument
        self.name = name
        self.id = id
        self.input = input
        self.output = output

        self.input_trigger = input_trigger
        self.output_trigger = output_trigger

    def __repr__(self):
        output_str = "Channel {name} (id={id})".format(name=self.name,
                                                       id=self.id)
        if self.input:
            output_str += ', input'
        if self.output:
            output_str += ', output'
        if self.input_trigger:
            output_str += ', input_trigger'
        if self.output_trigger:
            output_str += ', output_trigger'
        return output_str