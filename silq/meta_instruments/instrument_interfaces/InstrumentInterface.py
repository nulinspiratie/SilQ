import numpy as np
from silq.meta_instruments.PulseSequence import PulseSequence

class InstrumentInterface():
    def __init__(self, instrument):
        self.instrument = instrument

        self.input_channels = []
        self.output_channels = []

        # Connection with instrument that triggers this instrument
        self.trigger = None

        self.pulse_sequence = PulseSequence()

        self.pulse_implementations = []

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
    def __init__(self, name, input=False, output=False, input_trigger=False,
                 output_trigger=False):
        self.name = name

        self.input = input
        self.output = output

        self.input_trigger = input_trigger
        self.output_trigger = output_trigger
