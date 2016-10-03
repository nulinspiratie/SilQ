from silq.pulses import PulseSequence

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class InstrumentInterface(Instrument):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(name=instrument_name + '_interface', **kwargs)
        self.instrument = self.find_instrument(name=instrument_name)

        self.input_channels = {}
        self.output_channels = {}

        self.channels = {}

        self.add_parameter('instrument_name',
                           parameter_class=ManualParameter,
                           initial_value=instrument_name,
                           vals=vals.Anything())
        self.add_parameter('pulse_sequence',
                           parameter_class=ManualParameter,
                           initial_value=PulseSequence(),
                           vals=vals.Anything())

        self.pulse_implementations = []

    def __repr__(self):
        return '{} interface'.format(self.name)

    def get_pulse_implementation(self, pulse):
        for pulse_implementation in self.pulse_implementations:
            if pulse_implementation.is_implementation(pulse):
                return pulse_implementation
        else:
            return None

    def add_pulse(self, pulse):
        """
        Add a pulse to self.pulse_sequence. Necessary since the interface can be a remote instrument.
        Args:
            pulse: Pulse to add

        Returns:
            None
        """
        self.pulse_sequence().add(pulse)

    def clear_pulses(self):
        """
       Clear all pulses from self.pulse_sequence. Necessary since the interface can be a remote instrument.
        Args:
            None
        Returns:
            None
        """
        self.pulse_sequence().clear()

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