from silq.meta_instruments.PulseSequence import PulseSequence

class InstrumentInterface():
    def __init__(self, instrument):
        self.instrument = instrument

        self.input_channels = []
        self.output_channels = []

        # Connection with instrument that triggers this instrument
        self.trigger = None

        self.pulse_sequence = PulseSequence()

        self.implementations = []


class PulseImplementation():
    def __init__(self, pulse_class, pulse_conditions=[]):
        self.pulse_class = pulse_class
        self.pulse_conditions = [PulseCondition(self, property, condition) for
                                 (property, condition) in pulse_conditions]

    def add_pulse_condition(self, property, condition):
        self.pulse_conditions == [PulseCondition(self, property, condition)]


class PulseCondition():
    def __init__(self, pulse_class, property, condition):
        self.property = property

        self.verify_condition(condition)
        self.condition = condition

    def verify_condition(self, condition):
        if type(condition) is list:
            assert condition, "Condition must not be an empty list"
        elif type(condition) is dict:
            assert ('min' in condition or 'max' in condition), \
                "Dictionary condition must have either a 'min' or a 'max'"

    def satisfies(self, pulse):
        """
        Checks if a given pulse satisfies this Pulsecondition
        Args:
            pulse: Pulse to be checked

        Returns: Bool depending on if the pulse satisfies PulseCondition

        """
        property_value = getattr(pulse, self.property)

        # Test for condition
        if type(self.condition) is dict:
            # condition contains min and/or max
            if 'min' in self.condition and \
                            property_value < self.condition['min']:
                return False
            elif 'max' in self.condition and \
                            property_value > self.condition['max']:
                return False
            else:
                return True
        elif type(self.condition) is list:
            if property_value not in self.condition:
                return False
            else:
                return True
        else:
            raise Exception("Cannot interpret pulse condition: {}".format(
                self.condition))


class Channel:
    def __init__(self, name, input=False, output=False, input_trigger=False,
                 output_trigger=False):
        self.name = name

        self.input = input
        self.output = output

        self.input_trigger = input_trigger
        self.output_trigger = output_trigger
