import numpy as np


class PulseSequence:
    def __init__(self):
        self.pulses = []
        self.duration = 0

    def __getitem__(self, index):
        return self.pulses[index]

    def add(self, pulse):
        # TODO deal with case when pulses is a string (e.g. 'trigger')
        self.pulses.append(pulse)
        self.sort()

    def sort(self):
        t_start_list = np.array([pulse.t_start for pulse in self.pulses])
        idx_sorted = np.argsort(t_start_list)
        self.pulses = [self.pulses[idx] for idx in idx_sorted]

        # Update duration of PulseSequence
        self.duration = max([pulse.t_stop for pulse in self.pulses])
        return self.pulses

    def clear(self):
        self.pulses = []


class PulseImplementation():
    def __init__(self, pulse_class, pulse_conditions=[]):
        self.pulse_class = pulse_class
        self.pulse_conditions = [PulseCondition(self, property, condition) for
                                 (property, condition) in pulse_conditions]

    def __bool__(self):
        # Set truth value of a PulseImplementation to True, to make it easier
        # to use in list comprehensions.
        return True

    def add_pulse_condition(self, property, condition):
        self.pulse_conditions == [PulseCondition(self, property, condition)]

    def is_implementation(self, pulse):
        if pulse.__class__ != self.pulse_class:
            return False
        else:
            return np.all([pulse_condition.satisfies(pulse)
                           for pulse_condition in self.pulse_conditions])


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
        Checks if a given pulses satisfies this Pulsecondition
        Args:
            pulse: Pulse to be checked

        Returns: Bool depending on if the pulses satisfies PulseCondition

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
            raise Exception("Cannot interpret pulses condition: {}".format(
                self.condition))
