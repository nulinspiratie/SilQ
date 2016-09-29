import numpy as np

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


class Pulse:
    @classmethod
    def create_implementation(cls, pulse_implementation, pulse_conditions):
        return pulse_implementation(cls, pulse_conditions)

    def __init__(self, t_start, t_stop=None, duration=None,
                 connection=None):
        self.t_start = t_start

        if t_stop is not None:
            self.t_stop = t_stop
            self.duration = t_stop - t_start
        elif duration is not None:
            self.duration = duration
            self.t_stop = t_start + duration
        else:
            raise Exception("Must provide either t_stop or duration")

        self.connection = connection



class SinePulse(Pulse):
    def __init__(self, frequency, amplitude, **kwargs):
        super().__init__(**kwargs)

        self.frequency = frequency
        self.amplitude = amplitude

    def __repr__(self):
        return 'SinePulse(f={:.2f} MHz, A={}, t_start={}, t_stop={})'.format(
            self.frequency/1e6, self.amplitude, self.t_start, self.t_stop
        )


class DCPulse(Pulse):
    def __init__(self, amplitude, **kwargs):
        super().__init__(kwargs)

        self.amplitude = amplitude


class TriggerPulse(Pulse):
    def __init__(self, **kwargs):
        super().__init__(kwargs)