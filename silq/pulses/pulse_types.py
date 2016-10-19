import numpy as np
import copy

class Pulse:
    @classmethod
    def create_implementation(cls, pulse_implementation, pulse_conditions):
        return pulse_implementation(cls, pulse_conditions=pulse_conditions)

    def __init__(self, name='', t_start=None, t_stop=None, duration=None,
                 connection=None, connection_requirements={}):
        self.name = name

        # TODO Allow t_start to not be given
        self.t_start = t_start

        self.t_stop = t_stop
        self.duration = duration

        self.connection = connection

        # List of potential connection requirements.
        # These can be set so that the pulse can only be sent to connections
        # matching these requirements
        self.connection_requirements = connection_requirements

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        # Pulse is always equal to true
        return True

    def copy(self):
        return copy.copy(self)



class SinePulse(Pulse):
    def __init__(self, frequency, amplitude, **kwargs):
        super().__init__(**kwargs)

        if self.t_stop is not None:
            self.duration = self.t_stop - self.t_start
        elif self.duration is not None:
            self.t_stop = self.t_start + self.duration
        else:
            raise Exception("Must provide either t_stop or duration")

        self.frequency = frequency
        self.amplitude = amplitude

    def __repr__(self):
        return 'SinePulse(f={:.2f} MHz, A={}, t_start={}, t_stop={})'.format(
            self.frequency/1e6, self.amplitude, self.t_start, self.t_stop
        )


class DCPulse(Pulse):
    def __init__(self, amplitude, **kwargs):
        super().__init__(**kwargs)
        if self.t_stop is not None:
            self.duration = self.t_stop - self.t_start
        elif self.duration is not None:
            self.t_stop = self.t_start + self.duration
        else:
            raise Exception("Must provide either t_stop or duration")

        self.amplitude = amplitude

    def __repr__(self):
        return 'DCPulse(A={}, t_start={}, t_stop={})'.format(
            self.amplitude, self.t_start, self.t_stop
        )


class TriggerPulse(Pulse):
    duration = 1 # us
    def __init__(self, duration=duration, **kwargs):
        super().__init__(**kwargs)

        self.duration = duration # us

    def __repr__(self):
        return 'TriggerPulse(t_start={}, duration={})'.format(
            self.t_start, self.duration
        )