import numpy as np
import copy

from silq.tools.general_tools import get_truth

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
pulse_conditions = ['t_start', 't_stop', 'duration', 'acquire', 'connection',
                    'amplitude']

class Pulse:
    def __init__(self, name='', t_start=None, previous_pulse=None,
                 t_stop=None, duration=None,
                 acquire=False, connection=None, connection_requirements={}):
        self.name = name

        self._t_start = t_start
        self._duration = duration
        self._t_stop = t_stop

        if duration is None and t_stop is None:
            raise Exception("Must provide either t_stop or duration")

        self.previous_pulse = previous_pulse

        self.acquire = acquire
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

    @property
    def t_start(self):
        if self._t_start is not None:
            return self._t_start
        elif self.previous_pulse is not None:
            return self.previous_pulse.t_stop
        else:
            return None

    @t_start.setter
    def t_start(self, t_start):
        self._t_start = t_start

    @property
    def duration(self):
        if self._t_stop is not None:
            if self.t_start is not None:
                return self._t_stop - self.t_start
            else:
                return None
        else:
            return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    @property
    def t_stop(self):
        if self._t_stop is not None:
            return self._t_stop
        elif self.t_start is not None:
            return self.t_start + self._duration
        else:
            return None

    @t_stop.setter
    def t_stop(self, t_stop):
        self._t_stop = t_stop

    def _get_repr(self, properties_str):
        # properties_str = ', '.join(['{}: {}'.format(prop, getattr(self, prop))
        #                            for prop in properties])

        if self.connection:
            properties_str += '\n\tconnection: {}'.format(self.connection)
        if self.connection_requirements:
            properties_str += '\n\trequirements: {}'.format(
                self.connection_requirements)
        if hasattr(self, 'additional_pulses') and self.additional_pulses:
            properties_str += '\n\tadditional_pulses:'
            for pulse in self.additional_pulses:
                pulse_repr = '\t'.join(repr(pulse).splitlines(True))
                properties_str  += '\n\t{}'.format(pulse_repr)

        return '{pulse_type}({properties})'.format(
            pulse_type=self.__class__.__name__, properties=properties_str)

    def copy(self):
        pulse_copy = copy.deepcopy(self)
        return pulse_copy

    def satisfies_conditions(self, t_start=None, t_stop=None, duration=None,
                             acquire=None, connection=None, amplitude=None):
        """
        Checks if pulse satisfies certain conditions.
        Each kwarg is a condition, and can be a value (equality testing) or it
        can be a tuple (relation, value), in which case the relation is tested.
        Possible relations: '>', '<', '>=', '<=', '=='
        Args:
            t_start:
            t_stop:
            duration:
            acquire:
            connection:
            **kwargs:

        Returns:
            Bool depending on if all conditions are satisfied.
        """

        for property in pulse_conditions:

            # Get arg value from property name
            val = eval(property)

            if val is None:
                continue
            elif not hasattr(self, property):
                return False

            # If the arg is a tuple, the first element specifies its relation
            if isinstance(val, (list, tuple)):
                relation, val = val
            else:
                relation = '=='
            if not get_truth(test_val=getattr(self, property), target_val=val,
                             relation=relation):
                return False
        else:
            return True

    def get_voltage(self, t):
        raise NotImplementedError(
            'This method should be implemented in a subclass')


class SinePulse(Pulse):
    def __init__(self, frequency, amplitude, phase=0, **kwargs):
        super().__init__(**kwargs)

        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def __repr__(self):
        properties_str = 'f={:.2f} MHz, A={}, t_start={}, t_stop={}'.format(
            self.frequency/1e6, self.amplitude, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= t <= self.t_stop, \
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        return self.amplitude * np.sin(2 * np.pi * (self.frequency * t +
                                                    self.phase/360))

class DCPulse(Pulse):
    def __init__(self, amplitude, **kwargs):
        super().__init__(**kwargs)

        self.amplitude = amplitude

    def __repr__(self):
        properties_str = 'A={}, t_start={}, t_stop={}'.format(
            self.amplitude, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= t <= self.t_stop, \
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        return self.amplitude


class TriggerPulse(Pulse):
    duration = .1 # us

    def __init__(self, duration=duration, **kwargs):
        super().__init__(duration=duration, **kwargs)

    def __repr__(self):
        properties_str = 't_start={}, duration={}'.format(
            self.t_start, self.duration)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= t <= self.t_stop, \
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        return self.amplitude


class MeasurementPulse(Pulse):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        properties_str = 't_start={}, duration={}'.format(
            self.t_start, self.duration)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        raise NotImplementedError('Measurement pulses do not have a voltage')