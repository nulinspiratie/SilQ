import numpy as np
import copy
from .pulse_modules import PulseImplementation

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

    def _matches_attrs(self, other_pulse, exclude_attrs=[]):
            for attr in vars(self):
                if attr in exclude_attrs:
                    continue
                elif not hasattr(other_pulse, attr) \
                        or getattr(self, attr) != getattr(other_pulse, attr):
                    return False
            else:
                return True

    def __eq__(self, other):
        """
        Overwrite comparison with other (self == other).
        We want the comparison to return True if other is a pulse with the
        same attributes. This can be complicated since pulses can also be
        targeted, resulting in a pulse implementation. We therefore have to
        use a separate comparison when either is a Pulse implementation
        Args:
            other:

        Returns:

        """
        exclude_attrs = ['connection', 'connection_requirements']
        if isinstance(self, PulseImplementation):
            if isinstance(other, PulseImplementation):
                # Both pulses are pulse implementations
                # Check if their pulse classes are the same
                if self.pulse_class != other.pulse_class:
                    return False
                # All attributes must match
                return self._matches_attrs(other)
            else:
                # Only self is a pulse implementation
                if not isinstance(other, self.pulse_class):
                    return False

                # self is a pulse implementation, and so it must match all
                # the attributes of other. The other way around does not
                # necessarily hold, since a pulse implementation has more attrs
                if not other._matches_attrs(self, exclude_attrs=exclude_attrs):
                    return False
                else:
                    # Check if self.connections satisfies the connection
                    # requirements of other
                    return self.connection.satisfies_conditions(
                        **other.connection_requirements)
        elif isinstance(other, PulseImplementation):
            # Only other is a pulse implementation
            if not isinstance(self, other.pulse_class):
                return False

            # other is a pulse implementation, and so it must match all
            # the attributes of self. The other way around does not
            # necessarily hold, since a pulse implementation has more attrs
            if not self._matches_attrs(other, exclude_attrs=exclude_attrs):
                return False
            else:
                # Check if other.connections satisfies the connection
                # requirements of self
                return other.connection.satisfies_conditions(
                    **self.connection_requirements)
        else:
            # Neither self nor other is a pulse implementation
            if not isinstance(other, self.__class__):
                return False
            # All attributes must match
            return self._matches_attrs(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # Define custom hash, used for creating a set of unique elements
        return hash(tuple(sorted(self.__dict__.items())))

    def __bool__(self):
        # Pulse is always equal to true
        return True

    def _JSONEncoder(self):
        """
        Converts to JSON encoder for saving metadata

        Returns:
            JSON dict
        """
        return_dict = {}
        for attr, val in vars(self).items():
            if not attr == 'previous_pulse':
                return_dict[attr] = val
        return return_dict


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
        properties_str = 'f={:.2f} MHz, A={:.13}, t_start={:.13}, ' \
                         't_stop={:.13}'.format(
                self.frequency/1e6, self.amplitude, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= t <= self.t_stop, \
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        return self.amplitude * np.sin(2 * np.pi * (self.frequency * t +
                                                    self.phase/360))


class FrequencyRampPulse(Pulse):
    def __init__(self, frequency_start=None, frequency_stop=None,
                 frequency_center=None, frequency_deviation=None,
                 frequency_final=None, amplitude=None, power=None, **kwargs):
        super().__init__(**kwargs)

        if frequency_start is not None and frequency_stop is not None:
            self.frequency_start = frequency_start
            self.frequency_stop = frequency_stop
        elif frequency_center is not None and frequency_deviation is not None:
            self.frequency_start = frequency_center - frequency_deviation / 2
            self.frequency_stop = frequency_center + frequency_deviation / 2
        else:
            raise SyntaxError("Must provide either f_start & f_stop or "
                              "f_center and f_deviation")

        if frequency_final is not None:
            self.frequency_final = frequency_final
        else:
            self.frequency_final = frequency_stop

        self.amplitude = amplitude
        self.power = power

    @property
    def frequency_center(self):
        return (self.frequency_start + self.frequency_stop) / 2

    @frequency_center.setter
    def frequency_center(self, frequency_center):
        frequency_deviation = self.frequency_deviation
        self.frequency_start = frequency_center - frequency_deviation / 2
        self.frequency_stop = frequency_center + frequency_deviation / 2

    @property
    def frequency_deviation(self):
        return abs(self.frequency_start - self.frequency_stop)

    @frequency_deviation.setter
    def frequency_deviation(self, frequency_deviation):
        frequency_center = self.frequency_center
        self.frequency_start = frequency_center - frequency_deviation / 2
        self.frequency_stop = frequency_center + frequency_deviation / 2

    def __repr__(self):
        properties_str = 'f_start={:.2f} MHz, f_stop={:.2f}, A={:.13}, ' \
                         't_start={:.13}, t_stop={:.13}'.format(
            self.frequency_start/1e6, self.frequency_stop/1e6, self.amplitude,
            self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    # def get_voltage(self, t):
    #     if t == self.t_start:
    #         return self.amplitude_start
    #     elif t == self.t_stop:
    #         return self.amplitude_final
    #     else:
    #         raise NotImplementedError("Voltage not yet implemented")


class DCPulse(Pulse):
    def __init__(self, amplitude, **kwargs):
        super().__init__(**kwargs)

        self.amplitude = amplitude

    def __repr__(self):
        properties_str = 'A={:.13}, t_start={:.13}, t_stop={:.13}'.format(
            self.amplitude, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= t <= self.t_stop, \
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        return self.amplitude

class DCRampPulse(Pulse):
    def __init__(self, amplitude_start, amplitude_stop,
                 **kwargs):
        super().__init__(**kwargs)

        self.amplitude_start = amplitude_start
        self.amplitude_stop = amplitude_stop

    def __repr__(self):
        properties_str = 'A_start={:.13}, A_stop={:.13}, t_start={:.13}, ' \
                         't_stop={:.13}'.format(
            self.amplitude_start, self.amplitude_stop, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert (self.t_start <= min(t)) and (max(t) <= self.t_stop), \
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        slope = (self.amplitude_stop - self.amplitude_start) / self.duration
        offset = self.amplitude_start - slope * self.t_start

        return offset + slope * t


class TriggerPulse(Pulse):
    duration = .0001 # ms

    def __init__(self, duration=duration, **kwargs):
        self.duration = duration
        super().__init__(duration=duration, **kwargs)

    def __repr__(self):
        properties_str = 't_start={:.13}, duration={:.13}'.format(
            self.t_start, self.duration)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= t <= self.t_stop, \
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        return self.amplitude


class MarkerPulse(Pulse):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        properties_str = 't_start={:.13}, duration={:.13}'.format(
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
        properties_str = 't_start={:.13}, duration={:.13}'.format(
            self.t_start, self.duration)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        raise NotImplementedError('Measurement pulses do not have a voltage')