import numpy as np
import copy
from .pulse_modules import PulseImplementation

from qcodes import config

from silq.tools.general_tools import get_truth, SettingsClass

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
pulse_conditions = ['t_start', 't_stop', 'duration', 'acquire', 'initialize',
                    'connection', 'amplitude', 'enabled']

pulse_config = config['user'].get('pulses', {})
properties_config = config['user'].get('properties', {})


class Pulse(SettingsClass):
    def __init__(self, name=None, t_start=None, previous_pulse=None,
                 t_stop=None, delay_start=None, delay_stop=None,
                 duration=None, acquire=False, initialize=False,
                 connection=None, enabled=True, mode=None,
                 connection_requirements={}):
        self.mode = mode

        self.name = name
        self._previous_pulse = previous_pulse

        self._t_start = t_start
        self._duration = duration
        self._t_stop = t_stop
        self.delay_start = delay_start
        self.delay_stop = delay_stop

        self.acquire = acquire
        self.initialize = initialize
        self.enabled = enabled
        self.connection = connection

        # List of potential connection requirements.
        # These can be set so that the pulse can only be sent to connections
        # matching these requirements
        self.connection_requirements = connection_requirements

    def _matches_attrs(self, other_pulse, exclude_attrs=[]):
        # Add attrs that have a corresponding dependent property
        exclude_attrs += ['_t_start', '_t_stop', '_duration']

        for attr in list(vars(self)) + ['t_start', 't_stop']:
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

    def __getattribute__(self, item):
        """
        Used when requesting an attribute. If the attribute is explicitly set to
        None, it will check the config if the item exists.
        Args:
            item: Attribute to be retrieved

        Returns:

        """
        value = object.__getattribute__(self, item)
        if value is not None:
            return value
        # Cannot obtain mode or mode_str, since they are called in
        # _attribute_from_config
        elif item not in ['mode', 'mode_str', 'name']:
            # Retrieve value from config
            value = self._attribute_from_config(item)
            return value

    def __add__(self, other):
        """
        This method is called when adding two pulse instances by performing `pulse1 + pulse2`.

        Args:
            other (Pulse): The pulse instance to be added to self.

        Returns:
            combined_pulse (Pulse): A new pulse instance representing the combination of two pulses.

        """
        name = 'CombinationPulse_{}'.format(id(self)+id(other))
        return CombinationPulse(name, self, other, '+')

    def __radd__(self, other):
        """
        This method is called when reverse adding something to a pulse.
        The reason this method is implemented is so that the user can sum over multiple pulses by performing:

            combination_pulse = sum([pulse1, pulse2, pulse3])

        The sum method actually tries calling 0.__add__(pulse1), which doesn't exist, so it is converted into
        pulse1.__radd__(0).

        Args:
            other: an instance of unknown type that might be int(0)

        Returns:
            pulse (Pulse): Either self (if other is zero) or the sum of self and other.

        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        """
        This method is called when subtracting two pulse instances by performing `pulse1 - pulse2`.

        Args:
            other (Pulse): The pulse instance to be subtracted from self.

        Returns:
            combined_pulse (Pulse): A new pulse instance representing the combination of two pulses.

        """
        name = 'CombinationPulse_{}'.format(id(self)+id(other))
        return CombinationPulse(name, self, other, '-')

    def __mul__(self, other):
        """
        This method is called when multiplying two pulse instances by performing `pulse1 * pulse2`.

        Args:
            other (Pulse): The pulse instance to be multiplied with self.

        Returns:
            combined_pulse (Pulse): A new pulse instance representing the combination of two pulses.

        """
        name = 'CombinationPulse_{}'.format(id(self)+id(other))
        return CombinationPulse(name, self, other, '*')

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

    def _attribute_from_config(self, item):
        """
        Check if attribute exists somewhere in the config
        First, if pulse_config contains a key matching the pulse name,
        it will check for the attribute.
        If no success, it will check properties config if a key matches the item
        with self.mode appended. This is only checked if the pulse has a mode.
        Finally, it will check if properties_config contains the item
        """
        if self.name is not None and \
                        item in pulse_config.get(self.name + self.mode_str, {}):
            return pulse_config[self.name + self.mode_str][item]

        # Check if pulse attribute is in pulse_config
        if item in pulse_config.get(self.name, {}):
            return pulse_config[self.name][item]

        # check if {item}_{self.mode} is in properties_config
        # if mode is None, mode_str='', in which case it checks for {item}
        if (item + self.mode_str) in properties_config:
            return properties_config[item + self.mode_str]

        # Check if item is in properties config
        if item in properties_config:
            return properties_config[item]

        return None

    @property
    def t_start(self):
        if self._t_start is not None:
            return self._t_start
        elif self.previous_pulse is not None:
            if self.delay_start is not None:
                return self.previous_pulse.t_start + self.delay_start
            elif self.delay_stop is not None:
                return self.previous_pulse.t_stop + self.delay_stop
            else:
                return self.previous_pulse.t_stop
        else:
            # Check if item exists in config
            value = self._attribute_from_config('t_start')
            if value is not None:
                return value
            else:
                return 0

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
        elif self.t_start is not None and self.duration is not None:
            return self.t_start + self.duration
        else:
            return None

    @t_stop.setter
    def t_stop(self, t_stop):
        self._t_stop = t_stop

    @property
    def previous_pulse(self):
        if self._previous_pulse is None:
            return None
        elif self._previous_pulse.enabled:
            return self._previous_pulse
        else:
            return self._previous_pulse.previous_pulse

    @previous_pulse.setter
    def previous_pulse(self, previous_pulse):
        self._previous_pulse = previous_pulse

    @property
    def mode_str(self):
        return '' if self.mode is None else '_{}'.format(self.mode)

    def _get_repr(self, properties_str):
        pulse_name = self.name
        if self.mode is not None:
            pulse_name += ' ({})'.format(self.mode)
        if self.connection:
            properties_str += '\n\tconnection: {}'.format(self.connection)
        if self.connection_requirements:
            properties_str += '\n\trequirements: {}'.format(
                self.connection_requirements)
        if hasattr(self, 'additional_pulses') and self.additional_pulses:
            properties_str += '\n\tadditional_pulses:'
            for pulse in self.additional_pulses:
                pulse_repr = '\t'.join(repr(pulse).splitlines(True))
                properties_str += '\n\t{}'.format(pulse_repr)

        return '{pulse_type}({name}, {properties})'.format(
            pulse_type=self.__class__.__name__,
            name=pulse_name, properties=properties_str)

    def copy(self, fix_vars=False):
        """
        Creates a copy of a pulse.
        Args:
            fix_vars: If set to True, all of its vars are explicitly copied,
            ensuring that they are no longer linked to the settings

        Returns:
            Copy of pulse
        """
        pulse_copy = copy.deepcopy(self)
        if fix_vars:
            for var in vars(pulse_copy):
                setattr(pulse_copy, var, getattr(pulse_copy, var))

            # Also set dependent properties
            for var in ['t_start', 't_stop', 'previous_pulse']:
                setattr(pulse_copy, var, getattr(pulse_copy, var))
        return pulse_copy

    def satisfies_conditions(self, pulse_class=None,
                             t_start=None, t_stop=None, duration=None,
                             acquire=None, initialize=None, connection=None,
                             amplitude=None, enabled=None):
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
        if pulse_class is not None and not isinstance(self, pulse_class):
            return False

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


class SteeredInitialization(Pulse):
    def __init__(self, name=None, t_no_blip=None, t_max_wait=None,
                 t_buffer=None,
                 readout_threshold_voltage=None, **kwargs):
        super().__init__(name=name, t_start=0, duration=0, initialize=True,
                         **kwargs)

        self.t_no_blip = t_no_blip
        self.t_max_wait = t_max_wait
        self.t_buffer = t_buffer
        self.readout_threshold_voltage = readout_threshold_voltage

    def __repr__(self):
        properties_str = \
            't_no_blip={} ms, t_max_wait={}, t_buffer={}, V_th={}'.format(
                self.t_no_blip, self.t_max_wait, self.t_buffer,
                self.readout_threshold_voltage)
        return super()._get_repr(properties_str)


class SinePulse(Pulse):
    def __init__(self, name=None, frequency=None, phase=None,
                 power=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.frequency = frequency
        self.phase = phase
        self.power = power

        self.amplitude = power

    def __repr__(self):
        properties_str = 'f={:.2f} MHz, power={}, t_start={}, t_stop={}'.format(
            self.frequency / 1e6, self.power, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        return self.power * np.sin(2 * np.pi * (self.frequency * t + self.phase / 360))


class FrequencyRampPulse(Pulse):
    def __init__(self, name=None, frequency_start=None, frequency_stop=None,
                 frequency=None, frequency_deviation=None,
                 frequency_final='stop', amplitude=None, power=None,
                 frequency_sideband=None, **kwargs):
        super().__init__(name=name, **kwargs)

        if frequency_start is not None and frequency_stop is not None:
            self.frequency = (frequency_start + frequency_stop) / 2
            self.frequency_deviation = (frequency_stop - frequency_start)
        elif frequency is not None and frequency_deviation is not None:
            self.frequency = frequency
            self.frequency_deviation = frequency_deviation
        else:
            self.frequency = None
            self.frequency_deviation = None

        self._frequency_final = frequency_final
        self.frequency_sideband = frequency_sideband

        self.amplitude = amplitude
        self.power = power

    @property
    def frequency_start(self):
        return self.frequency - self.frequency_deviation

    @frequency_start.setter
    def frequency_start(self, frequency_start):
        frequency_stop = self.frequency_stop
        self.frequency = (frequency_start + frequency_stop) / 2
        self.frequency_deviation = (frequency_stop - frequency_start) / 2

    @property
    def frequency_stop(self):
        return self.frequency + self.frequency_deviation

    @frequency_stop.setter
    def frequency_stop(self, frequency_stop):
        frequency_start = self.frequency_start
        self.frequency = (frequency_start + frequency_stop) / 2
        self.frequency_deviation = (frequency_stop - frequency_start) / 2

    @property
    def frequency_final(self):
        if self._frequency_final == 'start':
            return self.frequency_start
        elif self._frequency_final == 'stop':
            return self.frequency_stop
        else:
            return self._frequency_final

    @frequency_final.setter
    def frequency_final(self, frequency_final):
        self._frequency_final = frequency_final

    def __repr__(self):
        properties_str = 'f_center={:.2f} MHz, f_dev={:.2f}, power={}, ' \
                         't_start={}, t_stop={}'.format(
            self.frequency / 1e6, self.frequency_deviation / 1e6,
            self.power, self.t_start, self.t_stop)

        if self.frequency_sideband is not None:
            properties_str += ', f_sb={}'.format(
                self.frequency_sideband)

        return super()._get_repr(properties_str)


class DCPulse(Pulse):
    def __init__(self, name=None, amplitude=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.amplitude = amplitude
        if self.amplitude is None:
            raise AttributeError("'{}' object has no attribute "
                                 "'amplitude'".format(self.__class__.__name__))

    def __repr__(self):
        properties_str = 'A={}, t_start={}, t_stop={}'.format(
            self.amplitude, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and  np.max(t) <= self.t_stop, \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        if hasattr(t, '__len__'):
            return np.ones(len(t))*self.amplitude
        else:
            return self.amplitude


class DCRampPulse(Pulse):
    def __init__(self, name=None, amplitude_start=None, amplitude_stop=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.amplitude_start = amplitude_start
        self.amplitude_stop = amplitude_stop

    def __repr__(self):
        properties_str = 'A_start={}, A_stop={}, t_start={}, t_stop={}'.format(
            self.amplitude_start, self.amplitude_stop, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert (self.t_start <= min(t)) and (max(t) <= self.t_stop), \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        slope = (self.amplitude_stop - self.amplitude_start) / self.duration
        offset = self.amplitude_start - slope * self.t_start

        return offset + slope * t


class TriggerPulse(Pulse):
    duration = .0001  # ms

    def __init__(self, name=None, duration=duration, **kwargs):
        self.duration = duration
        super().__init__(name=name, duration=duration, **kwargs)

    def __repr__(self):
        properties_str = 't_start={}, duration={}'.format(
            self.t_start, self.duration)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        if hasattr(t, '__len__'):
            return np.ones(len(t))*self.amplitude
        else:
            return self.amplitude


class MarkerPulse(Pulse):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def __repr__(self):
        properties_str = 't_start={}, duration={}'.format(
            self.t_start, self.duration)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= t <= self.t_stop, \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        return self.amplitude


class TriggerWaitPulse(Pulse):
    def __init__(self, name=None, t_start=None, **kwargs):
        super().__init__(name=name, t_start=t_start, duration=0, **kwargs)

    def __repr__(self):
        properties_str = 't_start={}'.format(
            self.t_start, self.duration)

        return super()._get_repr(properties_str)


class MeasurementPulse(Pulse):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def __repr__(self):
        properties_str = 't_start={}, duration={}'.format(
            self.t_start, self.duration)
        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        raise NotImplementedError('Measurement pulses do not have a voltage')


class CombinationPulse(Pulse):
    """
    This class represents pulses that are combinations of multiple pulse types.

    For example:
        CombinationPulse = SinePulse + DCPulse
        CombinationPulse = DCPulse * SinePulse

    Just like any other pulse, a CombinationPulse has a name, t_start and t_stop. t_start and t_stop are calculated and
    updated from the pulses that make up the combination.

    A CombinationPulse is itself a child of the Pulse class, therefore a CombinationPulse can also be used in
    consecutive combinations like:
        CombinationPulse1 = SinePulse1 + DCPulse
        CombinationPulse2 = SinePulse2 + CombinationPulse1

    Args:
        name (str): The name for this CombinationPulse.
        pulse1 (Pulse): The first pulse this combination is made up from.
        pulse2 (Pulse): The second pulse this combination is made up from.
        relation (str): The relation between pulse1 and pulse2. This must be one of the following:
            '+'     :   pulse1 + pulse2
            '-'     :   pulse1 - pulse2
            '*'     :   pulse1 * pulse2

    """

    def __init__(self, name=None, pulse1=None, pulse2=None, relation=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.pulse1 = pulse1
        self.pulse2 = pulse2
        self.relation = relation

        assert isinstance(pulse1, Pulse), 'pulse1 needs to be a Pulse'
        assert isinstance(pulse2, Pulse), 'pulse2 needs to be a Pulse'
        assert relation in ['+', '-', '*'], 'relation has a non-supported value'

    @property
    def t_start(self):
        return min(self.pulse1.t_start, self.pulse2.t_start)

    @t_start.setter
    def t_start(self, t_start):
        pass

    @property
    def t_stop(self):
        return max(self.pulse1.t_stop, self.pulse2.t_stop)

    @t_stop.setter
    def t_stop(self, t_stop):
        pass

    @property
    def combination_string(self):
        if isinstance(self.pulse1, CombinationPulse):
            pulse1_string = self.pulse1.combination_string
        else:
            pulse1_string = self.pulse1.name
        if isinstance(self.pulse2, CombinationPulse):
            pulse2_string = self.pulse2.combination_string
        else:
            pulse2_string = self.pulse2.name
        return '({pulse1} {relation} {pulse2})'.format(pulse1=pulse1_string,
                                                       relation=self.relation,
                                                       pulse2=pulse2_string)

    @property
    def pulse_details(self):
        if isinstance(self.pulse1, CombinationPulse):
            pulse1_details = self.pulse1.pulse_details
        else:
            pulse1_details = '\t {pulse} : {pulse_repr}\n'.format(pulse=self.pulse1.name,
                                                                  pulse_repr=repr(self.pulse1))
        if isinstance(self.pulse2, CombinationPulse):
            pulse2_details = self.pulse2.pulse_details
        else:
            pulse2_details = '\t {pulse} : {pulse_repr}\n'.format(pulse=self.pulse2.name,
                                                                  pulse_repr=repr(self.pulse2))
        return pulse1_details + pulse2_details

    def __repr__(self):
        return 'CombinationPulse of: {combination} with\n{details}'.format(combination=self.combination_string,
                                                                           details=self.pulse_details)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        result1 = np.zeros(t.shape[0])
        result2 = np.zeros(t.shape[0])

        pulse1_t = t[np.all([self.pulse1.t_start <= t, t <= self.pulse1.t_stop], axis=0)]
        pulse2_t = t[np.all([self.pulse2.t_start <= t, t <= self.pulse2.t_stop], axis=0)]

        voltage1 = self.pulse1.get_voltage(pulse1_t)
        voltage2 = self.pulse2.get_voltage(pulse2_t)

        result1[np.all([self.pulse1.t_start <= t, t <= self.pulse1.t_stop], axis=0)] = voltage1
        result2[np.all([self.pulse2.t_start <= t, t <= self.pulse2.t_stop], axis=0)] = voltage2

        if self.relation == '+':
            return result1 + result2
        elif self.relation == '-':
            return result1 - result2
        elif self.relation == '*':
            return result1 * result2


class AWGPulse(Pulse):
    """
    This class represents arbitrary waveform pulses that can be implemented by AWGs.

    This class allows the user to create a truly arbitrary pulse by either:
        - providing a function that converts time-stamps to waveform points
        - an array of waveform points

    The resulting AWGPulse can be sampled at different sample rates, interpolating between waveform points if necessary.

    Args:
        name (str): The name for this AWGPulse.
        fun (): The function used for calculating waveform points based on time-stamps.
        wf_array (np.array): Numpy array of (float) with time-stamps and waveform points.
        interpolate (bool): Flag for turning interpolation of the wf_array on (True) or off (False).

    """

    def __init__(self, name=None, fun=None, wf_array=None, interpolate=True, **kwargs):
        super().__init__(name=name, **kwargs)

        if fun:
            if not callable(fun):
                raise TypeError('The argument `function` must be a callable function.')
            self.from_function = True
            self.function = fun
        elif wf_array is not None:
            if not type(wf_array) == np.ndarray:
                raise TypeError('The argument `array` must be of type `np.ndarray`.')
            if not len(wf_array) == 2:
                raise TypeError('The argument `array` must be of length 2.')
            if not len(wf_array[0]) == len(wf_array[1]):
                raise TypeError('The argument `array` must have equal time-stamps and waveform points')
            assert np.all(np.diff(wf_array[0]) > 0), 'the time-stamps must be increasing'
            self.t_start = wf_array[0][0]
            self.t_stop = wf_array[0][-1]
            self.from_function = False
            self.array = wf_array
            self.interpolate = interpolate
        else:
            raise TypeError('Provide either a function or an array.')

    @classmethod
    def from_array(cls, array, **kwargs):
        return cls(wf_array=array, **kwargs)

    @classmethod
    def from_function(cls, function, **kwargs):
        return cls(fun=function, **kwargs)

    def __repr__(self):
        if self.from_function:
            properties_str = 'function:{}, t_start={}, t_stop={}'.format(self.function, self.t_start, self.t_stop)
        else:
            properties_str = 'array:{}, t_start={}, t_stop={}'.format(self.array.shape, self.t_start, self.t_stop)
        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        if self.from_function:
            return self.function(t)
        else:
            if self.interpolate:
                return np.interp(t, self.array[0], self.array[1])
            elif np.in1d(t, self.array[0]).all():
                mask = np.in1d(self.array[0], t)
                return self.array[1][mask]
            else:
                raise IndexError('All requested t-values must be in wf_array since interpolation is disabled.')
