import numpy as np
import copy
import inspect
from blinker import signal

from .pulse_modules import PulseImplementation, PulseMatch

from qcodes import config

from silq.tools.general_tools import get_truth, SettingsClass

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
pulse_conditions = ['t_start', 't_stop', 'duration', 'acquire', 'initialize',
                    'connection', 'amplitude', 'enabled']

pulse_config = config['user'].get('pulses', {})
properties_config = config['user'].get('properties', {})

class Pulse(SettingsClass):
    # ncalls = 0
    # calls = []
    # t_start_inspect = []
    signal = signal('pulse')
    def __init__(self, name=None, t_start=None, t_stop=None,
                 duration=None, acquire=False, initialize=False,
                 connection=None, enabled=True, mode=None,
                 connection_requirements={}):
        # Dict of attrs that are connected via blinker.signal to other pulses
        self._connected_attrs = {}
        self.mode = mode

        self.name = name

        self.t_start = t_start
        self.duration = duration
        self.t_stop = t_stop

        self.acquire = acquire
        self.initialize = initialize
        self.enabled = enabled
        self.connection = connection

        # List of potential connection requirements.
        # These can be set so that the pulse can only be sent to connections
        # matching these requirements
        self.connection_requirements = connection_requirements


    def _matches_attrs(self, other_pulse, exclude_attrs=[]):
        for attr in list(vars(self)):
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
        # Pulse.ncalls += 1
        # Pulse.calls.append(item)
        # if item == 't_start':
        #     outer_fun = inspect.getouterframes(inspect.currentframe())
        #     Pulse.t_start_inspect += [outer_fun]
        value = object.__getattribute__(self, item)
        if value is not None:
            return value
        # Cannot obtain mode or mode_str, since they are called in
        # _attribute_from_config
        elif item not in ['mode', 'mode_str', 'name']:
            # Retrieve value from config
            value = self._attribute_from_config(item)
            return value

    def __setattr__(self, key, value):
        if isinstance(value, PulseMatch):
            super().__setattr__(key, value())
            set_fun = value.signal_function(self, key)
            self._connected_attrs[key] = set_fun
            self.signal.connect(set_fun, sender=value.pulse)
        else:
            super().__setattr__(key, value)

            if key in self._connected_attrs:
                # Remove function from pulse signal because it no longer
                # depends on other pulse
                self.signal.disconnect(self._connected_attrs.pop(key))

            if len(list(self.signal.receivers_for(self))):
                # send signal to anyone listening that attribute has changed
                self.signal.send(self, **{key: value})
                if key in ['t_start', 'duration']:
                    # Also send signal that dependent property t_stop has changed
                    self.signal.send(self, t_stop=self.t_stop)

    def _JSONEncoder(self):
        """
        Converts to JSON encoder for saving metadata

        Returns:
            JSON dict
        """
        return_dict = {}
        for attr, val in vars(self).items():
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
    def t_stop(self):
        if self.t_start is not None and self.duration is not None:
            return self.t_start + self.duration
        else:
            return None

    @t_stop.setter
    def t_stop(self, t_stop):
        if t_stop is not None:
            # Setting duration sends a signal for duration and also t_stop
            self.duration = t_stop - self.t_start

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
                properties_str  += '\n\t{}'.format(pulse_repr)

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
            self.frequency/1e6, self.power, self.t_start, self.t_stop)

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            "voltage at {} us is not in the time range {} ms - {} ms of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        return self.amplitude * np.sin(2 * np.pi * (self.frequency * t * 1e-3 +
                                                    self.phase/360))


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
            self.frequency/1e6, self.frequency_deviation/1e6,
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
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        if hasattr(t, '__len__'):
            return [self.amplitude] * len(t)
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
            "voltage at {} us is not in the time range {} us - {} us of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        slope = (self.amplitude_stop - self.amplitude_start) / self.duration
        offset = self.amplitude_start - slope * self.t_start

        return offset + slope * t


class TriggerPulse(Pulse):
    duration = .0001 # ms

    def __init__(self, name=None, duration=duration, **kwargs):
        super().__init__(name=name, duration=duration, **kwargs)

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


class MarkerPulse(Pulse):

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

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
