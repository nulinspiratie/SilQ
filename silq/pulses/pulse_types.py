import numpy as np
from copy import deepcopy
import collections
from traitlets import HasTraits, Unicode, validate, TraitError
from blinker import Signal, signal
import logging
from functools import partial

from .pulse_modules import PulseMatch

from silq.tools.general_tools import get_truth, property_ignore_setter, freq_to_str
from silq import config

__all__ = ['Pulse', 'SteeredInitialization', 'SinePulse', 'FrequencyRampPulse',
           'DCPulse', 'DCRampPulse', 'TriggerPulse', 'MarkerPulse',
           'TriggerWaitPulse', 'MeasurementPulse', 'CombinationPulse',
           'AWGPulse', 'pulse_conditions']

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
pulse_conditions = ['name', 'id', 'environment', 't', 't_start', 't_stop',
                    'duration', 'acquire', 'initialize', 'connection',
                    'amplitude', 'enabled', 'average']

Signal.__deepcopy__ = lambda self, memo: Signal()
logger = logging.getLogger(__name__)


class Pulse(HasTraits):
    average = Unicode()
    signal = Signal()
    _connected_attrs = {}

    def __init__(self, name=None, id=None, environment='default', t_start=None,
                 t_stop=None, duration=None, acquire=False, initialize=False,
                 connection=None, enabled=True, average='none',
                 connection_label=None, connection_requirements={}):
        # Initialize signals (to notify change of attributes)
        self.signal = Signal()
        # Dict of attrs that are connected via blinker.signal to other pulses
        self._connected_attrs = {}
        super().__init__()

        self.name = name
        self.id = id

        if environment == 'default':
            environment = config.properties.get('default_environment',
                                                'default')
        self.environment = environment

        # Setup pulse config
        try:
            # Set pulse_config from SilQ environment config
            self.pulse_config = config[self.environment].pulses[self.name]
        except KeyError:
            self.pulse_config = None
        try:
            # Set properties_config from SilQ environment config
            self.properties_config = config[self.environment].properties
        except (KeyError, AttributeError):
            self.properties_config = None

        ### Setup signals
        # Connect changes in pulse config to handling method
        # If environment.pulses has no self.name key, this will never be called.
        signal(f'config:{self.environment}.pulses.{self.name}').connect(
            self._handle_config_signal)

        # Setup properties config. If pulse requires additional
        # properties_attrs, place them before calling Pulse.__init__,
        # else they are not added to attrs.
        # Make sure that self.properties_attrs is never replaced, only appended.
        # Else it is no longer used for self._handle_properties_config_signal.
        if not hasattr(self, 'properties_attrs'):
            # Create attr if it does not already exist
            self.properties_attrs = []
        self.properties_attrs += ['t_read', 't_skip']

        # Connect changes in properties config to handling method
        # If environment has no properties key, this will never be called.
        signal(f'config:{self.environment}.properties').connect(
            self._handle_properties_config_signal)


        ### Set attributes
        # Set attributes that can also be retrieved from pulse_config
        self.t_start = self._value_or_config('t_start', t_start)
        self.duration = self._value_or_config('duration', duration)
        self.t_stop = self._value_or_config('t_stop', t_stop)
        self.connection_label = self._value_or_config('connection_label',
                                                      connection_label)

        # Set attributes that can also be retrieved from properties_config
        if self.properties_config is not None:
            for attr in self.properties_attrs:
                setattr(self, attr, self.properties_config.get(attr, None))

        # Set attributes that should not be retrieved from pulse_config
        self.acquire = acquire
        self.initialize = initialize
        self.enabled = enabled
        self.connection = connection
        self.average = average

        # Pulses can have a PulseImplementation after targeting
        self.implementation = None

        # List of potential connection requirements.
        # These can be set so that the pulse can only be sent to connections
        # matching these requirements
        self.connection_requirements = connection_requirements

    @validate('average')
    def _valid_average(self, proposal):
        if proposal['value'] in ['none', 'trace', 'point']:
            return proposal['value']
        elif ('point_segment' in proposal['value'] or
                      'trace_segment' in proposal['value']):
            return proposal['value']
        else:
            return TraitError

    def _matches_attrs(self, other_pulse, exclude_attrs=[]):
        for attr in list(vars(self)):
            if attr in exclude_attrs:
                continue
            elif not hasattr(other_pulse, attr) \
                    or getattr(self, attr) != getattr(other_pulse, attr):
                return False
        else:
            return True

    def _handle_config_signal(self, _, select=None, **kwargs):
        """
        Update attr when attr in pulse config is modified
        Args:
            _: sender config (unused)
            select (Optional(List(str): list of attrs that can be set.
                Will update any attribute if not specified.
            **kwargs: {attr: new_val}

        Returns:

        """
        key, val = kwargs.popitem()
        if select is None or key in select:
            setattr(self, key, val)

    def _handle_properties_config_signal(self, arg, **kwargs):
        """ Update attr when attr in properties config is modified.

        Note:
            This method has to be defined separately, and cannot simply be
            defined using a partial on `_handle_config_signal`, as this will
            somehow cause it to always reference itself, and thus never be gc'ed

        Args:
            arg: Ignored handle arg passed by signal.send.
            **kwargs: handle kwargs

        """
        self._handle_config_signal(arg, select=self.properties_attrs, **kwargs)

    def __str__(self):
        # This is called by blinker.signal to get a repr. Instead of creating
        # a full repr which requires several attrs, this is much faster.
        pulse_class = self.__class__.__name__
        return f'{pulse_class}({self.full_name})'

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
        exclude_attrs = ['connection', 'connection_requirements', 'signal',
                         '_handle_properties_config_signal', '_connected_attrs',
                         'properties_config', 'pulse_config']

        if not isinstance(other, self.__class__):
            return False

        if self.implementation is not None:
            if other.implementation is not None:
                # Both pulses have pulse implementations
                # All attributes must match
                return self._matches_attrs(other, exclude_attrs=exclude_attrs)
            else:
                # Only self has a pulse implementation

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
        elif other.implementation is not None:
            # Only other has a pulse implementation

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
            # Neither self nor other has a pulse implementation
            # All attributes must match
            return self._matches_attrs(other, exclude_attrs=exclude_attrs)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # Define custom hash, used for creating a set of unique elements
        return hash(tuple(sorted(self.__dict__.items())))

    def __bool__(self):
        # Pulse is always equal to true
        return True

    def __setattr__(self, key, value):
        if isinstance(value, PulseMatch):
            previous_pulse_match = self._connected_attrs.get(key, None)
            if previous_pulse_match is not value:
                # Either no previous pulse, or it was different from value
                if isinstance(previous_pulse_match, PulseMatch):
                    # Disconnect previous PulseMatch
                    previous_pulse_match.origin_pulse.signal.disconnect(
                        previous_pulse_match)

            value.origin_pulse.signal.connect(value)
            value.target_pulse = self
            value.target_pulse_attr = key
            self._connected_attrs[key] = value

            super().__setattr__(key, value.value)

        else:
            if key == 'environment' and hasattr(self, key):
                # Disconnect previous handlers if they existed
                signal(f'config:{self.environment}.pulses.{self.name}'
                       ).disconnect(self._handle_config_signal)
                signal(f'config:{self.environment}.properties'
                       ).disconnect(self._handle_properties_config_signal)

                # Connect to new handlers
                signal(f'config:{value}.pulses.{self.name}').connect(
                    self._handle_config_signal)
                signal(f'config:{value}.properties').connect(
                    self._handle_properties_config_signal)

                if self.name in config[value].pulses:
                    # Replace pulse_config
                    self.pulse_config = config[value].pulses[self.name]
                    # Update all pulse attrs that exist in new pulse_config
                    for env_key, env_val in self.pulse_config.items():
                        if hasattr(self, env_key):
                            setattr(self, env_key, env_val)
                else:
                    self.pulse_config = None

                if 'properties' in config[value]:
                    # Repace properties_config
                    self.properties_attrs = config[value].properties
                    # Replace all attrs in new properties_config if they are
                    # in self.properties_attrs
                    for attr in self.properties_attrs:
                        if attr in config[value].properties:
                            setattr(self, attr, config[value].properties[attr])

            super().__setattr__(key, value)


            if key in self._connected_attrs:
                previous_pulse_match = self._connected_attrs.pop(key)
                # Remove function from pulse signal because it no longer
                # depends on other pulse
                previous_pulse_match.origin_pulse.signal.disconnect(
                    previous_pulse_match)

        if self.signal.receivers:
            # send signal to anyone listening that attribute has changed
            self.signal.send(self, **{key: value})
            if key in ['t_start', 'duration']:
                # Also send signal that dependent property t_stop has changed
                self.signal.send(self, t_stop=self.t_stop)

    def _value_or_config(self, key, value, default=None):
        """
        Decides what value to return depending on value and config.
        Used for setting pulse attributes at the start

        Args:
            key: key to check in config
            value: value to choose if not equal to None
            default: default value if no value specified. None by default

        Returns:
            if value is not None, return value
            elif config has key, return config[key]
            else return None

        """
        if value is not None:
            return value
        elif self.pulse_config is not None and key in self.pulse_config:
            return self.pulse_config[key]
        else:
            return default

    def __add__(self, other):
        """ This method is called when adding two pulses: `pulse1 + pulse2`.

        Args:
            other (Pulse): The pulse instance to be added to self.

        Returns:
            combined_pulse (Pulse): A new pulse instance representing the
                combination of two pulses.

        """
        name = 'CombinationPulse_{}'.format(id(self)+id(other))
        return CombinationPulse(name, self, other, '+')

    def __radd__(self, other):
        """ This method is called when reverse adding something to a pulse.

        The reason this method is implemented is so that the user can sum over
        multiple pulses by performing:

            combination_pulse = sum([pulse1, pulse2, pulse3])

        The sum method actually tries calling 0.__add__(pulse1), which doesn't
        exist, so it is converted into pulse1.__radd__(0).

        Args:
            other: an instance of unknown type that might be int(0)

        Returns:
            pulse (Pulse): Either self (if other is zero) or self + other.

        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        """ This method is called when subtracting two pulses: `pulse1 - pulse2`

        Args:
            other (Pulse): The pulse instance to be subtracted from self.

        Returns:
            combined_pulse (Pulse): A new pulse instance representing the
                combination of two pulses.

        """
        name = 'CombinationPulse_{}'.format(id(self)+id(other))
        return CombinationPulse(name, self, other, '-')

    def __mul__(self, other):
        """
        This method is called when multiplying two pulses: `pulse1 * pulse2`.

        Args:
            other (Pulse): The pulse instance to be multiplied with self.

        Returns:
            combined_pulse (Pulse): A new pulse instance representing the
                combination of two pulses.

        """
        name = 'CombinationPulse_{}'.format(id(self)+id(other))
        return CombinationPulse(name, self, other, '*')

    def __deepcopy__(self, *args):
        """
        Creates a copy of a pulse.
        Args:

        Returns:
            Copy of pulse
        """

        # Temporarily empty _connected_attrs as it may reference other pulses
        _connected_attrs, self._connected_attrs = self._connected_attrs, {}

        # Temporary remove __deepcopy__ to use deepcopy default method
        _deepcopy = Pulse.__deepcopy__
        try:
            del Pulse.__deepcopy__
            pulse_copy = deepcopy(self)
        finally:
            # restore __deepcopy__ and _connected_attrs
            Pulse.__deepcopy__ = _deepcopy
            self._connected_attrs = _connected_attrs

        # Add receiver for config signals
        if hasattr(self, 'environment'):
            # For PulseImplementation
            signal(f'config:{pulse_copy.environment}.pulses.'
                   f'{pulse_copy.name}').connect(
                pulse_copy._handle_config_signal)
            signal(f'config:{pulse_copy.environment}.properties').connect(
                pulse_copy._handle_properties_config_signal)
        return pulse_copy

    __copy__ = __deepcopy__

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

    @property
    def full_name(self):
        if self.id is None:
            return self.name
        else:
            return f'{self.name}[{self.id}]'

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, t_start):
        if t_start is not None:
            t_start = round(t_start, 11)
        self._t_start = t_start

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        if duration is not None:
            duration = round(duration, 11)
        self._duration = duration

    @property
    def t_stop(self):
        if self.t_start is not None and self.duration is not None:
            return round(self.t_start + self.duration, 11)
        else:
            return None

    @t_stop.setter
    def t_stop(self, t_stop):
        if t_stop is not None:
            # Setting duration sends a signal for duration and also t_stop
            self.duration = round(t_stop - self.t_start, 11)

    def _get_repr(self, properties_str):
        if self.connection:
            properties_str += f'\n\tconnection: {self.connection}'
        if self.connection_requirements:
            properties_str += f'\n\trequirements: {self.connection_requirements}'
        if hasattr(self, 'additional_pulses') and self.additional_pulses:
            properties_str += '\n\tadditional_pulses:'
            for pulse in self.additional_pulses:
                pulse_repr = '\t'.join(repr(pulse).splitlines(True))
                properties_str += f'\n\t{pulse_repr}'

        pulse_class = self.__class__.__name__
        return f'{pulse_class}({self.full_name}, {properties_str})'

    def satisfies_conditions(self, pulse_class=None, name=None, **kwargs):
        """
        Checks if pulse satisfies certain conditions.
        Each kwarg is a condition, and can be a value (equality testing) or it
        can be a tuple (relation, value), in which case the relation is tested.
        Possible relations: '>', '<', '>=', '<=', '=='
        Args:
            t_start:
            t:
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

        if name is not None:
            if name[-1] == ']':
                # Pulse id is part of name
                name, id = name[:-1].split('[')
                kwargs['id'] = int(id)
            kwargs['name'] = name

        for property, val in kwargs.items():
            if val is None:
                continue
            elif property == 't':
                if val < self.t_start or val >= self.t_stop:
                    return False
            elif not hasattr(self, property):
                return False
            else:
                # If arg is a tuple, the first element specifies its relation
                if isinstance(val, (list, tuple)):
                    relation, val = val
                else:
                    relation = '=='
                if not get_truth(test_val=getattr(self, property),
                                 target_val=val,
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

        self.t_no_blip = self._value_or_config('t_no_blip', t_no_blip)
        self.t_max_wait = self._value_or_config('t_max_wait', t_max_wait)
        self.t_buffer = self._value_or_config('t_buffer', t_buffer)
        self.readout_threshold_voltage = self._value_or_config(
            'readout_threshold_voltage', readout_threshold_voltage)

    def __repr__(self):
        try:
            properties_str = (f't_no_blip={self.t_no_blip} ms, ' +
                              f't_max_wait={self.t_max_wait}, ' +
                              f't_buffer={self.t_buffer}, ' +
                              f'V_th={self.readout_threshold_voltage}')
        except:
            properties_str = ''

        return super()._get_repr(properties_str)


class SinePulse(Pulse):
    def __init__(self, name=None, frequency=None, phase=None, amplitude=None,
                 power=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.frequency = self._value_or_config('frequency', frequency)
        self.phase = self._value_or_config('phase', phase)
        self.power = self._value_or_config('power', power)
        self.amplitude = self._value_or_config('amplitude', amplitude)

    def __repr__(self):
        properties_str = ''
        try:
            properties_str = f'f={freq_to_str(self.frequency)}'
            if self.power is not None:
                properties_str += f', power={self.power} dBm'

            if self.amplitude is not None:
                properties_str += f', A={self.amplitude} V'

            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"
        if self.phase is None:
            return self.amplitude * np.sin(2 * np.pi * self.frequency * (t - self.t_start))
        else:
            return self.amplitude * np.sin(2 * np.pi * (self.frequency * t + self.phase / 360))


class FrequencyRampPulse(Pulse):
    def __init__(self, name=None, frequency_start=None, frequency_stop=None,
                 frequency=None, frequency_deviation=None,
                 frequency_final='stop', amplitude=None, power=None,
                 frequency_sideband=None, **kwargs):
        super().__init__(name=name, **kwargs)

        if frequency_start is not None and frequency_stop is not None:
            self.frequency = (frequency_start + frequency_stop) / 2
            self.frequency_deviation = (frequency_stop - frequency_start)
        else:
            self.frequency = frequency
            if self.frequency is None:
                self.frequency = self.pulse_config.get('frequency', None)

            self.frequency_deviation = frequency_deviation
            if self.frequency_deviation is  None:
                self.frequency_deviation = self.pulse_config.get(
                    'frequency_deviation', None)

        self._frequency_final = frequency_final
        self.frequency_sideband = self._value_or_config('frequency_sideband',
                                                        frequency_sideband)

        self.amplitude = self._value_or_config('amplitude', amplitude)
        self.power = self._value_or_config('power', power)

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
        properties_str = ''
        try:
            properties_str = f'f={freq_to_str(self.frequency)}'
            properties_str += f', f_dev={freq_to_str(self.frequency_deviation)}'
            if self.frequency_sideband is not None:
                properties_str += ', f_sb={freq_to_str(self.frequency_sideband)}'
            properties_str += f', power={self.power}'
            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)


class DCPulse(Pulse):
    def __init__(self, name=None, amplitude=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.amplitude = self._value_or_config('amplitude', amplitude)

        if self.amplitude is None:
            raise AttributeError("'{}' object has no attribute "
                                 "'amplitude'".format(self.__class__.__name__))

    def __repr__(self):
        properties_str = ''
        try:
            properties_str += f'A={self.amplitude}'
            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)


    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        if isinstance(t, collections.Iterable):
            return np.ones(len(t)) * self.amplitude
        else:
            return self.amplitude


class DCRampPulse(Pulse):
    def __init__(self, name=None, amplitude_start=None, amplitude_stop=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.amplitude_start = self._value_or_config('amplitude_start',
                                                     amplitude_start)
        self.amplitude_stop = self._value_or_config('amplitude_stop',
                                                    amplitude_stop)

    def __repr__(self):
        properties_str = ''
        try:
            properties_str = f'A_start={self.amplitude_start}'
            properties_str += f', A_stop={self.amplitude_stop}'
            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            f"voltage at {t} s is not in the time range {self.t_start} s " \
            f"- {self.t_stop} s of pulse {self}"

        slope = (self.amplitude_stop - self.amplitude_start) / self.duration
        offset = self.amplitude_start - slope * self.t_start

        return offset + slope * t


class TriggerPulse(Pulse):
    duration = .1e-6

    def __init__(self, name=None, duration=duration, **kwargs):
        # Trigger pulses don't necessarily need a specific name
        if name is None:
            name = 'trigger'
        super().__init__(name=name, duration=duration, **kwargs)
        self.amplitude = self._value_or_config('amplitude', 1.0)

    def __repr__(self):
        try:
            properties_str = 't_start={}, duration={}'.format(
                self.t_start, self.duration)
        except:
            properties_str = ''
        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        if isinstance(t, collections.Iterable):
            return np.ones(len(t)) * self.amplitude
        else:
            return self.amplitude


class MarkerPulse(Pulse):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.amplitude = self._value_or_config('amplitude', 1.0)

    def __repr__(self):
        try:
            properties_str = 't_start={self.t_start}, duration={self.duration}'
        except:
            properties_str = ''
        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        if isinstance(t, collections.Iterable):
            return np.ones(len(t)) * self.amplitude
        else:
            return self.amplitude


class TriggerWaitPulse(Pulse):
    def __init__(self, name=None, t_start=None, **kwargs):
        super().__init__(name=name, t_start=t_start, duration=0, **kwargs)

    def __repr__(self):
        try:
            properties_str = 't_start={self.t_start}, duration={self.duration}'
        except:
            properties_str = ''

        return super()._get_repr(properties_str)


class MeasurementPulse(Pulse):
    def __init__(self, name=None, acquire=True, **kwargs):
        super().__init__(name=name, acquire=acquire, **kwargs)

    def __repr__(self):
        try:
            properties_str = f't_start={self.t_start}, duration={self.duration}'
        except:
            properties_str = ''
        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        raise NotImplementedError('Measurement pulses do not have a voltage')


class CombinationPulse(Pulse):
    """
    This class represents pulses that are combinations of multiple pulse types.

    For example:
        CombinationPulse = SinePulse + DCPulse
        CombinationPulse = DCPulse * SinePulse

    Like any other pulse, a CombinationPulse has a name, t_start and t_stop.
    t_start and t_stop are calculated and updated from the pulses that make up
    the combination.

    A CombinationPulse is itself a child of the Pulse class, therefore a
    CombinationPulse can also be used in consecutive combinations like:
        CombinationPulse1 = SinePulse1 + DCPulse
        CombinationPulse2 = SinePulse2 + CombinationPulse1

    Args:
        name (str): The name for this CombinationPulse.
        pulse1 (Pulse): The first pulse this combination is made up from.
        pulse2 (Pulse): The second pulse this combination is made up from.
        relation (str): The relation between pulse1 and pulse2.
            This must be one of the following:
                '+'     :   pulse1 + pulse2
                '-'     :   pulse1 - pulse2
                '*'     :   pulse1 * pulse2

    """

    def __init__(self, name=None, pulse1=None, pulse2=None, relation=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.pulse1 = pulse1
        self.pulse2 = pulse2
        self.relation = relation

        assert isinstance(pulse1, Pulse), 'pulse1 needs to be a Pulse'
        assert isinstance(pulse2, Pulse), 'pulse2 needs to be a Pulse'
        assert relation in ['+', '-', '*'], 'relation has a non-supported value'

    @property_ignore_setter
    def t_start(self):
        return min(self.pulse1.t_start, self.pulse2.t_start)

    @property_ignore_setter
    def t_stop(self):
        return max(self.pulse1.t_stop, self.pulse2.t_stop)

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
            pulse1_details = f'\t {self.pulse1.name} : {repr(self.pulse1)}\n'
        if isinstance(self.pulse2, CombinationPulse):
            pulse2_details = self.pulse2.pulse_details
        else:
            pulse2_details = f'\t {self.pulse2.name} : {repr(self.pulse2)}\n'
        return pulse1_details + pulse2_details

    def __repr__(self):
        properties_str = ''
        try:
            properties_str = 'combination: {self.combination}'
            properties_str += ', {self.pulse_details}'
        except:
            pass

        return super()._get_repr(properties_str)


    def get_voltage(self, t):
        assert self.t_start <= np.min(t) and np.max(t) <= self.t_stop, \
            "voltage at {} s is not in the time range {} s - {} s of " \
            "pulse {}".format(t, self.t_start, self.t_stop, self)

        result1 = np.zeros(t.shape[0])
        result2 = np.zeros(t.shape[0])

        pulse1_t = t[np.all([self.pulse1.t_start <= t, t <= self.pulse1.t_stop],
                            axis=0)]
        pulse2_t = t[np.all([self.pulse2.t_start <= t, t <= self.pulse2.t_stop],
                            axis=0)]

        voltage1 = self.pulse1.get_voltage(pulse1_t)
        voltage2 = self.pulse2.get_voltage(pulse2_t)

        result1[np.all([self.pulse1.t_start <= t, t <= self.pulse1.t_stop],
                       axis=0)] = voltage1
        result2[np.all([self.pulse2.t_start <= t, t <= self.pulse2.t_stop],
                       axis=0)] = voltage2

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
        properties_str = ''
        try:
            if self.from_function:
                properties_str = f'function:{self.function}'
            else:
                properties_str = f'array:{self.array.shape}'
            properties_str += ', t_start={self.t_start}'
            properties_str += ', duration={self.duration}'
        except:
            pass
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
