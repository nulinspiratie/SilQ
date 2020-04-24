from typing import Union, Sequence, Callable, List
import numpy as np
import collections
import logging

from silq.tools.general_tools import get_truth, property_ignore_setter, \
    freq_to_str, is_between

from qcodes.instrument.parameter_node import ParameterNode, parameter
from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators as vals

__all__ = ['Pulse', 'SteeredInitialization', 'SinePulse', 'MultiSinePulse',
           'FrequencyRampPulse', 'DCPulse', 'DCRampPulse', 'TriggerPulse',
           'MarkerPulse', 'TriggerWaitPulse', 'MeasurementPulse',
           'CombinationPulse', 'AWGPulse', 'pulse_conditions']

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
pulse_conditions = ['name', 'id', 't', 't_start', 't_stop',
                    'duration', 'acquire', 'initialize', 'connection',
                    'amplitude', 'enabled', 'average', 'pulse_class']

logger = logging.getLogger(__name__)


class Pulse(ParameterNode):
    """ Representation of physical pulse, component in a `PulseSequence`.

    A Pulse is a representation of a physical pulse, usually one that is
    outputted by an instrument. All pulses have specific timings, defined by
    ``t_start``, ``t_stop``, and ``duration``. Additional attributes specify
    ancillary properties, such as if the acquisition instrument should
    ``acquire`` the pulse. Specific pulse types (subclasses of `Pulse`) have
    additional properties, such as a ``frequency``.

    Pulses can be added to a `PulseSequence`, which can in turn be targeted by
    the `Layout`. Here, each `Pulse` is targeted to a `Connection`, which can
    modify the pulse (e.g. applying an amplitude scale). Next the pulse is
    targeted by the output and input `InstrumentInterface` of the connection,
    which provide an instrument-specific implementation of the pulse.

    Pulses usually have a name, which is used to retrieve any default properties
    from the config. If the pulse name is an entry in
    ``silq.config.{environment}.pulses``, the properties in that entry are
    used by default. These default values can be overridden by either passing
    them explicitly during the pulse initialization, or afterwards.

    Example:
        If ``silq.config.{environment}.pulses`` contains:

        >>> {'read': {'amplitude': 0.5, 'duration': 100e-3}}

        Then creating the following pulse will partially use these properties:

        >>> DCPulse('read', duration=200e-3)
        DCPulse('read', amplitude=0.5, duration=200e-3)

        Here the default ``amplitude`` value is used, but the duration is
        overridden during initialization.

    Parameters:
        name: Pulse name. If corresponding name is registered in pulse
            config, its properties will be copied to the pulse.
        id: Unique pulse identifier, assigned when added to `PulseSequence` if
            it already has another pulse with same name. Pre-existing pulse will
            be assigned id 0, and will increase for each successive pulse added.
        full_name: Pulse name, including id if not None.
            If id is not None, full_name is '{name}[{id}]'
        environment: Config environment to use for pulse config. If not set,
            default environment (``silq.config.properties.default_environment``)
            is used.
        t_start: Pulse start time. If undefined and added to `PulseSequence`, it
            will be set to `Pulse`.t_stop of last pulse.
            If no pulses are present, it will be set to zero.
        t_stop: Pulse stop time. Is updated whenever ``t_start`` or ``duration``
            is changed. Changing this modifies ``duration`` but not ``t_start``.
        duration: Pulse duration.
        acquire: Flag to acquire pulse. If True, pulse will be passed on to the
            acquisition `InstrumentInterface` by the `Layout` during targeting.
        initialize: Pulse is used for initialization. This signals that the
            pulse can exist before the pulse sequence starts. In this case,
            pulse duration should be zero.
        connection (Connection): Connection that pulse is targeted to.
            Is only set for targeted pulse.
        enabled: Pulse is enabled. If False, it still exists in a
            PulseSequence, but is not included in targeting.
        average: Pulse acquisition average mode. Allowed modes are:

            * **'none'**: No averaging (return ``samples x points_per_trace``).
            * **'trace'**: Average over time (return ``points_per_trace``).
            * **'point'**: Average over time and sample (return single point).
            * **'point_segment:{N}'** Segment trace into N segment, average
              each segment into a point.

        connection_label: `Connection` label that Pulse should be targeted to.
            These are defined in ``silq.config.{environment}.connections``.
            If unspecified, pulse can only be targeted if
            ``connection_requirements`` uniquely determine connection.
        connection_requirements: Requirements that a connection must satisfy for
            targeting. If ``connection_label`` is defined, these are ignored.
        pulse_config: Pulse config whose attributes to match. If it exists,
            equal to ``silq.config.{environment}.pulses.{pulse.name}``,
            otherwise equal to zero.
        properties_config: General properties config whose attributes to match.
            If it exists, equal to ``silq.config.{environment}.properties``,
            otherwise None. Only `Pulse`.properties_attrs are matched.
        properties_attrs (List[str]): Attributes in properties config to match.
            Should be defined in ``__init__`` before calling ``Pulse.__init__``.
        implementation (PulseImplementation): Pulse implementation for targeted
            pulse, see `PulseImplementation`.
        connect_to_config: Connect parameters to the config (default True)
    """
    # base config link to use for connecting pulse parameters to the config
    # Changing this will only affect pulses instantiated after change
    config_link = 'environment:pulses'
    multiple_senders = False

    def __init__(self,
                 name: str = None,
                 id: int = None,
                 t_start: float = None,
                 t_stop: float = None,
                 duration: float = None,
                 acquire: bool = False,
                 initialize: bool = False,
                 connection = None,
                 enabled: bool = True,
                 average: str = 'none',
                 connection_label: str = None,
                 connection_requirements: dict = {},
                 connect_to_config: bool = True):
        super().__init__(use_as_attributes=True,
                         log_changes=False,
                         simplify_snapshot=True)

        self.name = Parameter(initial_value=name, vals=vals.Strings(), set_cmd=None)
        self.id = Parameter(initial_value=id, vals=vals.Ints(allow_none=True),
                            set_cmd=None, wrap_get=False)
        self.full_name = Parameter()
        self['full_name'].get()  # Update to latest value

        ### Set attributes
        # Set attributes that can also be retrieved from pulse_config
        self.t_start = Parameter(initial_value=t_start,
                                 unit='s', set_cmd=None, wrap_get=False)
        self.duration = Parameter(initial_value=duration, unit='s', set_cmd=None, wrap_get=False)
        self.t_stop = Parameter(unit='s', wrap_get=False)

        # We separately set and get t_stop to ensure duration is also updated
        self.t_stop = t_stop
        self['t_stop'].get()

        # Since t_stop get/set cmd depends on t_start and duration, we perform
        # another set to ensure that duration is also set if t_stop is not None
        if self['t_stop'].raw_value is not None:
            self.t_stop = self['t_stop'].raw_value
        self.connection_label = Parameter(initial_value=connection_label, set_cmd=None)

        # Set attributes that should not be retrieved from pulse_config
        self.acquire = Parameter(initial_value=acquire, vals=vals.Bool(), set_cmd=None)
        self.initialize = Parameter(initial_value=initialize, vals=vals.Bool(), set_cmd=None)
        self.enabled = Parameter(initial_value=enabled, vals=vals.Bool(), set_cmd=None)
        self.connection = Parameter(initial_value=connection, set_cmd=None)
        self.average = Parameter(initial_value=average, vals=vals.Strings(), set_cmd=None)

        # Pulses can have a PulseImplementation after targeting
        self.implementation = None

        # List of potential connection requirements.
        # These can be set so that the pulse can only be sent to connections
        # matching these requirements
        self.connection_requirements = connection_requirements

        self._connected_to_config = False
        if connect_to_config:
            # Sets _connected_to_config to True
            self._connect_parameters_to_config()

    @parameter
    def average_vals(self, parameter, value):
        if value in ['none', 'trace', 'point']:
            return True
        elif ('point_segment' in value or 'trace_segment' in value):
            return True
        else:
            return False

    @parameter
    def full_name_get(self, parameter):
        if self.id is None:
            return self.name
        else:
            return f'{self.name}[{self.id}]'

    @parameter
    def t_start_set_parser(self, parameter, t_start):
        if t_start is not None:
            t_start = round(t_start, 11)
        return t_start

    @parameter
    def t_start_set(self, parameter, t_start):
        # Emit a t_stop signal when t_start is set
        self['t_start']._latest['raw_value'] = t_start
        self['t_stop'].set(self.t_stop, evaluate=False)

    @parameter
    def duration_set_parser(self, parameter, duration):
        if duration is not None:
            duration = round(duration, 11)
        return duration

    @parameter
    def duration_set(self, parameter, duration):
        # Emit a t_stop signal when duration is set
        self['duration']._latest['raw_value'] = duration
        self['t_stop'].set(self.t_stop, evaluate=False)

    @parameter
    def t_stop_get(self, parameter):
        if self.t_start is not None and self.duration is not None:
            val = round(self.t_start + self.duration, 11)
        else:
            val = None
        parameter._save_val(val)  # Explicit save_val since we don't wrap_get
        return val

    @parameter
    def t_stop_set(self, parameter, t_stop):
        if t_stop is not None:
            # Setting duration sends a signal for duration
            # do not evaluate as it otherwise sends a second t_stop signal
            self['duration'].set(round(t_stop - self.t_start, 11),
                                 evaluate=False)

    def __eq__(self, other):
        """Comparison when pulses are equal

        Pulses are equal if all of their parameters (excluding list below) are
        equal. Furthermore, their classes need to be identical, which further
        means that any non-pulse object will never be equal.

        Connections are handled slightly differently. For pulses to be the same,
        they must either have the same connection or connection_label.
        Alternatively, if one pulse has a connection and the other a connection
        label, the label of the first pulse's connection is compared instead.
        If one pulse has a connection/connection label, and the other has
        neither, the pulses are not equal.

        Excluded parameters:
            - id
            - connection_requirements

        Returns:
            True if all above conditions hold, False otherwise.
        """
        exclude_parameters = ['connection', 'connection_label', 'id', 'full_name']

        if not self.matches_parameter_node(other, exclude_parameters=exclude_parameters):
            return False

        # Perform additional checks based on connection (labels).
        # Pulses are equal if they have the same connection, or has a matching
        # label, or if both pulses don't have a connection and connection label.
        if self.connection is not None:
            if other.connection is not None:
                return self.connection == other.connection
            elif other.connection_label is not None:
                return self.connection.label == other.connection_label
            else:
                return False
        elif self.connection_label is not None:
            if other.connection is not None:
                return self.connection_label == other.connection.label
            elif other.connection_label is not None:
                return self.connection_label == other.connection_label
            else:
                return False
        else:
            return other.connection is None and other.connection_label is None

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """Define custom hash, used for creating a set of unique elements"""
        return hash(tuple(sorted(self.parameters.items())))

    def __bool__(self):
        """Pulse is always equal to True"""
        return True

    def __add__(self,
                other: 'Pulse') -> 'CombinationPulse':
        """ This method is called when adding two pulses: ``pulse1 + pulse2``.

        Args:
            other: The pulse instance to be added to self.

        Returns:
            A new pulse instance representing the combination of two pulses.

        """
        name = f'CombinationPulse_{id(self)+id(other)}'
        return CombinationPulse(name, self, other, '+')

    def __radd__(self, other) -> 'Pulse':
        """ This method is called when reverse adding something to a pulse.

        The reason this method is implemented is so that the user can sum over
        multiple pulses by performing:

        >>> combination_pulse = sum([pulse1, pulse2, pulse3])

        The sum method actually tries calling 0.__add__(pulse1), which doesn't
        exist, so it is converted into pulse1.__radd__(0).

        Args:
            other: an instance of unknown type that might be int(0)

        Returns:
            Either self (if other is zero) or self + other.

        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other: 'Pulse') -> 'CombinationPulse':
        """Called when subtracting two pulses: ``pulse1 - pulse2``

        Args:
            other: The pulse instance to be subtracted from self.

        Returns:
            A new pulse instance representing the combination of two pulses.
        """
        name = f'CombinationPulse_{id(self)+id(other)}'
        return CombinationPulse(name, self, other, '-')

    def __mul__(self, other: 'Pulse') -> 'CombinationPulse':
        """Called when multiplying two pulses: ``pulse1 * pulse2``.

        Args:
            other: The pulse instance to be multiplied with self.

        Returns:
            A new pulse instance representing the combination of two pulses.

        """
        name = f'CombinationPulse_{id(self)+id(other)}'
        return CombinationPulse(name, self, other, '*')

    def __copy__(self):
        """Create a copy of the pulse.

        Aside from using the default copy feature of the ParameterNode, this
        also connects the copied parameters to the config if the original ones
        are also connected
        """
        self_copy = super().__copy__()
        if self._connected_to_config:
            self_copy._connect_parameters_to_config()
        return self_copy

    def _get_repr(self, properties_str):
        """Get standard representation for pulse.

        Should be appended in each Pulse subclass."""
        if self.connection:
            properties_str += f'\n\tconnection: {self.connection}'
        elif self.connection_label:
            properties_str += f'\n\tconnection_label: {self.connection_label}'
        if self.connection_requirements:
            properties_str += f'\n\trequirements: {self.connection_requirements}'
        if hasattr(self, 'additional_pulses') and self.additional_pulses:
            properties_str += '\n\tadditional_pulses:'
            for pulse in self.additional_pulses:
                pulse_repr = '\t'.join(repr(pulse).splitlines(True))
                properties_str += f'\n\t{pulse_repr}'

        pulse_class = self.__class__.__name__
        return f'{pulse_class}({self.full_name}, {properties_str})'

    def _connect_parameters_to_config(self, parameters=None):
        """Connect Pulse parameters to config using Pulse.config_link.

        By connecting a parameter, every time the corresponding config value
        changes, this in turn changes the parameter value.

        The config link is {Pulse.config_link}.{self.name}.{parameter.name}

        Args:
             parameters: Parameters to Connect. Can be

             - None: Connect all parameters in self.parameters
             - str list: Connect all parameter with given string names
             - Parameter list: Connect all parameters in list
        """
        if isinstance(parameters, list):
            if isinstance(parameters[0], str):
                parameters = {parameter: self.parameters[parameter]
                              for parameter in parameters}
            else:
                parameters = {parameter.name: parameter
                              for parameter in parameters}
        elif parameters is None:
            parameters = self.parameters

        for parameter_name, parameter in parameters.items():
            config_link = f'{self.config_link}.{self.name}.{parameter_name}'
            config_value = parameter.set_config_link(config_link=config_link)

            # Update parameter value if not yet set, and set in config
            if parameter.raw_value is None and config_value is not None:
                parameter(config_value)

        self._connected_to_config = True

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=None):
        snapshot = super().snapshot_base()
        if snapshot['connection']:
            snapshot['connection'] = repr(snapshot['connection'])
        return snapshot

    def satisfies_conditions(self,
                             pulse_class = None,
                             name: str=None,
                             **kwargs) -> bool:
        """Checks if pulse satisfies certain conditions.

        Each kwarg is a condition, and can be a value (equality testing) or it
        can be a tuple (relation, value), in which case the relation is tested.
        Possible relations: '>', '<', '>=', '<=', '=='

        Args:
            pulse_class: Pulse must have specific class.
            name: Pulse must have name, which may include id.
            **kwargs: Additional pulse attributes to be satisfied.
                Examples are ``t_start``, ``connection``, etc.
                Time ``t`` can also be passed, in which case the condition is
                satisfied if t is between `Pulse`.t_start and `Pulse`.t_stop
                (including limits).

        Returns:
            True if all conditions are satisfied.
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
            elif property not in self.parameters:
                return False
            else:
                # If arg is a tuple, the first element specifies its relation
                if isinstance(val, (list, tuple)):
                    relation, val = val
                    if not get_truth(test_val=self.parameters[property].get_latest(),
                            # test_val=getattr(self, property),
                                     target_val=val,
                                     relation=relation):
                        return False
                elif self.parameters[property]._latest['raw_value'] != val:
                    return False
        else:
            return True

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        raise NotImplementedError('Pulse.get_voltage should be implemented in a subclass')


class SteeredInitialization(Pulse):
    """Initialization pulse to ensure a spin-down electron is loaded.

    This is performed by continuously measuring at the read stage until no blip
    has been measured for ``t_no_blip``, or until ``t_max_wait`` has elapsed.

    Parameters:
        name: Pulse name
        t_no_blip: Min duration without measuring blips. If condition is met,
            an event should be fired to the primary instrument to start the
            pulse sequence.
        t_max_wait: Maximum wait time for the no-blip condition.
            If ``t_max_wait`` has elapsed, an event should be fired to the
            primary instrument to start the pulse seqeuence.
        t_buffer: Duration of a single acquisition buffer. Shorter buffers mean
            that one can more closely approach ``t_no_blip``, but too short
            buffers may cause lagging.
        readout_threshold_voltage: Threshold voltage for a blip.
        **kwargs: Additional parameters of `Pulse`.
    """
    def __init__(self,
                 name: str = None,
                 t_no_blip: float = None,
                 t_max_wait: float = None,
                 t_buffer: float = None,
                 readout_threshold_voltage: float = None,
                 **kwargs):
        super().__init__(name=name, t_start=0, duration=0, initialize=True,
                         **kwargs)

        self.t_no_blip = Parameter(initial_value=t_no_blip, unit='s',
                                   set_cmd=None, vals=vals.Numbers())
        self.t_max_wait = Parameter(initial_value=t_max_wait, unit='s',
                                   set_cmd=None, vals=vals.Numbers())
        self.t_buffer = Parameter(initial_value=t_buffer, unit='s',
                                   set_cmd=None, vals=vals.Numbers())
        self.readout_threshold_voltage = Parameter(initial_value=readout_threshold_voltage,
                                                   unit='V', set_cmd=None,
                                                   vals=vals.Numbers())

        self._connect_parameters_to_config(
            ['t_no_blip', 't_max_wait', 't_buffer', 'readout_threshold_voltage'])

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
    """Sinusoidal pulse

    Parameters:
        name: Pulse name
        frequency: Pulse frequency
        phase: Pulse phase
        amplitude: Pulse amplitude. If not set, power must be set.
        power: Pulse power. If not set, amplitude must be set.
        offset: amplitude offset, zero by default
        frequency_sideband: Mixer sideband frequency (off by default).
        sideband_mode: Sideband frequency to apply. This feature must
            be existent in interface. Not used if not set.
        phase_reference: What point in the the phase is with respect to.
            Can be two modes:

            - 'absolute': phase is with respect to `Pulse.t_start`.
            - 'relative': phase is with respect to t=0 (phase-coherent).

        **kwargs: Additional parameters of `Pulse`.

    Notes:
        Either amplitude or power must be set, depending on the instrument
        that should output the pulse.
    """
    def __init__(self,
                 name: str = None,
                 frequency: float = None,
                 phase: float = None,
                 amplitude: float = None,
                 power: float = None,
                 offset: float = None,
                 frequency_sideband: float = None,
                 sideband_mode: float = None,
                 phase_reference: str = None,
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.frequency = Parameter(initial_value=frequency, unit='Hz',
                                   set_cmd=None, vals=vals.Numbers())
        self.phase = Parameter(initial_value=phase, unit='deg', set_cmd=None,
                               vals=vals.Numbers())
        self.power = Parameter(initial_value=power, unit='dBm', set_cmd=None,
                               vals=vals.Numbers())
        self.amplitude = Parameter(initial_value=amplitude, unit='V',
                                   set_cmd=None, vals=vals.Numbers())
        self.offset = Parameter(initial_value=offset, unit='V', set_cmd=None,
                                vals=vals.Numbers())
        self.frequency_sideband = Parameter(initial_value=frequency_sideband,
                                            unit='Hz', set_cmd=None,
                                            vals=vals.Numbers())
        self.sideband_mode = Parameter(initial_value=sideband_mode, set_cmd=None,
                                       vals=vals.Enum('IQ', 'double'))
        self.phase_reference = Parameter(initial_value=phase_reference,
                                         set_cmd=None, vals=vals.Enum('relative',
                                                                      'absolute'))
        self._connect_parameters_to_config(
            ['frequency', 'phase', 'power', 'amplitude', 'phase', 'offset',
             'frequency_sideband', 'sideband_mode', 'phase_reference'])

        if self.sideband_mode is None:
            self.sideband_mode = 'IQ'
        if self.phase_reference is None:
            self.phase_reference = 'absolute'
        if self.phase is None:
            self.phase = 0
        if self.offset is None:
            self.offset = 0

    def __repr__(self):
        properties_str = ''
        try:
            properties_str = f'f={freq_to_str(self.frequency)}'
            properties_str += f', phase={self.phase} deg '
            properties_str += '(rel)' if self.phase_reference == 'relative' else '(abs)'
            if self.power is not None:
                properties_str += f', power={self.power} dBm'

            if self.amplitude is not None:
                properties_str += f', A={self.amplitude} V'

            if self.offset:
                properties_str += f', offset={self.offset} V'
            if self.frequency_sideband is not None:
                properties_str += f'f_sb={freq_to_str(self.frequency_sideband)} ' \
                                  f'{self.sideband_mode}'

            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        if self.phase_reference == 'relative':
            t = t - self.t_start

        amplitude = self.amplitude
        if amplitude is None:
            assert self.power is not None, f'Pulse {self.name} does not have a specified power or amplitude.'
            if self['power'].unit == 'dBm':
                # This formula assumes the source is 50 Ohm matched and power is in dBm
                # A factor of 2 comes from the conversion from amplitude to RMS.
                amplitude = np.sqrt(10**(self.power/10) * 1e-3 * 100)

        waveform = amplitude * np.sin(2 * np.pi * (self.frequency * t + self.phase / 360))
        waveform += self.offset

        return waveform


class MultiSinePulse(Pulse):
    """MultiSinusoidal pulse

    Parameters:
        name: Pulse name
        frequency: main Pulse frequency - !!! to be checked !!!
        frequencies: list of pulse frequencies
        phase: Pulse phase
        amplitude: Pulse amplitude. If not set, power must be set.
        power: Pulse power. If not set, amplitude must be set.
        offset: amplitude offset, zero by default
        frequency_sideband: Mixer sideband frequency (off by default).
        sideband_mode: Sideband frequency to apply. This feature must
            be existent in interface. Not used if not set.
        phase_reference: What point in the the phase is with respect to.
            Can be two modes:
            - 'absolute': phase is with respect to `Pulse.t_start`.
            - 'relative': phase is with respect to t=0 (phase-coherent).

        **kwargs: Additional parameters of `Pulse`.

    Notes:
        Either amplitude or power must be set, depending on the instrument
        that should output the pulse.
    """
    def __init__(self,
                 name: str = None,
                 frequency: float = None,
                 frequencies: List[float] = None,
                 phase: float = None,
                 amplitude: float = None,
                 power: float = None,
                 offset: float = None,
                 frequency_sideband: float = None,
                 sideband_mode: float = None,
                 phase_reference: str = None,
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self.frequency = Parameter(initial_value=frequency, unit='Hz',
                                   set_cmd=None, vals=vals.Numbers())
        self.frequencies = Parameter(initial_value=frequencies, unit='Hz',
                                     set_cmd=None, vals=vals.Lists())
        self.phase = Parameter(initial_value=phase, unit='deg', set_cmd=None,
                               vals=vals.Numbers())
        self.power = Parameter(initial_value=power, unit='dBm', set_cmd=None,
                               vals=vals.Numbers())
        self.amplitude = Parameter(initial_value=amplitude, unit='V',
                                   set_cmd=None, vals=vals.Numbers())
        self.offset = Parameter(initial_value=offset, unit='V', set_cmd=None,
                                vals=vals.Numbers())
        self.frequency_sideband = Parameter(initial_value=frequency_sideband,
                                            unit='Hz', set_cmd=None,
                                            vals=vals.Numbers())
        self.sideband_mode = Parameter(initial_value=sideband_mode, set_cmd=None,
                                       vals=vals.Enum('IQ', 'double'))
        self.phase_reference = Parameter(initial_value=phase_reference,
                                         set_cmd=None, vals=vals.Enum('relative',
                                                                      'absolute'))
        self._connect_parameters_to_config(
            ['frequency', 'frequencies', 'phase', 'power', 'amplitude', 'phase',
             'offset', 'frequency_sideband', 'sideband_mode', 'phase_reference'])

        if self.sideband_mode is None:
            self.sideband_mode = 'IQ'
        if self.phase_reference is None:
            self.phase_reference = 'absolute'
        if self.phase is None:
            self.phase = 0
        if self.offset is None:
            self.offset = 0

    def __repr__(self):
        properties_str = ''
        try:
            properties_str = f'f_main={freq_to_str(self.frequency)}'
            properties_str += f'f={[freq_to_str(freq) for freq in self.frequencies]}'
            properties_str += f', phase={self.phase} deg '
            properties_str += '(rel)' if self.phase_reference == 'relative' else '(abs)'
            if self.power is not None:
                properties_str += f', power={self.power} dBm'

            if self.amplitude is not None:
                properties_str += f', A={self.amplitude} V'

            if self.offset:
                properties_str += f', offset={self.offset} V'
            if self.frequency_sideband is not None:
                properties_str += f'f_sb={freq_to_str(self.frequency_sideband)} ' \
                                  f'{self.sideband_mode}'

            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        if self.phase_reference == 'relative':
            t = t - self.t_start

        amplitude = self.amplitude
        if amplitude is None:
            assert self.power is not None, f'Pulse {self.name} does not have a specified power or amplitude.'
            if self['power'].unit == 'dBm':
                # This formula assumes the source is 50 Ohm matched and power is in dBm
                # A factor of 2 comes from the conversion from amplitude to RMS.
                amplitude = np.sqrt(10**(self.power/10) * 1e-3 * 100)
        waveform = 0.0
        for frequency in self.frequencies:
            waveform += amplitude * np.sin(2 * np.pi * (frequency * t + self.phase / 360))
        waveform += self.offset
        # We need to insure the waveform is limited by +- 1V:
        waveform = waveform/np.max(waveform)
        return waveform


class FrequencyRampPulse(Pulse):
    """Linearly increasing/decreasing frequency `Pulse`.

    Parameters:
        name: Pulse name
        frequency_start: Start frequency
        frequency_stop: Stop frequency.
        frequency: Center frequency, only used if ``frequency_start`` and
            ``frequency_stop`` not used.
        frequency_deviation: Frequency deviation, only used if
            ``frequency_start`` and ``frequency_stop`` not used.
        frequency_final: Can be either ``start`` or ``stop`` indicating the
            frequency when reaching ``frequency_stop`` should go back to the
            initial frequency or stay at current frequency. Useful if the
            pulse doesn't immediately stop at the end (this depends on how
            the corresponding instrument/interface is programmed).
        amplitude: Pulse amplitude. If not set, power must be set.
        power: Pulse power. If not set, amplitude must be set.
        offset: amplitude offset, zero by default
        frequency_sideband: Sideband frequency to apply. This feature must
            be existent in interface. Not used if not set.
        sideband_mode: Type of mixer sideband ('IQ' by default)

        **kwargs: Additional parameters of `Pulse`.

    Notes:
        Either amplitude or power must be set, depending on the instrument
        that should output the pulse.
    """
    def __init__(self,
                 name: str = None,
                 frequency_start: float = None,
                 frequency_stop: float = None,
                 frequency: float = None,
                 frequency_deviation: float = None,
                 amplitude: float = None,
                 power: float = None,
                 offset: float = None,
                 phase: float = None,
                 frequency_sideband: float = None,
                 sideband_mode=None,
                 phase_reference: str = None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        if frequency_start is not None and frequency_stop is not None:
            frequency = (frequency_start + frequency_stop) / 2
            frequency_deviation = (frequency_stop - frequency_start) / 2

        self.frequency = Parameter(initial_value=frequency, unit='Hz',
                                   set_cmd=None, vals=vals.Numbers())
        self.frequency_deviation = Parameter(initial_value=frequency_deviation,
                                             unit='Hz', set_cmd=None,
                                             vals=vals.Numbers())
        self.frequency_start = Parameter(unit='Hz', set_cmd=None,
                                         vals=vals.Numbers())
        self.frequency_stop = Parameter(unit='Hz', set_cmd=None,
                                        vals=vals.Numbers())
        self.frequency_sideband = Parameter(initial_value=frequency_sideband,
                                            unit='Hz', set_cmd=None,
                                            vals=vals.Numbers())
        self.sideband_mode = Parameter(initial_value=sideband_mode, set_cmd=None,
                                       vals=vals.Enum('IQ', 'double'))
        self.amplitude = Parameter(initial_value=amplitude, unit='V', set_cmd=None,
                                   vals=vals.Numbers())
        self.power = Parameter(initial_value=power, unit='dBm', set_cmd=None,
                               vals=vals.Numbers())
        self.phase = Parameter(initial_value=phase, unit='deg', set_cmd=None,
                               vals=vals.Numbers())
        self.offset = Parameter(initial_value=offset, unit='V', set_cmd=None,
                                vals=vals.Numbers())
        self.phase_reference = Parameter(initial_value=phase_reference,
                                         set_cmd=None, vals=vals.Enum('relative',
                                                                      'absolute'))
        self._connect_parameters_to_config(
            ['frequency', 'frequency_deviation', 'frequency_start',
             'frequency_stop', 'frequency_sideband', 'sideband_mode',
             'amplitude', 'power', 'phase', 'offset', 'phase_reference'])

        # Set default value for sideband_mode after connecting parameters,
        # because its value may have been retrieved from config
        if self.sideband_mode is not None:
            self.sideband_mode = 'IQ'
        if self.phase is None:
            self.phase = 0
        if self.offset is None:
            self.offset = 0
        if self.phase_reference is None:
            self.phase_reference = 'relative'

    @parameter
    def frequency_start_get(self, parameter):
        return self.frequency - self.frequency_deviation

    @parameter
    def frequency_start_set(self, parameter, frequency_start):
        frequency_stop = self.frequency_stop
        self.frequency = (frequency_start + frequency_stop) / 2
        self.frequency_deviation = (frequency_stop - frequency_start) / 2

    @parameter
    def frequency_stop_get(self, parameter):
        return self.frequency + self.frequency_deviation

    @parameter
    def frequency_stop_set(self, parameter, frequency_stop):
        frequency_start = self.frequency_start
        self.frequency = (frequency_start + frequency_stop) / 2
        self.frequency_deviation = (frequency_stop - frequency_start) / 2

    def __repr__(self):
        properties_str = ''
        try:
            properties_str = f'f={freq_to_str(self.frequency)}'
            properties_str += f', f_dev={freq_to_str(self.frequency_deviation)}'
            if self.frequency_sideband is not None:
                properties_str += f', f_sb={freq_to_str(self.frequency_sideband)}' \
                                  f'{self.sideband_mode}'

            if self.power is not None:
                properties_str += f', power={self.power} dBm'

            if self.amplitude is not None:
                properties_str += f', A={self.amplitude} V'

            if self.offset:
                properties_str += f', offset={self.offset} V'

            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)

    def get_voltage(self, t):
        frequency_rate = self.frequency_deviation / self.duration
        frequency_start = self.frequency - self.frequency_deviation
        amplitude = self.amplitude
        if amplitude is None:
            assert self.power is not None, f'Pulse {self.name} does not have a specified power or amplitude.'
            if self['power'].unit == 'dBm':
                # This formula assumes the source is 50 Ohm matched and power is in dBm
                # A factor of 2 comes from the conversion from amplitude to RMS.
                amplitude = np.sqrt(10 ** (self.power / 10) * 1e-3 * 100)

        if self.phase_reference == 'relative':
            t = t - self.t_start

        return amplitude * np.sin(2 * np.pi * (frequency_start * t + frequency_rate * np.power(t,2) / 2))


class DCPulse(Pulse):
    """DC (fixed-voltage) `Pulse`.

    Parameters:
        name: Pulse name
        amplitue: Pulse amplitude
        **kwargs: Additional parameters of `Pulse`.
    """
    def __init__(self,
                 name: str = None, amplitude: float = None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.amplitude = Parameter(initial_value=amplitude, unit='V',
                                   set_cmd=None)

        self._connect_parameters_to_config(['amplitude'])


    def __repr__(self):
        properties_str = ''
        try:
            properties_str += f'A={self.amplitude}'
            properties_str += f', t_start={self.t_start}'
            properties_str += f', duration={self.duration}'
        except:
            pass

        return super()._get_repr(properties_str)

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        if isinstance(t, collections.Iterable):
            return np.ones(len(t)) * self.amplitude
        else:
            return self.amplitude


class DCRampPulse(Pulse):
    """Linearly ramping voltage `Pulse`.

    Parameters:
        name: Pulse name
        amplitude_start: Start amplitude of pulse.
        amplitude_stop: Final amplitude of pulse.
        **kwargs: Additional parameters of `Pulse`.
    """
    def __init__(self,
                 name: str = None,
                 amplitude_start: float = None,
                 amplitude_stop: float = None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.amplitude_start = Parameter(initial_value=amplitude_start,
                                         unit='V', set_cmd=None,
                                         vals=vals.Numbers())
        self.amplitude_stop = Parameter(initial_value=amplitude_stop, unit='V',
                                        set_cmd=None, vals=vals.Numbers())

        self._connect_parameters_to_config(['amplitude_start', 'amplitude_stop'])

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

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
            f"voltage at {t} s is not in the time range {self.t_start} s " \
            f"- {self.t_stop} s of pulse {self}"

        slope = (self.amplitude_stop - self.amplitude_start) / self.duration
        offset = self.amplitude_start - slope * self.t_start

        return offset + slope * t


class TriggerPulse(Pulse):
    """Triggering pulse.

    Parameters:
        name: Pulse name.
        duration: Pulse duration (default 100 ns).
        amplitude: Pulse amplitude (default 1V).
        **kwargs: Additional parameters of `Pulse`.

    """
    default_duration = 100e-9
    default_amplitude = 1.0

    def __init__(self,
                 name: str = 'trigger',
                 duration: float = default_duration,
                 amplitude: float = default_amplitude,
                 **kwargs):
        super().__init__(name=name, duration=duration, **kwargs)

        self.amplitude = Parameter(initial_value=amplitude, unit='V',
                                   set_cmd=None, vals=vals.Numbers())

        self._connect_parameters_to_config(['amplitude'])

    def __repr__(self):
        try:
            properties_str = f't_start={self.t_start}, duration={self.duration}'
        except:
            properties_str = ''
        return super()._get_repr(properties_str)

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        if isinstance(t, collections.Iterable):
            return np.ones(len(t)) * self.amplitude
        else:
            return self.amplitude


class MarkerPulse(Pulse):
    """Marker pulse

    Parameters:
        name: Pulse name.
        amplitude: Pulse amplitude (default 1V).
        **kwargs: Additional parameters of `Pulse`.

    """
    default_amplitude = 1.0

    def __init__(self,
                 name: str = None,
                 amplitude: float = default_amplitude,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.amplitude = Parameter(initial_value=amplitude, unit='V',
                                   set_cmd=None, vals=vals.Numbers())

        self._connect_parameters_to_config(['amplitude'])

        if self.amplitude is not None:
            self.amplitude = self.default_amplitude

    def __repr__(self):
        try:
            properties_str = f't_start={self.t_start}, duration={self.duration}'
        except:
            properties_str = ''
        return super()._get_repr(properties_str)

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
            f"voltage at {t} s is not in the time range " \
            f"{self.t_start} s - {self.t_stop} s of pulse {self}"

        # Amplitude can only be provided in an implementation.
        # This is dependent on input/output channel properties.
        if isinstance(t, collections.Iterable):
            return np.ones(len(t)) * self.amplitude
        else:
            return self.amplitude


class TriggerWaitPulse(Pulse):
    """Pulse that wait until condition is met and then applies trigger

    Parameters:
        name: Pulse name
        t_start: Pulse start time.
        **kwargs: Additional parameters of `Pulse`.

    Note:
        Duration is fixed at 0s.

    See Also:
        `SteeredInitialization`
    """
    def __init__(self,
                 name: str = None,
                 t_start: float = None,
                 **kwargs):
        super().__init__(name=name, t_start=t_start, duration=0, **kwargs)

    def __repr__(self):
        try:
            properties_str = 't_start={self.t_start}, duration={self.duration}'
        except:
            properties_str = ''

        return super()._get_repr(properties_str)


class MeasurementPulse(Pulse):
    """Pulse that is only used to signify an acquiition

    This pulse is not directed to any interface other than the acquisition
    interface.

    Parameters:
        name: Pulse name.
        acquire: Acquire pulse (default True)
        **kwargs: Additional parameters of `Pulse`.
    """
    def __init__(self, name=None, acquire=True, **kwargs):
        super().__init__(name=name, acquire=acquire, **kwargs)

    def __repr__(self):
        try:
            properties_str = f't_start={self.t_start}, duration={self.duration}'
        except:
            properties_str = ''
        return super()._get_repr(properties_str)

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        raise NotImplementedError('Measurement pulses do not have a voltage')


class CombinationPulse(Pulse):
    """Pulse that is a combination of multiple pulse types.

    Like any other pulse, a CombinationPulse has a name, t_start and t_stop.
    t_start and t_stop are calculated and updated from the pulses that make up
    the combination.

    A CombinationPulse is itself a child of the Pulse class, therefore a
    CombinationPulse can also be used in consecutive combinations like:

    >>> CombinationPulse1 = SinePulse1 + DCPulse
    >>>CombinationPulse2 = SinePulse2 + CombinationPulse1

    Examples:
        >>> CombinationPulse = SinePulse + DCPulse
        >>> CombinationPulse = DCPulse * SinePulse

    Parameters:
        name: The name for this CombinationPulse.
        pulse1: The first pulse this combination is made up from.
        pulse2: The second pulse this combination is made up from.
        relation: The relation between pulse1 and pulse2.
            This must be one of the following:

            * '+'     :   pulse1 + pulse2
            * '-'     :   pulse1 - pulse2
            * '*'     :   pulse1 * pulse2

        **kwargs: Additional kwargs of `Pulse`.

    """

    def __init__(self,
                 name: str = None,
                 pulse1: Pulse = None,
                 pulse2: Pulse = None,
                 relation: str = None,
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


    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
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
    """Arbitrary waveform pulses that can be implemented by AWGs.

    This class allows the user to create a truly arbitrary pulse by either:
        - providing a function that converts time-stamps to waveform points
        - an array of waveform points

    The resulting AWGPulse can be sampled at different sample rates,
    interpolating between waveform points if necessary.

    Parameters:
        name: Pulse name.
        fun: The function used for calculating waveform points based on
            time-stamps.
        wf_array: Numpy array of (float) with time-stamps and waveform points.
        interpolate: Use interpolation of the wf_array.

    """

    def __init__(self,
                 name: str = None,
                 fun:Callable = None,
                 wf_array: np.ndarray = None,
                 interpolate: bool = True,
                 **kwargs):
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

    def get_voltage(self, t: Union[float, Sequence]) -> Union[float, np.ndarray]:
        """Get voltage(s) at time(s) t.

        Raises:
            AssertionError: not all ``t`` between `Pulse`.t_start and
                `Pulse`.t_stop
        """
        assert is_between(t, self.t_start, self.t_stop), \
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
