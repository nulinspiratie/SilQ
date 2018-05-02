from typing import List, Dict, Any, Union, Tuple
import numpy as np
from copy import copy, deepcopy
from blinker import Signal
from matplotlib import pyplot as plt

__all__ = ['PulseRequirement', 'PulseSequence', 'PulseImplementation']


class PulseRequirement():
    """`Pulse` attribute requirement for a `PulseImplementation`

    This class is used in Interfaces when registering a `PulseImplementation`,
    to impose additional constraints for implementing the pulse.

    The class is never directly instantiated, but is instead created from a dict
    passed to the ``pulse_requirements`` kwarg of a `PulseImplementation`.

    Example:
        For an AWG can apply sine pulses, but only up to its Nyquist limit
        ``max_frequency``, the following implementation is used:

        >>> SinePulseImplementation(
                pulse_requirements=[('frequency', {'max': max_frequency})])

    Args:
        property: Pulse attribute for which to place a constraint.
        requirement: Requirement that a property must satisfy.

            * If a dict, allowed keys are ``min`` and ``max``, the value being
              the minimum/maximum value.
            * If a list, the property must be an element in the list.
    """
    def __init__(self,
                 property: str,
                 requirement: Union[list, Dict[str, Any]]):
        self.property = property

        self.verify_requirement(requirement)
        self.requirement = requirement

    def verify_requirement(self, requirement):
        """Verifies that the requirement is valid.

        A valid requirement is either a list, or a dict with keys ``min`` and/or
        ``max``.

        Raises:
            AssertionError: Requirement is not valid.
        """
        if type(requirement) is list:
            assert requirement, "Requirement must not be an empty list"
        elif type(requirement) is dict:
            assert ('min' in requirement or 'max' in requirement), \
                "Dictionary condition must have either a 'min' or a 'max'"

    def satisfies(self, pulse) -> bool:
        """Checks if a given pulses satisfies this PulseRequirement.

        Args:
            pulse: Pulse to be verified.

        Returns:
            True if pulse satisfies PulseRequirement.

        Raises:
            Exception: Pulse requirement cannot be interpreted.

        """
        property_value = getattr(pulse, self.property)

        # Test for condition
        if type(self.requirement) is dict:
            # requirement contains min and/or max
            if 'min' in self.requirement and \
                            property_value < self.requirement['min']:
                return False
            elif 'max' in self.requirement and \
                            property_value > self.requirement['max']:
                return False
            else:
                return True
        elif type(self.requirement) is list:
            if property_value not in self.requirement:
                return False
            else:
                return True
        else:
            raise Exception(
                "Cannot interpret pulses requirement: {self.requirement}")


class PulseSequence:
    """`Pulse` container that can be targeted in the `Layout`.

    It can be used to store untargeted or targeted pulses.

    If multiple pulses with the same name are added, `Pulse`.id is set for the
    pulses sharing the same name, starting with 0 for the first pulse.

    **Retrieving pulses**
        To retrieve a pulse with name 'read':

        >>> pulse_sequence['read']
        >>> pulse_sequence.get_pulse(name='read')

        Both methods work, but the latter is more versatile, as it also allows
        filtering of pulses by discriminants other than name.

        If there are multiple pulses with the same name, the methods above will
        raise an error because there is no unique pulse with name ``read``.
        Instead, the `Pulse`.id must also be passed to discriminate the pulses:

        >>> pulse_sequence['read[0]']
        >>> pulse_sequence.get_pulse(name='read', id=0)

        Both methods return the first pulse added whose name is 'read'.

    **Iterating over pulses**
        Pulses in a pulse sequence can be iterated over via:

        >>> for pulse in pulse_sequence:
        >>>     # perform actions

        This will return the pulses sorted by `Pulse`.t_start.
        Pulses for which `Pulse`.enabled is False are ignored.

    **Checking if pulse sequence contains a pulse**
        Pulse sequences can be treated similar to a list, and so checking if a
        pulse exists in a list is done as such:

        >>> pulse in pulse_sequence

        Note that this does not compare object equality, but only checks if all
        attributes match.

    **Checking if a pulse sequence contains pulses**
        Checking if a pulse sequence contains pulses is similar to a list:

        >>> if pulse_sequence:
        >>>     # pulse_sequence contains pulses

    **Targeting a pulse sequence in the `Layout`**
        A pulse sequence can be targeted in the layout, which will distribute
        the pulses among it's `InstrumentInterface` such that the pulse sequence
        is executed. Targeting of a pulse sequence is straightforward:

        >>> layout.pulse_sequence = pulse_sequence

        After this, the instruments can be configured via `Layout.setup`.


    Parameters:
        pulses (List[Pulse]): `Pulse` list to place in PulseSequence.
            Pulses can also be added later using `PulseSequence.add`.
        allow_untargeted_pulses (bool): Allow untargeted pulses (without
            corresponding `Pulse`.implementation) to be added to PulseSequence.
            `InstrumentInterface`.pulse_sequence should have this unchecked.
        allow_targeted_pulses (bool): Allow targeted pulses (with corresponding
            `Pulse`.implementation) to be added to PulseSequence.
            `InstrumentInterface`.pulse_sequence should have this checked.
        allow_pulse_overlap (bool): Allow pulses to overlap in time. If False,
            an error will be raised if a pulse is added that overlaps in time.
            If pulse has a `Pulse`.connection, an error is only raised if
            connections match as well.
        final_delay (float): Final delay in pulse sequence after final pulse is
            finished.
        duration (float): Total duration of pulse sequence. Equal to
            `Pulse`.t_stop of last pulse, plus any `PulseSequence`.final_delay.
        enabled_pulses (List[Pulse]): `Pulse` list with `Pulse`.enabled True.
            Updated when a pulse is added or `Pulse`.enabled is changed.
        disabled_pulses (List[Pulse]): Pulse list with `Pulse`.enabled False.
            Updated when a pulse is added or `Pulse`.enabled is changed.
        t_start_list (List[float]): `Pulse`.t_start list for all enabled pulses.
            Can contain duplicates if pulses share the same `Pulse`.t_start.
        t_stop_list (List[float]): `Pulse`.t_stop list for all enabled pulses.
            Can contain duplicates if pulses share the same `Pulse`.t_stop.
        t_list (List[float]): Combined list of `Pulse`.t_start and
            `Pulse`.t_stop for all enabled pulses. Does not contain duplicates.

    Notes:
        * If pulses are added without `Pulse`.t_start defined, the pulse is
          assumed to start after the last pulse finishes, and a `PulseMatch`
          is created to sync it's `Pulse`.t_start with `Pulse`.t_stop of the
          previous pulse.
        * All pulses in the pulse sequence are listened to via `Pulse`.signal.
          Any time an attribute of a pulse changes, a signal will be emitted,
          which can then be interpreted by the pulse sequence.
    """
    def __init__(self,
                 pulses: list = [],
                 allow_untargeted_pulses: bool = True,
                 allow_targeted_pulses: bool = True,
                 allow_pulse_overlap: bool = True,
                 final_delay: float = None):
        self.allow_untargeted_pulses = allow_untargeted_pulses
        self.allow_targeted_pulses = allow_targeted_pulses
        self.allow_pulse_overlap = allow_pulse_overlap

        # These are needed to separate pulse and connection conditions
        from silq.meta_instruments.layout import connection_conditions
        from silq.pulses import pulse_conditions
        self.connection_conditions = connection_conditions
        self.pulse_conditions = pulse_conditions

        self._duration = None
        self.final_delay = final_delay

        self.pulses = []
        self.enabled_pulses = []
        self.disabled_pulses = []

        if pulses:
            self.add(*pulses)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.enabled_pulses[index]
        elif isinstance(index, str):
            pulses = [p for p in self.pulses
                      if p.satisfies_conditions(name=index)]
            assert len(pulses) == 1, f"Could not find unique pulse with name " \
                                     f"{index}, pulses found:\n{pulses}"
            return pulses[0]

    def __len__(self):
        return len(self.enabled_pulses)

    def __bool__(self):
        return len(self.enabled_pulses) > 0

    def __contains__(self, item):
        if isinstance(item, str):
            pulses = [pulse for pulse in self.pulses if pulse.name == item]
            return len(pulses) > 0
        else:
            return item in self.pulses

    def __repr__(self):
        output = str(self) + '\n'
        for pulse in self.enabled_pulses:
            pulse_repr = repr(pulse)
            # Add a tab to each line
            pulse_repr = '\t'.join(pulse_repr.splitlines(True))
            output += '\t' + pulse_repr + '\n'

        if self.disabled_pulses:
            output += '\t\n\tDisabled pulses:\n'
            for pulse in self.disabled_pulses:
                pulse_repr = repr(pulse)
                # Add a tab to each line
                pulse_repr = '\t'.join(pulse_repr.splitlines(True))
                output += '\t' + pulse_repr + '\n'
        return output

    def __str__(self):
        return f'PulseSequence with {len(self.pulses)} pulses, ' \
               f'duration: {self.duration}'

    def __eq__(self, other):
        """Overwrite comparison with other (self == other).

        We want the comparison to return True if other is a pulse with the
        same attributes. This can be complicated since pulses can also be
        targeted, resulting in a pulse implementation. We therefore have to
        use a separate comparison when either is a Pulse implementation
        """
        if not isinstance(other, PulseSequence):
            return False
        # All attributes must match
        return self._matches_attrs(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self, *args):
        return deepcopy(self)

    def _ipython_key_completions_(self):
        """Tab completion for IPython, i.e. pulse_sequence["p..."] """
        return [pulse.full_name for pulse in self.pulses]

    def _matches_attrs(self,
                       other_pulse_sequence: 'PulseSequence',
                       exclude_attrs: List[str] = []) -> bool:
        """Checks if another pulse sequence is the same (same attributes).

        This is used when comparing pulse sequences. Usually pulse sequences
        are equal if their attributes are equal, not object equality.
        This includes pulses.

        Args:
            other_pulse_sequence: Pulse sequence to compare.
            exclude_attrs: Attributes to skip.

        Returns:
            True if all attributes are equal (except those in ``exclude_attrs``)
        """
        for attr in vars(self):
            if attr in exclude_attrs:
                continue
            elif not hasattr(other_pulse_sequence, attr) \
                    or getattr(self, attr) != getattr(other_pulse_sequence, attr):
                return False
        else:
            return True

    def _JSONEncoder(self) -> dict:
        """Converts to JSON encoder for saving metadata

        Returns:
            JSON dict
        """
        return {
            'allow_untargeted_pulses': self.allow_untargeted_pulses,
            'allow_targeted_pulses': self.allow_targeted_pulses,
            'allow_pulse_overlap': self.allow_pulse_overlap,
            'pulses': [pulse._JSONEncoder() for pulse in self.pulses]
        }

    def _handle_signal(self, pulse, **kwargs):
        """Handler for pulse signal (only handles attribute ``enabled``)"""
        key, val = kwargs.popitem()
        if key == 'enabled':
            if val is True:
                if pulse not in self.enabled_pulses:
                    self.enabled_pulses.append(pulse)
                if pulse in self.disabled_pulses:
                    self.disabled_pulses.remove(pulse)
            elif val is False:
                if pulse in self.enabled_pulses:
                    self.enabled_pulses.remove(pulse)
                if pulse not in self.disabled_pulses:
                    self.disabled_pulses.append(pulse)

    @property
    def duration(self):
        if self._duration is not None:
            return self._duration
        elif self.enabled_pulses:
            duration = max(pulse.t_stop for pulse in self.enabled_pulses)
        else:
            duration = 0

        if self.final_delay is not None:
            duration += self.final_delay

        return np.round(duration, 11)

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    @property
    def t_start_list(self):
        return sorted({pulse.t_start for pulse in self.enabled_pulses})

    @property
    def t_stop_list(self):
        return sorted({pulse.t_stop for pulse in self.enabled_pulses})

    @property
    def t_list(self):
        return sorted(set(self.t_start_list + self.t_stop_list + [self.duration]))

    def add(self, *pulses):
        """Adds pulse(s) to the PulseSequence.

        Args:
            *pulses (Pulse): Pulses to add

        Returns:
            List[Pulse]: Added pulses, which are copies of the original pulses.

        Raises:
            SyntaxError: The added pulse overlaps with another pulses and
                `PulseSequence`.allow_pulses_overlap is False
            SyntaxError: The added pulse is untargeted and
                `PulseSequence`.allow_untargeted_pulses is False
            SyntaxError: The added pulse is targeted and
                `PulseSequence`.allow_targeted_pulses is False

        Note:
            When a pulse is added, it is first copied, to ensure that the
            original pulse remains unmodified.
        """
        added_pulses = []

        for pulse in pulses:

            if not self.allow_pulse_overlap and \
                    any(self.pulses_overlap(pulse, p)
                        for p in self.enabled_pulses):
                raise SyntaxError(
                    'Cannot add pulse because it overlaps.\n'
                    'Pulse 1: {}\n\nPulse2: {}'.format(
                        pulse, [p for p in self.enabled_pulses
                                if self.pulses_overlap(pulse, p)]))
            elif pulse.implementation is None and \
                    not self.allow_untargeted_pulses:
                raise SyntaxError(f'Cannot add untargeted pulse {pulse}')
            elif pulse.implementation is not None and \
                    not self.allow_targeted_pulses:
                raise SyntaxError(f'Not allowed to add targeted pulse {pulse}')
            elif pulse.duration is None:
                raise SyntaxError(f'Pulse {pulse} duration must be specified')
            else:
                # Check if pulse with same name exists
                pulse_copy = copy(pulse)
                pulse_copy.id = None # Remove any pre-existing pulse id
                if pulse.name is not None:
                    pulses_same_name = self.get_pulses(name=pulse.name)
                    if pulses_same_name:
                        # Ensure id is unique
                        if pulses_same_name[0].id is None:
                            pulses_same_name[0].id = 0
                            pulse_copy.id = 1
                        else:
                            max_id = max(p.id for p in pulses_same_name)
                            pulse_copy.id = max_id + 1

                if pulse_copy.t_start is None:
                    if self: # There exist pulses in this pulse_sequence
                        # Add last pulse of this pulse_sequence to the pulse
                        # the previous_pulse.t_stop will be used as t_start
                        t_stop_max = max(self.t_stop_list)
                        last_pulse = self.get_pulses(t_stop=t_stop_max,
                                                     enabled=True)[-1]

                        pulse_copy.t_start = PulseMatch(last_pulse, 't_stop')
                    else:
                        pulse_copy.t_start = 0
                self.pulses.append(pulse_copy)
                pulse_copy.signal.connect(self._handle_signal)
                added_pulses.append(pulse_copy)

                if pulse_copy.enabled:
                    self.enabled_pulses.append(pulse_copy)
                else:
                    self.disabled_pulses.append(pulse_copy)
        self.sort()

        return added_pulses

    def remove(self, *pulses):
        """Removes `Pulse` or pulses from pulse sequence

        Args:
            pulses: Pulse(s) to remove from PulseSequence

        Raises:
            AssertionError: No unique pulse found
        """
        for pulse in pulses:
            if isinstance(pulse, str):
                pulses_name = [p for p in self.pulses if p.full_name==pulse]
                assert len(pulses_name) == 1, f'No unique pulse {pulse} found' \
                                              f', pulses: {len(pulses_name)}'
                pulse = pulses_name[0]
            else:
                pulses = [p for p in self if p == pulse]
                assert len(pulses) == 1, f'No unique pulse {pulse} found' \
                                         f', pulses: {pulses}'
            self.pulses.remove(pulse)
            if pulse.enabled:
                self.enabled_pulses.remove(pulse)
            else:
                self.disabled_pulses.remove(pulse)
            pulse.signal.disconnect(self._handle_signal)
        self.sort()

    def sort(self):
        """Sort pulses by `Pulse`.t_start"""
        self.pulses = sorted(self.pulses, key=lambda p: p.t_start)
        self.enabled_pulses = sorted(self.enabled_pulses,
                                     key=lambda p: p.t_start)

    def clear(self):
        """Clear all pulses from pulse sequence."""
        for pulse in self.pulses:
            pulse.signal.disconnect(self._handle_signal)
        self.pulses = []
        self.enabled_pulses = []
        self.disabled_pulses = []

    def pulses_overlap(self, pulse1, pulse2) -> bool:
        """Tests if pulse1 and pulse2 overlap in time and connection.

        Args:
            pulse1 (Pulse): First pulse
            pulse2 (Pulse): Second pulse

        Returns:
            True if pulses overlap

        Note:
            If either of the pulses does not have a connection, this is not tested.
        """
        if (pulse1.t_stop <= pulse2.t_start) or \
                (pulse1.t_start >= pulse2.t_stop):
            #
            return False
        elif pulse1.connection is not None and pulse2.connection is not None \
                and pulse1.connection != pulse2.connection:
            # If the outputs are different, they don't overlap
            return False
        else:
            return True

    def get_pulses(self, enabled=True, **conditions):
        """Get list of pulses in pulse sequence satisfying conditions

        Args:
            enabled: Pulse must be enabled
            **conditions: Additional connection and pulse conditions.

        Returns:
            List[Pulse]: Pulses satisfying conditions

        See Also:
            `Pulse.satisfies_conditions`, `Connection.satisfies_conditions`.
        """
        pulses = self.pulses
        # Filter pulses by pulse conditions
        pulse_conditions = {k: v for k, v in conditions.items()
                            if k in self.pulse_conditions + ['pulse_class']}
        pulses = [pulse for pulse in pulses
                  if pulse.satisfies_conditions(
                enabled=enabled, **pulse_conditions)]

        # Filter pulses by pulse connection conditions
        connection_conditions = {k: v for k, v in conditions.items()
                                 if k in self.connection_conditions}
        if connection_conditions:
            pulses = [pulse for pulse in pulses if
                      pulse.connection is not None and
                      pulse.connection.satisfies_conditions(
                          **connection_conditions)]

        return pulses

    def get_pulse(self, **conditions):
        """Get unique pulse in pulse sequence satisfying conditions.

        Args:
            **conditions: Connection and pulse conditions.

        Returns:
            Pulse: Unique pulse satisfying conditions

        See Also:
            `Pulse.satisfies_conditions`, `Connection.satisfies_conditions`.

        Raises:
            RuntimeError: No unique pulse satisfying conditions
        """
        pulses = self.get_pulses(**conditions)

        if not pulses:
            return None
        elif len(pulses) == 1:
            return pulses[0]
        else:
            raise RuntimeError(f'Found more than one pulse satisfiying {conditions}')

    def get_connection(self, **conditions):
        """Get unique connections from any pulse satisfying conditions.

        Args:
            **conditions: Connection and pulse conditions.

        Returns:
            Connection: Unique Connection satisfying conditions

        See Also:
            `Pulse.satisfies_conditions`, `Connection.satisfies_conditions`.

        Raises:
            AssertionError: No unique connection satisfying conditions.
        """
        pulses = self.get_pulses(**conditions)
        connections = [pulse.connection for pulse in pulses]
        assert len(set(connections)) == 1, "Found {} connections instead of " \
                                           "one".format(len(set(connections)))
        return connections[0]

    def get_transition_voltages(self,
                                pulse = None,
                                connection = None,
                                t: float = None) -> Tuple[float, float]:
        """Finds the voltages at the transition between two pulses.

        Args:
            pulse (Pulse): Pulse starting at transition voltage. If not
                provided, ``connection`` and ``t`` must both be provided.
            connection (Connection): connection along which the voltage
                transition occurs
            t (float): Time at which the voltage transition occurs.

        Returns:
            (Voltage before transition, voltage after transition)
        """
        if pulse is not None:
            post_pulse = pulse
            connection = pulse.connection
            t = pulse.t_start
        elif connection is not None and t is not None:
            post_pulse = self.get_pulse(connection=connection, t_start=t)
        else:
            raise TypeError('Not enough arguments provided')

        # Find pulses thar stop sat t. If t=0, the pulse before this
        #  will be the last pulse in the sequence
        pre_pulse = self.get_pulse(connection=connection,
                                   t_stop=(self.duration if t == 0 else t))
        if pre_pulse is not None:
            pre_voltage = pre_pulse.get_voltage(self.duration if t == 0 else t)
        elif connection.output['channel'].output_TTL:
            # Choose pre voltage as low from TTL
            pre_voltage = connection.output['channel'].output_TTL[0]
        else:
            raise RuntimeError('Could not determine pre voltage for transition')

        post_voltage = post_pulse.get_voltage(t)

        return pre_voltage, post_voltage

    def get_trace_shapes(self,
                         sample_rate: int,
                         samples: int):
        """ Get dictionary of trace shapes for given sample rate and samples

        Args:
            sample_rate: Acquisition sample rate
            samples: acquisition samples.

        Returns:
            Dict[str, tuple]:
            {`Pulse`.full_name: trace_shape}

        Note:
            trace shape depends on `Pulse`.average
        """

        shapes = {}
        for pulse in self:
            if not pulse.acquire:
                continue
            pts = round(pulse.duration * sample_rate)
            if pulse.average == 'point':
                shape = (1,)
            elif pulse.average == 'trace':
                shape = (pts, )
            else:
                shape = (samples, pts)

            shapes[pulse.full_name] = shape

        return shapes

    def plot(self, output_arg=None, output_channel=None, dt=1e-6,
             subplots=False, scale_ylim=True):
        pulses = self.get_pulses(output_arg=output_arg,
                                 output_channel=output_channel)

        connections = {pulse.connection for pulse in pulses}
        connections = sorted(connections,
                             key=lambda connection: connection.output['str'])
        if subplots:
            fig, axes = plt.subplots(len(connections), sharex=True,
                                     figsize=(10, 1.5 * len(connections)))
        else:
            fig, axes = plt.subplots(1, figsize=(10, 4))

        t_list = np.arange(0, self.duration, dt)
        voltages = {}
        for k, connection in enumerate(connections):
            connection_pulses = [pulse for pulse in pulses if
                                 pulse.connection == connection]
            #         print('connection_pulses', connection_pulses)
            connection_voltages = np.nan * np.ones(len(t_list))
            for pulse in connection_pulses:
                pulse_t_list = np.arange(pulse.t_start, pulse.t_stop, dt)
                start_idx = np.argmax(t_list >= pulse.t_start)
                # Determine max_pts because sometimes there is a rounding error
                max_pts = len(connection_voltages[
                              start_idx:start_idx + len(pulse_t_list)])
                #             print('pulse', pulse, ', start_idx', start_idx, ', len(pulse_t_list)', len(pulse_t_list))
                connection_voltages[
                start_idx:start_idx + len(pulse_t_list)] = pulse.get_voltage(
                    pulse_t_list[:max_pts])
            voltages[connection.output['str']] = connection_voltages

            ax = axes[k] if isinstance(axes, np.ndarray) else axes
            ax.plot(t_list, connection_voltages, label=connection.output["str"])
            if not subplots:
                ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlim(0, self.duration)
            ax.legend()

        if scale_ylim:
            min_voltage = np.nanmin(np.concatenate(tuple(voltages.values())))
            max_voltage = np.nanmax(np.concatenate(tuple(voltages.values())))
            voltage_difference = max_voltage - min_voltage
            for ax in axes:
                ax.set_ylim(min_voltage - 0.05 * voltage_difference,
                            max_voltage + 0.05 * voltage_difference)

        fig.tight_layout()
        if subplots:
            fig.subplots_adjust(hspace=0)
        return t_list, voltages

    def up_to_date(self) -> bool:
        """Checks if a pulse sequence is up to date or needs to be generated.

        Used by `PulseSequenceGenerator`.

        Returns:
            True by default, can be overridden in subclass.
        """
        return True


class PulseImplementation:
    """`InstrumentInterface` implementation for a `Pulse`.

    Each `InstrumentInterface` should have corresponding pulse implementations
    for the pulses it can output. These should be subclasses of the
    `PulseImplementation`.

    When a `PulseSequence` is targeted in the Layout, each `Pulse` is directed
    to the relevant `InstrumentInterface`, which will call target the pulse
    using the corresponding PulseImplementation. During `Pulse` targeting,
    a copy of the pulse is made, and the PulseImplementation is added to
    `Pulse`.implementation.

    **Creating a PulseImplementation**
        A PulseImplementation is specific for a certain `Pulse`, which should
        be defined in `PulseImplementation`.pulse_class.

        A `PulseImplementation` subclass may override the following methods:

        * `PulseImplementation.target_pulse`
        * `PulseImplementation.get_additional_pulses`
        * `PulseImplementation.implement`

    Args:
        pulse_requirements: Requirements that pulses must satisfy to allow
            implementation.
    """
    pulse_config = None
    pulse_class = None

    def __init__(self, pulse_requirements=[]):
        self.signal = Signal()
        self._connected_attrs = {}
        self.pulse = None

        # List of conditions that a pulse must satisfy to be targeted
        self.pulse_requirements = [PulseRequirement(property, condition) for
                                 (property, condition) in pulse_requirements]

    def __ne__(self, other):
        return not self.__eq__(other)

    def _matches_attrs(self, other_pulse, exclude_attrs=[]):
        for attr in list(vars(self)):
            if attr in exclude_attrs:
                continue
            elif not hasattr(other_pulse, attr) \
                    or getattr(self, attr) != getattr(other_pulse, attr):
                return False
        else:
            return True

    def add_pulse_requirement(self,
                              property: str,
                              requirement: Union[list, Dict[str, Any]]):
        """Add requirement that any pulse must satisfy to be targeted"""
        self.pulse_requirements += [PulseRequirement(property, requirement)]

    def satisfies_requirements(self,
                               pulse,
                               match_class: bool = True):
        """Checks if a pulse satisfies pulse requirements

        Args:
            pulse (Pulse): Pulse that is checked
            match_class: Pulse class must match
                `PulseImplementation`.pulse_class
        """
        if match_class and not self.pulse_class == pulse.__class__:
            return False
        else:
            return np.all([pulse_requirements.satisfies(pulse)
                           for pulse_requirements in self.pulse_requirements])

    def target_pulse(self,
                     pulse,
                     interface,
                     connections: list,
                     **kwargs):
        """Tailors a PulseImplementation to a specific pulse.

        Targeting happens in three stages:

        1. Both the pulse and pulse implementation are copied.
        2. `PulseImplementation` of the copied pulse is set to the copied
           pulse implementation, and `PulseImplementation`.pulse is set to the
           copied pulse. This way, they can both reference each other.
        3. The targeted pulse is returned

        Args:
            pulse (Pulse): Pulse to be targeted.
            interface (InstrumentInterface) interface to which this
                PulseImplementation belongs.
            connections (List[Connection]): All connections in `Layout`.
            **kwargs: Additional unused kwargs

        Raises:
            TypeError: Pulse class does not match
                `PulseImplementation`.pulse_class

        """
        if not isinstance(pulse, self.pulse_class):
            raise TypeError(f'Pulse {pulse} must be type {self.pulse_class}')

        targeted_pulse = copy(pulse)
        pulse_implementation = deepcopy(self)
        targeted_pulse.implementation = pulse_implementation
        pulse_implementation.pulse = targeted_pulse
        return targeted_pulse

    def get_additional_pulses(self, interface):
        """Provide any additional pulses needed such as triggering pulses

        The additional pulses can be requested should usually have
        `Pulse`.connection_conditions specified to ensure that the pulse is
        sent to the right connection.

        Args:
            interface (InstrumentInterface): Interface to which this
                PulseImplementation belongs

        Returns:
            List[Pulse]: List of additional pulses needed.
        """
        return []

    def implement(self, *args, **kwargs) -> Any:
        """Implements a targeted pulse for an InstrumentInterface.

        This method is called during `InstrumentInterface.setup`.

        Implementation of a targeted pulse is very dependent on the interface.
        For an AWG, this method may return a list of waveform points.
        For a triggering source, this method may return the triggering time.
        In very simple cases, this method may not even be necessary.

        Args:
            *args: Interface-specific args to use
            **kwargs: Interface-specific kwargs to use

        Returns:
            Instrument-specific return values.

        See Also:
            Other interface source codes may serve as a guide for this method.
        """
        raise NotImplementedError('This method should be implemented in a subclass')
