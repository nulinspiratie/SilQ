from typing import List, Dict, Any, Union, Tuple, Sequence
import numpy as np
from copy import copy, deepcopy
copy_alias = copy  # Alias for functions that have copy as a kwarg
from blinker import Signal
from matplotlib import pyplot as plt
from functools import partial

from qcodes.instrument.parameter_node import parameter
from qcodes import ParameterNode, Parameter
from qcodes.utils import validators as vals
from qcodes.instrument.parameter_node import __deepcopy__ as _deepcopy_parameterNode

__all__ = ['PulseRequirement', 'PulseSequence', 'PulseImplementation']


def __deepcopy__(self, memodict={}):
    duration = getattr(self.parameters['duration'], '_duration', None)
    self_copy = _deepcopy_parameterNode(self, memodict=memodict)
    self_copy.parameters['duration']._duration = duration
    return self_copy

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

    def __repr__(self):
        return f'{self.property} - {self.requirement}'

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


class PulseSequence(ParameterNode):
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
        duration (float): Total duration of pulse sequence. Equal to
            `Pulse`.t_stop of last pulse, unless explicitly set.
            Can be reset to t_stop of last pulse by setting to None, and will
            automatically be reset every time a pulse is added/removed.
        final_delay (Union[float, None]): Optional final delay at the end of
            the pulse sequence. The interface of the primary instrument should
            incorporate any final delay. The default is .5 ms
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
          assumed to start after the last pulse finishes, and a connection is
          made with the attribute `t_stop` of the last pulse, such that if the
          last pulse t_stop changes, t_start is changed accordingly.
        * All pulses in the pulse sequence are listened to via `Pulse`.signal.
          Any time an attribute of a pulse changes, a signal will be emitted,
          which can then be interpreted by the pulse sequence.
    """

    _deepcopy_skip_parameters = [
        'my_enabled_pulses',
        'enabled_pulses',
        'my_disabled_pulses',
        'disabled_pulses',
        'pulses',
    ]
    connection_conditions = None
    pulse_conditions = None
    default_final_delay = .5e-3
    def __init__(self,
                 pulses: list = None,
                 pulse_sequences = (),
                 name='',
                 enabled: bool = True,
                 allow_untargeted_pulses: bool = True,
                 allow_targeted_pulses: bool = True,
                 allow_pulse_overlap: bool = True,
                 final_delay: float = None):
        super().__init__(
            use_as_attributes=True,
            log_changes=False,
            simplify_snapshot=True
        )

        self.__deepcopy__ = partial(__deepcopy__, self)

        self.name = Parameter(vals=vals.Strings(), set_cmd=None, initial_value=name)
        self.full_name = Parameter(vals=vals.Strings(), initial_value=name)
        self.enabled = Parameter(vals=vals.Bool(), set_cmd=None, initial_value=enabled)

        # For PulseSequence.satisfies_conditions, we need to separate conditions
        # into those relating to pulses and to connections. We perform an import
        # here because it otherwise otherwise leads to circular imports
        if self.connection_conditions is None or self.pulse_conditions is None:
            from silq.meta_instruments.layout import connection_conditions
            from silq.pulses import pulse_conditions
            PulseSequence.connection_conditions = connection_conditions
            PulseSequence.pulse_conditions = pulse_conditions

        self.allow_untargeted_pulses = Parameter(initial_value=allow_untargeted_pulses,
                                                 set_cmd=None,
                                                 vals=vals.Bool())
        self.allow_targeted_pulses = Parameter(initial_value=allow_targeted_pulses,
                                               set_cmd=None,
                                               vals=vals.Bool())
        self.allow_pulse_overlap = Parameter(initial_value=allow_pulse_overlap,
                                             set_cmd=None,
                                             vals=vals.Bool())

        self.t_start = Parameter(unit='s', set_cmd=None, initial_value=0)
        self.duration = Parameter(unit='s', set_cmd=None)
        self.parameters['duration']._duration = None
        self.t_stop = Parameter(unit='s', set_cmd=None)

        self.final_delay = Parameter(unit='s', set_cmd=None, vals=vals.Numbers())
        if final_delay is not None:
            self.final_delay = final_delay
        else:
            self.final_delay = self.default_final_delay

        self.t_list = Parameter(initial_value=[0], snapshot_value=False)
        self.t_start_list = Parameter(initial_value=[], snapshot_value=False)
        self.t_stop_list = Parameter(snapshot_value=False)

        self.pulse_sequences = Parameter(vals=vals.Iterables(), initial_value=())

        self.my_enabled_pulses = Parameter(
            initial_value=[],
            set_cmd=None,
            vals=vals.Iterables(),
            snapshot_value=False,
            docstring='Enabled pulses that are not from a nested pulse sequence'
        )
        self.enabled_pulses = Parameter(
            initial_value=(),
            set_cmd=None,
            vals=vals.Iterables(),
            snapshot_value=False,
            docstring='Enabled pulses, including those from nested pulse sequences'
        )
        self.my_disabled_pulses = Parameter(
            initial_value=[],
            set_cmd=None,
            vals=vals.Iterables(),
            snapshot_value=False,
            docstring='Disabled pulses that are not from a nested pulse sequence'
        )
        self.disabled_pulses = Parameter(
            initial_value=(),
            set_cmd=None,
            vals=vals.Iterables(),
            snapshot_value=False,
            docstring='Disabled pulses, including those from nested pulse sequences'
        )
        self.my_pulses = Parameter(
            initial_value=[],
            vals=vals.Iterables(),
            set_cmd=None,
            snapshot_value=False,
            docstring="All pulses that are not from a nested pulse sequence"
        )
        self.pulses = Parameter(
            initial_value=(),
            vals=vals.Iterables(),
            set_cmd=None,
            docstring="All pulses, including those from nested pulse sequences"
        )

        self.flags = Parameter(
            initial_value=dict(),
            set_cmd=None,
            docstring="Optional flags to be passed onto instrument interfaces." \
                      "key must be the interface name, and value a dict."
        )
        self.modifiers = []

        # Remember last pulse of pulse sequence, to ensure t_stop of pulse sequence
        # is kept up to date via signalling
        self._last_pulse = None

        if pulse_sequences:
            self.pulse_sequences = pulse_sequences

        self.duration = None  # Reset duration to t_stop of last pulse
        # Perform a separate set to ensure set method is called
        self.pulses = pulses or ()

    @parameter
    def t_start_set_parser(self, parameter, t_start):
        if t_start is not None:
            t_start = round(t_start, 11)
        return t_start

    @parameter
    def t_start_set(self, parameter, t_start):
        parameter._latest['raw_value'] = t_start
        # Make sure all pulses have an up to date t_start and t_stop for snapshotting
        for pulse in self.pulses:
            pulse['t_start']()

        # Update t_stop
        parameter._latest['raw_value'] = t_start
        self['t_stop']()

    @parameter
    def duration_get(self, parameter):
        if parameter._duration is not None:
            return parameter._duration
        else:
            if self.enabled_pulses:
                duration = max([self.t_start] + self.t_stop_list) - self.t_start
            else:
                duration = 0

            return np.round(duration, 11)

    @parameter
    def duration_set_parser(self, parameter, duration):
        if duration is None:
            parameter._duration = None
            duration = max([self.t_start] + self.t_stop_list) - self.t_start
        else:
            parameter._duration = np.round(duration, 11)
            duration =  parameter._duration

        # Update t_stop
        parameter._latest['raw_value'] = duration
        self['t_stop']()

    @parameter
    def t_stop_get(self, parameter):
        return np.round(self.t_start + self.duration, 11)

    @parameter
    def t_start_list_get(self, parameter):
        # Use get_latest for speedup
        return sorted({pulse['t_start'].get_raw()
                       for pulse in self.enabled_pulses})

    @parameter
    def t_stop_list_get(self, parameter):
        # Use get_latest for speedup
        return sorted({pulse['t_stop'].get_raw()
                       for pulse in self.enabled_pulses})

    @parameter
    def t_list_get(self, parameter):
        # Note: Set does not work accurately when dealing with floating point numbers to remove duplicates
        # t_list = self.t_start_list + self.t_stop_list + [self.duration]
        # return sorted(list(np.unique(np.round(t_list, decimals=8)))) # Accurate to 10 ns
        return sorted(set(self.t_start_list + self.t_stop_list + [self.duration]))

    @parameter
    def pulses_set_parser(self, parameter, pulses):
        # We modify the set_parser instead of set, since we don't want to set
        # pulses to the original pulses, but to the added (copied) pulses
        self.clear(clear_pulse_sequences=False)
        added_pulses = self.quick_add(*pulses)
        self.finish_quick_add()
        return added_pulses

    @parameter
    def pulses_get(self, parameter):
        my_pulses = self.my_pulses
        nested_pulses = [
            p for pulse_sequence in self.pulse_sequences
            for p in pulse_sequence.pulses
            if pulse_sequence.enabled
        ]
        pulses = (*my_pulses, *nested_pulses)
        if my_pulses and nested_pulses:
            pulses = sorted(pulses, key=lambda pulse: pulse.t_start)
        return tuple(pulses)

    @parameter
    def enabled_pulses_get(self, parameter):
        my_enabled_pulses = self.my_enabled_pulses
        nested_enabled_pulses = [
            p for pulse_sequence in self.pulse_sequences
            for p in pulse_sequence.enabled_pulses
            if pulse_sequence.enabled
        ]
        enabled_pulses = (*my_enabled_pulses, *nested_enabled_pulses)
        if my_enabled_pulses and nested_enabled_pulses:
            enabled_pulses = sorted(enabled_pulses, key=lambda pulse: pulse.t_start)
        return tuple(enabled_pulses)

    @parameter
    def disabled_pulses_get(self, parameter):
        disabled_pulses = self.my_disabled_pulses + [
            p for pulse_sequence in self.pulse_sequences
            for p in pulse_sequence.disabled_pulses
            if pulse_sequence.enabled
        ]
        return tuple(disabled_pulses)

    @parameter
    def pulse_sequences_set(self, parameter, pulse_sequences):
        self.clear()
        self.add_pulse_sequences(*pulse_sequences)

    @parameter
    def full_name_get(self, parameter):
        if isinstance(self.parent, PulseSequence):
            return f'{self.parent.full_name}.{self.name}'
        else:
            return self.name

    @property
    def parents(self):
        parents = []
        layer = self
        while isinstance(layer.parent, PulseSequence):
            parents.append(layer.parent)
            layer = layer.parent
        return parents

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.enabled_pulses[index]
        elif isinstance(index, str):
            pulses = [p for p in self.pulses
                      if p.satisfies_conditions(name=index)]
            if pulses:
                if len(pulses) != 1:
                    raise KeyError(f"Could not find unique pulse with name "
                                   f"{index}, pulses found:\n{pulses}")
                return pulses[0]
            else:
                return super().__getitem__(index)

    def __iter__(self):
        yield from self.enabled_pulses

    def __len__(self):
        return len(self.enabled_pulses)

    def __bool__(self):
        return len(self.enabled_pulses) > 0

    def __contains__(self, item):
        if isinstance(item, str):
            return any(pulse for pulse in self.pulses
                      if item in [pulse.name, pulse.full_name])
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
        name = self.name or ''
        return f'PulseSequence {name} with {len(self.pulses)} pulses, ' \
               f'duration: {self.duration}'

    def __eq__(self, other):
        """Overwrite comparison with other (self == other).

        We want the comparison to return True if other is a pulse with the
        same attributes. This can be complicated since pulses can also be
        targeted, resulting in a pulse implementation. We therefore have to
        use a separate comparison when either is a Pulse implementation
        """
        if not isinstance(other, PulseSequence):
            # print('Other pulse sequence is not actually a pulse sequence')
            return False

        for parameter_name, parameter in self.parameters.items():
            if parameter_name == 'pulse_sequences':
                if len(parameter()) != len(other.pulse_sequences):
                    # print("Other pulse sequence contains differing number of "
                    #       "nested pulse sequences")
                    return False

                for pseq1, pseq2 in zip(parameter(), other.pulse_sequences):
                    if pseq1 != pseq2:
                        return False
            elif not parameter_name in other.parameters:
                # print(f"Pulse sequence parameter doesn't exist: {parameter_name}")
                return False
            elif parameter() != getattr(other, parameter_name):
                # print(f"Pulse sequence parameter doesn't match: {parameter_name}")
                return False
        # All parameters match
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self, *args):
        return self.copy(connect_to_config=True)

    def copy(self, connect_to_config=True):
        # Temporarily remove pulses from parameter so they won't be deepcopied
        backup = {
            key: self.parameters[key]._latest for key in [
                'pulses', 'enabled_pulses', 'disabled_pulses',
                'my_pulses', 'my_enabled_pulses', 'my_disabled_pulses',
                'pulse_sequences'
            ]
        }
        try:
            # Clear stored values of pulses
            for key in backup:
                if key.startswith('my_'):
                    self.parameters[key]._latest = {'value': [], 'raw_value': [], 'ts': None}
                else:
                    self.parameters[key]._latest = {'value': (), 'raw_value': (), 'ts': None}

            self_copy = super().__copy__()
        finally:
            # Restore pulses
            for key in backup:
                self.parameters[key]._latest = backup[key]

        self_copy._last_pulse = None

        # Add pulses (which will create copies)
        self_copy.my_pulses = [
            pulse.copy(connect_to_config=connect_to_config)
            for pulse in self.my_pulses
        ]

        # Copy nested pulse sequences
        if self.pulse_sequences:
            pulse_sequences = [
                pulse_sequence.copy(connect_to_config=connect_to_config)
                for pulse_sequence in self.pulse_sequences
            ]
            self_copy.pulse_sequences = pulse_sequences  # TODO

        # If duration is fixed (i.e. pulse_sequence.duration=val), ensure this
        # is also copied
        self_copy['duration']._duration = self['duration']._duration

        self_copy._update_enabled_disabled_pulses()

        return self_copy

    def _ipython_key_completions_(self):
        """Tab completion for IPython, i.e. pulse_sequence["p..."] """
        return [pulse.full_name for pulse in self.pulses]

    def generate(self):
        if self.pulse_sequences:
            self.clear(clear_pulse_sequences=False)

            for pulse_sequence in self.pulse_sequences:
                if pulse_sequence.enabled:
                    pulse_sequence.generate()

            # Ensure all stop times match
            self._link_pulse_sequences()

        for modifier in self.modifiers:
            modifier(self)

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=[]):
        """
        State of the pulse sequence as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac)

        Returns:
            dict: base snapshot
        """
        # Ensure the following paraeters have the latest values
        for parameter_name in ['duration', 't_list', 't_start_list', 't_stop_list']:
            self.parameters[parameter_name].get()

        snap = super().snapshot_base(update=update,
                                     params_to_skip_update=params_to_skip_update)
        snap.pop('enabled_pulses', None)
        snap.pop('my_enabled_pulses', None)
        snap.pop('my_pulses', None)

        snap['pulses'] = [pulse.snapshot(update=update,
                                         params_to_skip_update=params_to_skip_update)
                          for pulse in self.pulses]

        snap['pulse_sequences'] = [
            pulse_sequence.snapshot(
                update=update, params_to_skip_update=params_to_skip_update
            )
            for pulse_sequence in self.pulse_sequences
        ]
        return snap

    def add(self, *pulses,
            reset_duration: bool = True,
            copy: bool = True,
            nest: bool = False,
            connect: bool = True,
            set_parent:bool = None
            ):
        """Adds pulse(s) to the PulseSequence.

        Args:
            *pulses (Pulse): Pulses to add
            reset_duration: Reset duration of pulse sequence to t_stop of final
                pulse
            copy: Copy the pulse when adding to the pulse sequence
            nest: Add pulse to a nested pulse sequence if it belongs there
            connect: Connect pulse.t_start to end of previous pulse.

        Returns:
            List[Pulse]: Added pulses, which are copies of the original pulses.

        Raises:
            AssertionError: The added pulse overlaps with another pulses and
                `PulseSequence`.allow_pulses_overlap is False
            AssertionError: The added pulse is untargeted and
                `PulseSequence`.allow_untargeted_pulses is False
            AssertionError: The added pulse is targeted and
                `PulseSequence`.allow_targeted_pulses is False
            ValueError: If a pulse has no duration

        Note:
            When a pulse is added, it is first copied, to ensure that the
            original pulse remains unmodified.
            For an speed-optimized version, see `PulseSequence.quick_add`
        """
        # If set_parent is None, the parent is only set if the pulse is copied
        if set_parent is None:
            set_parent = copy

        pulses_no_duration = [pulse for pulse in pulses if pulse.duration is None]
        if pulses_no_duration:
            raise ValueError(
                'Please specify pulse duration in silq.config.pulses for the '
                'following pulses: ' + ', '.join(p.name for p in pulses_no_duration)
            )

        if copy:
            pulse_copies = []
            for pulse in pulses:
                # Copy pulse to ensure original pulse is unmodified
                # We do this before performing checks to ensure that the pulse parent
                # is set, and consequently that the t_start incorporates any nonzero
                # t_start of the pulse sequence
                pulse_copy = copy_alias(pulse)
                pulse_copy.id = None  # Remove any pre-existing pulse id
                pulse_copies.append(pulse_copy)
        else:
            pulse_copies = pulses

        added_pulses = []

        for pulse_copy in pulse_copies:
            # Check if we need to add pulse to a nested pulse sequence
            if nest and pulse_copy.parent is not None and pulse_copy.parent.full_name != self.full_name:
                nested_sequence = self.get_pulse_sequence(pulse_copy.parent.full_name)
                if nested_sequence is None:
                    raise RuntimeError(
                        f'Could not find nested pulse sequence '
                        f'{pulse_copy.parent.full_name} for {pulse_copy}'
                    )
                else:
                    # Add to nested pulse sequence. Note that we already copied pulse
                    added_pulse, = nested_sequence.quick_add(
                        pulse_copy, copy=False, set_parent=set_parent
                    )
                    added_pulses.append(added_pulse)
                    continue

            if set_parent:
                pulse_copy.parent = self

            # Perform checks to see if pulse can be added
            if (not self.allow_pulse_overlap
                    and pulse_copy.t_start is not None
                    and any(p for p in self.enabled_pulses
                            if self.pulses_overlap(pulse_copy, p))):
                overlapping_pulses = [p for p in self.enabled_pulses
                                      if self.pulses_overlap(pulse_copy, p)]
                raise AssertionError(f'Cannot add pulse {pulse_copy} because it '
                                     f'overlaps with {overlapping_pulses}')
            assert pulse_copy.implementation is not None or self.allow_untargeted_pulses, \
                f'Cannot add untargeted pulse {pulse_copy}'
            assert pulse_copy.implementation is None or self.allow_targeted_pulses, \
                f'Not allowed to add targeted pulse {pulse_copy}'

            # Check if pulse with same name exists, if so ensure unique id
            if pulse_copy.name is not None:
                pulses_same_name = self.get_pulses(name=pulse_copy.name)

                if pulses_same_name:
                    if pulses_same_name[0].id is None:
                        pulses_same_name[0].id = 0
                        pulse_copy.id = 1
                    else:
                        max_id = max(p.id for p in pulses_same_name)
                        pulse_copy.id = max_id + 1

            # If pulse does not have t_start defined, it will be attached to
            # the end of the last pulse on the same connection(_label)
            if pulse_copy.t_start is None and self.pulses:
                # Find relevant pulses that share same connection(_label)
                relevant_pulses = self.get_pulses(
                    connection=pulse_copy.connection,
                    connection_label=pulse_copy.connection_label
                )
                if relevant_pulses:
                    last_pulse = max(
                        relevant_pulses,
                        key=lambda pulse: pulse.parameters['t_stop'].raw_value
                    )
                    if connect:
                        last_pulse['t_stop'].connect(pulse_copy['t_start'], update=True)
                    else:
                        pulse_copy.t_start = last_pulse.t_stop

            if pulse_copy.t_start is None:  # No relevant pulses found
                pulse_copy.t_start = self.t_start

            self.my_pulses.append(pulse_copy)
            if pulse_copy.enabled:
                self.my_enabled_pulses.append(pulse_copy)
            else:
                self.my_disabled_pulses.append(pulse_copy)
            added_pulses.append(pulse_copy)
            # TODO attach pulsesequence to some of the pulse attributes
            pulse_copy['enabled'].connect(self._update_enabled_disabled_pulses,
                                          update=False)

        self.sort()

        if reset_duration:  # Reset duration to t_stop of last pulse
            self.duration = None

        self._update_last_pulse()

        return added_pulses

    def quick_add(self, *pulses,
                  copy: bool = True,
                  connect: bool = True,
                  reset_duration: bool = True,
                  nest=False,
                  set_parent: bool = None):
        """"Quickly add pulses to a sequence skipping steps and checks.

        This method is used in the during the `Layout` targeting of a pulse
        sequence, and should generally only be used if speed is a crucial factor.

        Note:
            When using this method, make sure to finish adding pulses with
            `PulseSequence.finish_quick_add`.

        The following steps are skipped and are performed in
        `PulseSequence.finish_quick_add`:

        - Assigning a unique pulse id if multiple pulses share the same name
        - Sorting pulses
        - Ensuring no pulses overlapped

        Args:
            *pulses: List of pulses to be added. Note that these won't be copied
                if ``copy`` is False, and so the t_start may be set
            copy: Whether to copy the pulse before applying operations
            reset_duration: Reset duration of pulse sequence to t_stop of final
                pulse
            nest: When True, if a pulse is in a nested pulse sequence, it will
                be copied to the same nested pulse sequence.
                This requires that this pulse sequence also contains a nested
                pulse sequence with the same name
            set_parent: Whether to set the pulse parent, either to the current
                or the nested pulse sequence if applicable.
                If set to None, the pulse parent is only set if the pulse is
                copied.

        Returns:
            Added pulses. If copy is False, the original pulses are returned.

        Note:
            If copy is False, the id of original pulses may be set when calling
            `PulseSequence.quick_add`.

        """
        # If set_parent is None, the parent is only set if the pulse is copied
        if set_parent is None:
            set_parent = copy

        pulses_no_duration = [pulse for pulse in pulses if pulse.duration is None]
        if pulses_no_duration:
            raise SyntaxError('Please specify pulse duration in silq.config.pulses'
                              ' for the following pulses: ' +
                              ', '.join(str(p.name) for p in pulses_no_duration))

        added_pulses = []
        for pulse in pulses:
            # Check if we need to add pulse to a nested pulse sequence
            if nest and pulse.parent is not None and pulse.parent.full_name != self.full_name:
                nested_sequence = self.get_pulse_sequence(pulse.parent.full_name)
                if nested_sequence is None:
                    raise RuntimeError(
                        f'Could not find nested pulse sequence '
                        f'{pulse.parent.full_name} for {pulse}'
                    )
                else:
                    added_pulse, = nested_sequence.quick_add(
                        pulse,
                        copy=copy,
                        connect=connect,
                        reset_duration=reset_duration,
                        set_parent=set_parent
                    )
                    added_pulses.append(added_pulse)
                    continue

            assert pulse.implementation is not None or self.allow_untargeted_pulses, \
                f'Cannot add untargeted pulse {pulse}'
            assert pulse.implementation is None or self.allow_targeted_pulses, \
                f'Not allowed to add targeted pulse {pulse}'

            if copy:
                pulse = copy_alias(pulse)

            if set_parent:
                pulse.parent = self

            # TODO set t_start if not set
            # If pulse does not have t_start defined, it will be attached to
            # the end of the last pulse on the same connection(_label)
            if pulse.t_start is None and self.pulses:
                # Find relevant pulses that share same connection(_label)
                relevant_pulses = self.get_pulses(connection=pulse.connection,
                                                  connection_label=pulse.connection_label)
                if relevant_pulses:
                    last_pulse = max(relevant_pulses,
                                     key=lambda pulse: pulse.parameters['t_stop'].raw_value)
                    pulse.t_start = last_pulse.t_stop
                    if connect:
                        last_pulse['t_stop'].connect(pulse['t_start'], update=False)
            if pulse.t_start is None:  # No relevant pulses found
                pulse.t_start = self.t_start

            self.my_pulses.append(pulse)
            added_pulses.append(pulse)
            if pulse.enabled:
                self.my_enabled_pulses.append(pulse)
            else:
                self.my_disabled_pulses.append(pulse)

            # TODO attach pulsesequence to some of the pulse attributes
            if connect:
                pulse['enabled'].connect(self._update_enabled_disabled_pulses,
                                         update=False)

        if reset_duration:  # Reset duration to t_stop of last pulse
            self.duration = None

        return added_pulses

    def finish_quick_add(self, connect=True):
        """Finish adding pulses via `PulseSequence.quick_add`

        Steps performed:

        - Sorting of pulses
        - Checking that pulses do not overlap
        - Adding unique id's to pulses in case a name is shared by pulses

        """
        try:
            self.sort()

            if not self.allow_pulse_overlap:  # Check pulse overlap
                active_pulses = []
                for pulse in self.enabled_pulses:
                    new_active_pulses = []
                    for active_pulse in active_pulses:
                        if active_pulse.t_stop <= pulse.t_start:
                            continue
                        else:
                            new_active_pulses.append(active_pulse)
                        assert not self.pulses_overlap(pulse, active_pulse), \
                            f"Pulses overlap:\n\t{repr(pulse)}\n\t{repr(active_pulse)}"

                    new_active_pulses.append(pulse)
                    active_pulses = new_active_pulses

            # Ensure all pulses have a unique full_name. This is done by attaching
            # a unique id if multiple pulses share the same name
            unique_names = set(pulse.name for pulse in self.pulses)
            for name in unique_names:
                same_name_pulses = self.get_pulses(name=name)

                # Add ``id`` if several pulses share the same name
                if len(same_name_pulses) > 1:
                    for k, pulse in enumerate(same_name_pulses):
                        pulse.id = k

            if connect:
                self._update_last_pulse()

            for pulse_sequence in self.pulse_sequences:
                pulse_sequence.finish_quick_add(connect=connect)
        except AssertionError:  # Likely error is that pulses overlap
            self.clear()
            raise

    def add_pulse_sequences(self, *pulse_sequences):
        """Add pulse sequence(s) as nested pulse sequences

        Args:
            *pulse_sequences: pulse sequences to sequentially append.

        Notes:
            - Successive pulse sequences are connected to each other such
              that if a previous pulse sequence duration changes, the subsequent
              pulse sequences are also shifted.
            - Pulse sequences are not copied. This means that if you want to
              nest a pulse sequence to multiple parent pulse sequences, you must
              manually copy the pulse sequence.
            - For each nested pulse sequence, pulse_sequence.parent is set to
             this pulse sequence.

        Raises:
             RuntimeError if current pulse sequence already contains pulses
        """
        if self.my_pulses:
            raise RuntimeError(
               'Cannot add nested pulse sequence when also containing pulses'
            )

        for pulse_sequence in pulse_sequences:
            pulse_sequence.parent = self

        self['pulse_sequences']._latest['raw_value'] = (*self.pulse_sequences, *pulse_sequences)

        self._link_pulse_sequences()

    def insert_pulse_sequence(self, index, pulse_sequence):
        """Insert a nested pulse sequence at a specific pulse sequence index

        Args:
            index: Index at which to insert pulse sequence
            pulse_sequence: Pulse sequence to insert
        """
        if self.my_pulses:
            raise RuntimeError(
               'Cannot add nested pulse sequence when also containing pulses'
            )

        pulse_sequence.parent = self

        pulse_sequences = list(self.pulse_sequences)
        pulse_sequences.insert(index, pulse_sequence)
        self['pulse_sequences']._latest['raw_value'] = tuple(pulse_sequences)

        self._link_pulse_sequences()

    def remove_pulse_sequence(self, pulse_sequence: Union["PulseSequence", int, str]):
        """Remove pulse sequence

        Args:
            pulse_sequence: Pulse sequence to remove. Can be one of the following:
                int: Index of pulse sequence to remove
                str: Name of pulse sequence to remove
                PulseSequence: pulse sequence object to remove
        """
        pulse_sequences = list(self.pulse_sequences)

        if isinstance(pulse_sequence, int):
            pulse_sequence = pulse_sequences[pulse_sequence]
        elif isinstance(pulse_sequence, str):
            pulse_sequence = next(
                pseq for pseq in pulse_sequences if pseq.name == pulse_sequence
            )

        pulse_sequences.remove(pulse_sequence)
        self['pulse_sequences']._latest['raw_value'] = tuple(pulse_sequences)

        self._link_pulse_sequences()

    def _link_pulse_sequences(self, *args):
        """

        Args:
            *args: Optional args that are ignored. Added because it allows
                signals to be connected to this function

        Returns:

        """
        # First remove all pre-existing signal receivers
        for pulse_sequence in self.pulse_sequences:
            if getattr(pulse_sequence['t_stop'], 'signal', None) is not None:
                pulse_sequence['t_stop'].signal.receivers.clear()

                # Remove any pre-existing connection to pulse_sequence.enabled
                # This may link to another parent pulse sequence
                enabled_receivers = pulse_sequence['enabled'].signal.receivers
                for receiver_key, receiver in list(enabled_receivers.items()):
                    if receiver.func_name == '_link_pulse_sequences':
                        enabled_receivers.pop(receiver_key)

            pulse_sequence['enabled'].connect(
                self._link_pulse_sequences, update=False
            )

        # Add new signal receivers
        enabled_pulse_sequences = [pseq for pseq in self.pulse_sequences if pseq.enabled]
        for k, pulse_sequence in enumerate(enabled_pulse_sequences):
            if k == 0:  # First pulse sequence
                pulse_sequence.t_start = 0
            else:
                previous_pulse_sequence = enabled_pulse_sequences[k-1]
                previous_pulse_sequence['t_stop'].connect(
                    pulse_sequence['t_start'], update=True
                )

    def remove(self, *pulses):
        """Removes `Pulse` or pulses from pulse sequence

        Args:
            pulses: Pulse(s) to remove from PulseSequence

        Raises:
            AssertionError: No unique pulse found
        """
        for pulse in pulses:
            if isinstance(pulse, str):
                pulses_same_name = [p for p in self.pulses if p.full_name==pulse]
            else:
                pulses_same_name = [p for p in self if p == pulse]

            assert len(pulses_same_name) == 1, \
                f'No unique pulse {pulse} found, pulses: {pulses_same_name}'
            pulse_same_name = pulses_same_name[0]

            self.my_pulses.remove(pulse_same_name)

            # TODO disconnect all pulse attributes
            pulse_same_name['enabled'].disconnect(self._update_enabled_disabled_pulses)

        self._update_enabled_disabled_pulses()
        self.sort()
        self.duration = None  # Reset duration to t_stop of last pulse

    def sort(self):
        """Sort pulses by `Pulse`.t_start"""
        self.my_pulses.sort(key=lambda p: p.t_start)
        self.my_enabled_pulses.sort(key=lambda p: p.t_start)

    def clear(self, clear_pulse_sequences = True):
        """Clear all pulses from pulse sequence."""
        for pulse in self.pulses:
            # TODO: remove all signal connections
            pulse['enabled'].disconnect(self._update_enabled_disabled_pulses)
        self.my_pulses.clear()
        self.my_enabled_pulses.clear()
        self.my_disabled_pulses.clear()
        if clear_pulse_sequences:
            self['pulse_sequences']._latest = {'value': (), 'raw_value': (), 'ts': None}

        self.duration = None  # Reset duration to t_stop of last pulse

    @staticmethod
    def pulses_overlap(pulse1, pulse2) -> bool:
        """Tests if pulse1 and pulse2 overlap in time and connection.

        Args:
            pulse1 (Pulse): First pulse
            pulse2 (Pulse): Second pulse

        Returns:
            True if pulses overlap

        Note:
            If either of the pulses does not have a connection, this is not tested.
        """
        if (pulse1.t_stop <= pulse2.t_start) or (pulse1.t_start >= pulse2.t_stop):
            return False
        elif pulse1.connection is not None:
            if pulse2.connection is not None:
                return pulse1.connection == pulse2.connection
            elif pulse2.connection_label is not None:
                return pulse1.connection.label == pulse2.connection_label
            else:
                return False
        elif pulse1.connection_label is not None:
            # Overlap if the pulse connection labels overlap
            labels = [pulse2.connection_label, getattr(pulse2.connection, 'label', None)]
            return pulse1.connection_label in labels
        else:
            return True

    def get_pulses(self, name=None, enabled=True, connection=None,
                   connection_label=None, **conditions):
        """Get list of pulses in pulse sequence satisfying conditions

        Args:
            name: pulse name
            enabled: Pulse must be enabled
            connection: pulse must have connection
            **conditions: Additional connection and pulse conditions.

        Returns:
            List[Pulse]: Pulses satisfying conditions

        See Also:
            `Pulse.satisfies_conditions`, `Connection.satisfies_conditions`.
        """
        pulses = self.enabled_pulses if enabled else self.pulses
        # Filter pulses by pulse conditions
        if name is not None:
            conditions['name'] = name
        pulse_conditions = {k: v for k, v in conditions.items()
                            if k in self.pulse_conditions and v is not None}
        pulses = [pulse for pulse in pulses if pulse.satisfies_conditions(**pulse_conditions)]

        # Filter pulses by pulse connection conditions
        connection_conditions = {k: v for k, v in conditions.items()
                                 if k in self.connection_conditions
                                 and v is not None}

        if connection:
            pulses = [pulse for pulse in pulses if
                      pulse.connection == connection or
                      pulse.connection_label == connection.label != None]
            return pulses  # No further filtering required
        elif connection_label is not None:
            pulses = [pulse for pulse in pulses if
                      getattr(pulse.connection, 'label', None) == connection_label or
                      pulse.connection_label == connection_label]
            return pulses # No further filtering required

        if connection_conditions:
            pulses = [pulse for pulse in pulses if
                      pulse.connection is not None and
                      pulse.connection.satisfies_conditions(**connection_conditions)]

        pulses = sorted(pulses, key=lambda pulse: pulse.t_start)
        return pulses

    def get_pulse(self, name=None, **conditions):
        """Get unique pulse in pulse sequence satisfying conditions.

        Args:
            name: Pulse name
            **conditions: Connection and pulse conditions.

        Returns:
            Pulse: Unique pulse satisfying conditions

        See Also:
            `Pulse.satisfies_conditions`, `Connection.satisfies_conditions`.

        Raises:
            RuntimeError: No unique pulse satisfying conditions
        """
        if name is not None:
            conditions['name'] = name

        pulses = self.get_pulses(**conditions)

        if not pulses:
            return None
        elif len(pulses) == 1:
            return pulses[0]
        else:
            raise RuntimeError(f'Found more than one pulse satisfiying {conditions}')

    def get_pulse_sequence(self, name):
        pulse_sequences = self.get_pulse_sequences(nested=True)

        same_full_name = [
            pulse_sequence for pulse_sequence in pulse_sequences
            if pulse_sequence.full_name == name
        ]

        if len(same_full_name) > 1:
            raise RuntimeError(f'Found multiple pulse sequences with name {name}')
        elif len(same_full_name) == 1:
            return same_full_name[0]
        else:
            same_name = [
                pulse_sequence for pulse_sequence in pulse_sequences
                if pulse_sequence.name == name
            ]

            if len(same_name) > 1:
                raise RuntimeError(f'Found multiple pulse sequences with name {name}')
            elif len(same_name) == 1:
                return same_name[0]
            else:
                return None

    def get_pulse_sequences(self, nested=True):
        """Get all pulse sequences, including nested sequences

        Args:
            nested: Include nested pulse sequences (recursively)

        Returns:
            list of pulse sequences

        """
        pulse_sequences = self.pulse_sequences
        if nested:
            for pulse_sequence in self.pulse_sequences:
                pulse_sequences += pulse_sequence.get_pulse_sequences(nested=True)

        return pulse_sequences

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
        connections = list({pulse.connection for pulse in pulses})
        assert len(connections) == 1, \
            f"No unique connection found satisfying {conditions}. " \
            f"Connections: {connections}"
        return connections[0]

    def get_transition_voltages(self,
                                pulse = None,
                                connection = None,
                                t: float = None) -> Tuple[float, float]:
        """Finds the voltages at the transition between two pulses.

        Note:
            This method can potentially cause issues, and should be avoided
            until it's better thought through

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

    def get_trace_slices(
            self,
            sample_rate: Union[int, float],
            capture_full_traces: bool,
            filter_acquire: bool = True,
            return_slice: bool = True) -> Dict[str, Union[slice, tuple]]:
        """Get the trace slices for each pulse that should be acquired

        This method is used primarily for storing metadata when saving traces.

        Args:
            sample_rate: Acquisition sample rate
            capture_full_traces: Whether the digitizer acquires traces from the
                start of the pulse_sequence (t=0) or from the first pulse with
                acquire=True.
            filter_acquire: Whether to only return the slices of pulses with
                acquire=True
            return_slice: Whether to return slice objects,
                or a tuple with (start_idx, stop_idx)

        Returns:
            A dict whose keys are pulse full names.
            Values are either a slice or tuple corresponding to
            the pulse start and stop index in an digitized trace.
        """
        pulse_slices = {}
        for k, pulse in enumerate(self.pulses):
            if not pulse.acquire and filter_acquire:  # Pulse was not acquired
                continue

            start_idx = int(pulse.t_start * sample_rate)
            stop_idx = int(pulse.t_stop * sample_rate)

            pulse_slices[pulse.full_name] = slice(start_idx, stop_idx)

        # If capture_full_traces == False, acquisition was started from first
        # pulse with acquire == True onwards.
        if not capture_full_traces:
            # Find  start time of measurement
            min_pts = min(pulse_slice.start for pulse_slice in pulse_slices.values())

            # Subtract start of acquisition from each slice
            for pulse, pulse_slice in pulse_slices.items():
                pulse_slices[pulse] = slice(
                    pulse_slice.start - min_pts, pulse_slice.stop - min_pts
                )

        if not return_slice:
            pulse_slices = {k: (v.start, v.stop) for k, v in pulse_slices.items()}

        return pulse_slices

    def plot(self, t_range=None, points=2001, subplots=False, scale_ylim=True,
             figsize=None, legend=True,
             **connection_kwargs):
        pulses = self.get_pulses(**connection_kwargs)

        connection_pulse_list = {}
        for pulse in pulses:
            if pulse.connection_label is not None:
                connection_label = pulse.connection_label
            elif pulse.connection is not None:
                if pulse.connection.label is not None:
                    connection_label = pulse.connection.label
                else:
                    connection_label = pulse.connection.output['str']
            else:
                connection_label = 'Other'

            if connection_label not in connection_pulse_list:
                connection_pulse_list[connection_label] = [pulse]
            else:
                connection_pulse_list[connection_label].append(pulse)

        if subplots:
            figsize = figsize or 10, 1.5 * len(connection_pulse_list)
            fig, axes = plt.subplots(len(connection_pulse_list), 1, sharex=True,
                                     figsize=figsize)
        else:
            figsize = figsize or (10, 4)
            fig, ax = plt.subplots(1, figsize=figsize)
            axes = [ax]

        # Generate t_list
        if t_range is None:
            t_range = (0, self.duration)
        sample_rate = (t_range[1] - t_range[0]) / points
        t_list = np.linspace(*t_range, points)

        voltages = {}
        for k, (connection_label, connection_pulses) in enumerate(
                connection_pulse_list.items()):

            connection_voltages = np.nan * np.ones(len(t_list))
            for pulse in connection_pulses:
                pulse_t_list = np.arange(pulse.t_start, pulse.t_stop,
                                         sample_rate)
                start_idx = np.argmax(t_list >= pulse.t_start)
                # Determine max_pts because sometimes there is a rounding error
                max_pts = len(connection_voltages[
                              start_idx:start_idx + len(pulse_t_list)])
                connection_voltages[
                start_idx:start_idx + len(pulse_t_list)] = pulse.get_voltage(
                    pulse_t_list[:max_pts])
            voltages[connection_label] = connection_voltages

            if subplots:
                ax = axes[k]

            ax.plot(t_list, connection_voltages, label=connection_label)
            if not subplots:
                ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlim(0, self.duration)

            if legend:
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
        return t_list, voltages, fig, axes

    def up_to_date(self) -> bool:
        """Checks if a pulse sequence is up to date or needs to be generated.

        Used by `PulseSequenceGenerator`.

        Returns:
            True by default, can be overridden in subclass.
        """
        return all(pulse_sequence.up_to_date() for pulse_sequence in self.pulse_sequences)

    def _update_enabled_disabled_pulses(self, *args):
        """Separate pulses into enabled and disabled pulses"""
        self.my_enabled_pulses = [pulse for pulse in self.my_pulses if pulse.enabled]
        self.my_disabled_pulses = [pulse for pulse in self.my_pulses if not pulse.enabled]

    def _update_last_pulse(self):
        """Attaches pulse_sequence.t_stop to t_stop of last pulse

        Called whenever pulses are added
        Notes:
            - If the t_stop of a pulse that is not the last pulse is increased
              such that it becomes the last t_stop, this will cause issues since
              it is not connected to pulse_sequence.t_stop.

        """
        if not self.my_pulses:
            return

        last_pulse = max(self.my_pulses, key=lambda p: p.t_stop)

        if self._last_pulse == last_pulse:
            return
        else:
            if self._last_pulse is not None:
                # Remove connection from previous last pulse
                self._last_pulse['t_stop'].disconnect(self['t_stop'])

            # Update last pulse and save connect
            # Using object.__setattr__ since pulses are ParameterNodes and they
            # will otherwise attach as a nested node
            object.__setattr__(self, '_last_pulse', last_pulse)
            self._last_pulse['t_stop'].connect(self['t_stop'])

    def clone_skeleton(self, pulse_sequence):
        """Clone the structure of a pulse sequence into this one

        This skeleton copy includes copies of all nested pulse sequences,
        but does not include copies of the pulses
        """
        self.clear()
        for subsequence in pulse_sequence.pulse_sequences:
            clone_subsequence = PulseSequence()
            clone_subsequence.clone_skeleton(subsequence)
            self.add_pulse_sequences(clone_subsequence)

        self.name = pulse_sequence.name
        self.duration = pulse_sequence.duration
        self.final_delay = pulse_sequence.final_delay
        self.enabled = pulse_sequence.enabled
        self.flags = copy_alias(pulse_sequence.flags)


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
                     copy=False,
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

        if copy:
            targeted_pulse = pulse.copy(connect_parameters_to_config=False)
        else:
            targeted_pulse = pulse
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
        raise NotImplementedError('PulseImplementation.implement should be '
                                  'implemented in a subclass')

def find_matching_pulse_sequence(pulse, pulse_sequence):
    """Find matching (nested) pulse sequence where a pulse should be placed in

    Args:
        pulse: pulse for which pulse sequence should be found
        pulse_sequence: Primary pulse sequence. The nested pulse sequences are
            recursively traversed until a bottom-level pulse sequence is found
            that has a t_start and t_stop encompassing the pulse

    Returns:
        Pulse sequence, either nested or the original pulse sequence that has
            been passed as an arg.
    """
    # assert pulse.t_start + 1e-12 >= pulse_sequence.t_start
    # assert pulse.t_stop - 1e-12 <= pulse_sequence.t_stop

    for subpulse_sequence in pulse_sequence.pulse_sequences:
        if not subpulse_sequence.enabled:
            continue
        elif pulse.t_start >= subpulse_sequence.t_start - 1e-12 \
                and pulse.t_stop <= subpulse_sequence.t_stop + 1e-12:
            return find_matching_pulse_sequence(pulse, subpulse_sequence)
    else:
        return pulse_sequence
