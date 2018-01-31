import numpy as np
from copy import copy, deepcopy
from blinker import Signal

__all__ = ['PulseMatch', 'PulseRequirement', 'PulseSequence',
           'PulseImplementation']


class PulseMatch():
    def __init__(self, origin_pulse, origin_pulse_attr, delay=0,
                 target_pulse=None, target_pulse_attr=None):
        """
        Object used to match a pulse attribute to another pulse attribute
        Args:
            origin_pulse: Origin pulse that a target pulse is matched to
            origin_pulse_attr: Attribute of origin pulse
            delay: Offset from pulse attribute vavlue
        """
        self.origin_pulse = origin_pulse
        self.origin_pulse_attr = origin_pulse_attr
        self.delay = delay

        self.target_pulse = target_pulse
        self.target_pulse_attr = target_pulse_attr

    @property
    def value(self):
        return getattr(self.origin_pulse, self.origin_pulse_attr) + self.delay

    def __call__(self, sender, **kwargs):
        """
        Set value of target
        Args:
            sender:
            **kwargs:

        """
        if self.origin_pulse_attr in kwargs:
            setattr(self.target_pulse, self.target_pulse_attr, self)


class PulseRequirement():
    def __init__(self, property, requirement):
        self.property = property

        self.verify_requirement(requirement)
        self.requirement = requirement

    def verify_requirement(self, requirement):
        if type(requirement) is list:
            assert requirement, "Requirement must not be an empty list"
        elif type(requirement) is dict:
            assert ('min' in requirement or 'max' in requirement), \
                "Dictionary condition must have either a 'min' or a 'max'"

    def satisfies(self, pulse):
        """
        Checks if a given pulses satisfies this PulseRequirement
        Args:
            pulse: Pulse to be checked

        Returns: Bool depending on if the pulses satisfies PulseRequirement

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
            raise Exception("Cannot interpret pulses requirement: {}".format(
                self.requirement))


class PulseSequence:
    def __init__(self, pulses=[], allow_untargeted_pulses=True,
                 allow_targeted_pulses=True, allow_pulse_overlap=True,
                 final_delay=None):
        """
        A PulseSequence object is a container for pulses.
        It can be used to store untargeted or targeted pulses
        Args:
            allow_pulse_overlap:
        """

        self.allow_untargeted_pulses = allow_untargeted_pulses
        self.allow_targeted_pulses = allow_targeted_pulses
        self.allow_pulse_overlap = allow_pulse_overlap

        # These are needed to separate
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
        if not isinstance(other, PulseSequence):
            return False
        # All attributes must match
        return self._matches_attrs(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self, *args):
        return deepcopy(self)

    def _matches_attrs(self, other_pulse_sequence, exclude_attrs=[]):
            for attr in vars(self):
                if attr in exclude_attrs:
                    continue
                elif not hasattr(other_pulse_sequence, attr) \
                        or getattr(self, attr) != \
                                getattr(other_pulse_sequence, attr):
                    return False
            else:
                return True

    def _JSONEncoder(self):
        """
        Converts to JSON encoder for saving metadata

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

    def replace(self, pulse_sequence):
        """
        Replace all attributes of this pulse_sequence with another one
        Args:
            pulse_sequence: New pulse_sequence

        Returns:
            None
        """
        # Copy over all attributes from the pulse
        for attr, val in vars(pulse_sequence).items():
            setattr(self, attr, deepcopy(val))

    def add(self, *pulses):
        """
        Adds pulse(s) to the PulseSequence.
        Pulses can be a list of pulses or a single pulse.
        It performs the following additional checks before adding a pulse
        - If the pulse overlaps with other pulses and
            PulseSequence.allow_pulses_overlap is False, it raises a SyntaxError
        - If the pulses are untargeted and PulseSequence.allow_untargeted_pulses
            is False, it raises a SyntaxError
        - If the pulses are targeted and PulseSequence.allow_targeted_pulses
            is False, it raises a SyntaxError
        Args:
            *pulses: pulses to add

        Returns:
            None
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
        """
        Adds pulse(s) to the PulseSequence.
        Pulses can be a list of pulses or a single pulse.
        It performs the following additional checks before adding a pulse
        - If the pulse overlaps with other pulses and
            PulseSequence.allow_pulses_overlap is False, it raises a SyntaxError
        - If the pulses are untargeted and PulseSequence.allow_untargeted_pulses
            is False, it raises a SyntaxError
        - If the pulses are targeted and PulseSequence.allow_targeted_pulses
            is False, it raises a SyntaxError
        Args:
            pulses: pulse or list of pulses to add

        Returns:
            None
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
        self.pulses = sorted(self.pulses, key=lambda p: p.t_start)
        self.enabled_pulses = sorted(self.enabled_pulses,
                                     key=lambda p: p.t_start)

    def clear(self):
        for pulse in self.pulses:
            pulse.signal.disconnect(self._handle_signal)
        self.pulses = []
        self.enabled_pulses = []
        self.disabled_pulses = []

    def pulses_overlap(self, pulse1, pulse2):
        """
        Tests if pulse1 and pulse2 overlap in time and connection.
        If either of the pulses does not have a connection, this is not tested.
        Args:
            pulse1: First pulse
            pulse2: Second pulse

        Returns:
            True or False depending on if they overlap
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
        pulses = self.get_pulses(**conditions)

        if not pulses:
            return None
        elif len(pulses) == 1:
            return pulses[0]
        else:
            raise RuntimeError('Found more than one pulse satisfiying '
                               'conditions {}'.format(conditions))

    def get_connection(self, **conditions):

        pulses = self.get_pulses(**conditions)
        connections = [pulse.connection for pulse in pulses]
        assert len(set(connections)) == 1, "Found {} connections instead of " \
                                           "one".format(len(set(connections)))
        return connections[0]

    def get_transition_voltages(self, pulse=None, connection=None, t=None):
        """
        Finds the voltages at the transition between two pulses.
        Args:
            pulse: pulse starting at transition voltage. If not provided,
                connection and t must both be provided
            connection: connection along which the voltage transition occurs
            t: Time at which the voltage transition occurs

        Returns:
            Tuple with voltage before, after transition
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

    def get_trace_shapes(self, sample_rate, samples):
        """ Obtain diction"""

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

    def up_to_date(self):
        """ Whether a pulse sequence needs to be generated.
        Can be overridden in subclass """
        return True

class PulseImplementation:
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

    def add_pulse_requirement(self, property, requirement):
        self.pulse_requirements += [PulseRequirement(property, requirement)]

    def satisfies_requirements(self, pulse, match_class=True):
        if match_class and not self.pulse_class == pulse.__class__:
            return False
        else:
            return np.all([pulse_requirements.satisfies(pulse)
                           for pulse_requirements in self.pulse_requirements])

    def target_pulse(self, pulse, interface, **kwargs):
        """
        This tailors a PulseImplementation to a specific pulse.
        """
        if not isinstance(pulse, self.pulse_class):
            raise TypeError(f'Pulse {pulse} must be type {self.pulse_class}')

        targeted_pulse = copy(pulse)
        pulse_implementation = deepcopy(self)
        targeted_pulse.implementation = pulse_implementation
        pulse_implementation.pulse = targeted_pulse
        return targeted_pulse

    def get_additional_pulses(self, interface):
        return []

    def implement(self, **kwargs):
        raise NotImplementedError(
            'This method should be implemented in a subclass')
