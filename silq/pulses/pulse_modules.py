import numpy as np
import copy
import inspect


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
    def __init__(self, allow_untargeted_pulses=True,
                 allow_targeted_pulses=True, allow_pulse_overlap=True):
        """
        A PulseSequence object is a container for pulses.
        It can be used to store untargeted or targeted pulses
        Args:
            allow_pulse_overlap:
        """
        self.pulses = []
        self.duration = 0
        self.allow_untargeted_pulses = allow_pulse_overlap
        self.allow_targeted_pulses = allow_targeted_pulses
        self.allow_pulse_overlap = allow_pulse_overlap

        # These are needed to separate
        from silq.meta_instruments.layout import connection_conditions
        from silq.pulses import pulse_conditions
        self.connection_conditions = connection_conditions
        self.pulse_conditions = pulse_conditions
    def __getitem__(self, index):
        return self.pulses[index]

    def __len__(self):
        return len(self.pulses)

    def __bool__(self):
        return len(self.pulses) > 0

    def __repr__(self):
        output = 'PulseSequence with {} pulses, duration: {}\n'.format(
            len(self.pulses), self.duration)
        for pulse in self.pulses:
            pulse_repr = repr(pulse)
            # Add a tab to each line
            pulse_repr = '\t'.join(pulse_repr.splitlines(True))
            output += '\t' + pulse_repr + '\n'
        return output

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
            setattr(self, attr, copy.deepcopy(val))

    def add(self, pulses):
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
        if not isinstance(pulses, list):
            pulses = [pulses]

        for pulse in pulses:
            if not self.allow_pulse_overlap and \
                    any(self.pulses_overlap(pulse, p) for p in self.pulses):
                raise SyntaxError(
                    'Cannot add pulse because it overlaps.\n'
                    'Pulse 1: {}\n\nPulse2: {}'.format(
                        pulse, [p for p in self.pulses
                                if self.pulses_overlap(pulse, p)]))
            elif not isinstance(pulse, PulseImplementation) and \
                    not self.allow_untargeted_pulses:
                raise SyntaxError(
                    'Not allowed to add untargeted pulse {}'.format(pulse))
            elif isinstance(pulse, PulseImplementation) and \
                    not self.allow_targeted_pulses:
                raise SyntaxError(
                    'Not allowed to add targeted pulse {}'.format(pulse))
            else:
                self.pulses.append(pulse)

        self.sort()
        self.duration = max([pulse.t_stop for pulse in self.pulses])

    def sort(self):
        t_start_list = np.array([pulse.t_start for pulse in self.pulses])
        idx_sorted = np.argsort(t_start_list)
        self.pulses = [self.pulses[idx] for idx in idx_sorted]

        # Update duration of PulseSequence
        # self.duration = max([pulse.t_stop for pulse in self.pulses])
        return self.pulses

    def clear(self):
        self.pulses = []
        self.duration = 0

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
                and pulse1.connection['str'] != pulse2.connection['str']:
            # If the outputs are different, they don't overlap
            return False
        else:
            return True

    def get_pulses(self, **conditions):
        pulses = self.pulses

        # Filter pulses by pulse conditions
        pulse_conditions = {k: v for k, v in conditions.items()
                            if k in self.pulse_conditions}
        pulses = [pulse for pulse in pulses
                  if pulse.satisfies_conditions(**pulse_conditions)]

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
        elif connection.output['channel'].output_TTL is not None:
            # Choose pre voltage as low from TTL
            pre_voltage = connection.output['channel'].output_TTL[0]
        else:
            raise RuntimeError('Could not determine pre voltage for transition')

        post_voltage = post_pulse.get_voltage(t)

        return pre_voltage, post_voltage


class PulseImplementation:
    def __init__(self, pulse_class, pulse_requirements=[]):
        self.pulse_class = pulse_class

        # List of conditions that a pulse must satisfy to be targeted
        self.pulse_requirements = [PulseRequirement(property, condition) for
                                 (property, condition) in pulse_requirements]

        # List of pulses that need to be implemented along with this pulse.
        # An example is a triggering pulse. Each pulse has requirements in
        # pulse.connection_requirements, such as that the pulse must be provided
        # from the triggering instrument
        self.additional_pulses = []

    def add_pulse_requirement(self, property, requirement):
        self.pulse_requirements += [PulseRequirement(property, requirement)]

    def satisfies_requirements(self, pulse, match_class=True):
        if match_class and not self.pulse_class == pulse.__class__:
            return False
        else:
            return np.all([pulse_requirements.satisfies(pulse)
                           for pulse_requirements in self.pulse_requirements])

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        '''
        This tailors a PulseImplementation to a specific pulse.
        This is useful for reasons such as adding pulse_requirements such as a
        triggering pulse
        Args:
            pulse: pulse to target
            interface: instrument interface of targeted
            is_primary: whether or not the instrument is the primary instrument
        Returns:
            Copy of pulse implementation, targeted to specific pulse
        '''
        # First create a copy of this pulse implementation
        targeted_pulse = self.copy()

        # Copy over all attributes from the pulse
        for attr, val in vars(pulse).items():
            setattr(targeted_pulse, attr, copy.deepcopy(val))
        return targeted_pulse

    def implement(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')