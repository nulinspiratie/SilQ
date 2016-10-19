import numpy as np

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
    def __init__(self):
        self.pulses = []
        self.duration = 0

    def __getitem__(self, index):
        return self.pulses[index]

    def __repr__(self):
        output= 'PulseSequence with {} pulses, duration: {}\n'.format(
            len(self.pulses), self.duration)
        for pulse in self.pulses:
            pulse_repr = repr(pulse)
            # Add a tab to each line
            pulse_repr = '\t'.join(pulse_repr.splitlines(True))
            output += pulse_repr + '\n'
        return output

    def add(self, pulse):
        # TODO deal with case when pulses is a string (e.g. 'trigger')
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

    def satisfies_requirements(self, pulse):
        if not self.pulse_class == pulse.__class__:
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
            setattr(targeted_pulse, attr, val)
        return targeted_pulse

    def implement(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')