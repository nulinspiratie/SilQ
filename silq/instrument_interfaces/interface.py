from silq.pulses import PulseSequence

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class InstrumentInterface(Instrument):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(name=instrument_name + '_interface', **kwargs)
        self.instrument = self.find_instrument(name=instrument_name)

        self.input_channels = {}
        self.output_channels = {}

        self.channels = {}

        self._pulse_sequence = PulseSequence()
        self.add_parameter('instrument_name',
                           parameter_class=ManualParameter,
                           initial_value=instrument_name,
                           vals=vals.Anything())
        self.add_parameter('pulse_sequence',
                           get_cmd=lambda: self._pulse_sequence,
                           set_cmd=self._set_pulse_sequence,
                           vals=vals.Anything())

        self.pulse_implementations = []

    def __repr__(self):
        return '{} interface'.format(self.name)

    def get_pulse_implementation(self, pulse, is_primary=False):
        """
        Get a target implementation of a pulse if the instrument is capable
        of implementing the pulse. Whether or not it is capable depends on if
        it has an implementation that satisfies the pulse requirements
        Args:
            pulse: pulse to be targeted
            is_primary: whether or not the instrument is the primary instrument

        Returns:
            Targeted pulse if it can be implemented. Otherwise None
        """
        for pulse_implementation in self.pulse_implementations:
            if pulse_implementation.satisfies_requirements(pulse):
                return pulse_implementation.target_pulse(pulse, self,
                                                         is_primary=is_primary)
        else:
            return None

    def _set_pulse_sequence(self, val):
        """
        set function for parameter self.pulse_sequence.
        This function is created because properties/methods of the
        pulse_sequence cannot directly be
        modified/accessed from outside an interface object.
        This set command therefore allows access to properties/methods.

        Usage is as follows:
        If val is a PulseSequence, self.pulse_sequence will be replaced.
        if val is string 'sort' or 'clear', that function will be called.
        If val is a tuple (key (str), value), the action is dependent on key
        If key is 'add', then value is the pulse to be added.
        Else key should be a property, and value is its updated value
        Args:
            val: Either a PulseSequence, string, or a tuple (key, value)
                 (see above)

        Returns:
            None
        """
        if type(val) == PulseSequence:
            self._pulse_sequence = val
        elif type(val) == str:
            if val == 'clear':
                self._pulse_sequence.clear()
            elif val == 'sort':
                self._pulse_sequence.sort()
            else:
                raise Exception(
                    'Pulse sequence command {} not understood'.format(val))
        elif type(val) == tuple:
            key, value = val
            if key == 'add':
                self._pulse_sequence.add(value)
            else:
                setattr(self._pulse_sequence, key, value)
        else:
            raise Exception(
                'Pulse sequence command {} not understood'.format(val))

    def setup(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def start(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def stop(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')


class Channel:
    def __init__(self, instrument, name, id=None, input=False, output=False,
                 input_trigger=False, output_trigger=False,
                 input_TTL=False, output_TTL=False):
        self.instrument = instrument
        self.name = name
        self.id = id
        self.input = input
        self.output = output

        self.input_trigger = input_trigger
        self.output_trigger = output_trigger

        self.input_TTL = input_TTL
        self.output_TTL = output_TTL

    def __repr__(self):
        output_str = "Channel {name} (id={id})".format(name=self.name,
                                                       id=self.id)
        if self.input:
            output_str += ', input'
        if self.output:
            output_str += ', output'
        if self.input_trigger:
            output_str += ', input_trigger'
        if self.output_trigger:
            output_str += ', output_trigger'
        return output_str
