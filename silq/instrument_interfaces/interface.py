from functools import partial

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

from silq.pulses import PulseSequence


class InstrumentInterface(Instrument):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(name=instrument_name + '_interface', **kwargs)
        self.instrument = self.find_instrument(name=instrument_name)

        self._input_channels = {}
        self._output_channels = {}

        self._channels = {}

        self._pulse_sequence = PulseSequence(allow_untargeted_pulses=False,
                                             allow_pulse_overlap=False)
        self._input_pulse_sequence = PulseSequence(
            allow_untargeted_pulses=False, allow_pulse_overlap=False)
        self.add_parameter('instrument_name',
                           parameter_class=ManualParameter,
                           initial_value=instrument_name,
                           vals=vals.Anything())
        self.add_parameter('pulse_sequence',
                           get_cmd=lambda: self._pulse_sequence,
                           set_cmd=partial(self._set_pulse_sequence,
                                           self._pulse_sequence),
                           vals=vals.Anything())
        self.add_parameter('input_pulse_sequence',
                           get_cmd=lambda: self._input_pulse_sequence,
                           set_cmd=partial(self._set_pulse_sequence,
                                           self._input_pulse_sequence),
                           vals=vals.Anything())

        self.pulse_implementations = []

    def __repr__(self):
        return '{} interface'.format(self.name)

    def get_channel(self, channel_name):
        """
        Retrieve a channel
        Args:
            channel_name: name of channel

        Returns:
            Channel whose name corresponds to channel_name
        """

        return self._channels[channel_name]

    def get_pulse_implementation(self, pulse, connections=None,
                                 is_primary=False):
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
                return pulse_implementation.target_pulse(
                    pulse, self, connections=connections, is_primary=is_primary)
        else:
            return None

    def _set_pulse_sequence(self, pulse_sequence, val):
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
            pulse_sequence.replace(val)
        elif type(val) == str:
            if val == 'clear':
                pulse_sequence.clear()
            elif val == 'sort':
                pulse_sequence.sort()
            else:
                raise Exception(
                    'Pulse sequence command {} not understood'.format(val))
        elif type(val) == tuple:
            key, value = val
            if key == 'add':
                pulse_sequence.add(value)
            else:
                setattr(pulse_sequence, key, value)
        else:
            raise Exception(
                'Pulse sequence command {} not understood'.format(val))

    def get_final_additional_pulses(self):
        raise NotImplementedError(
            'This method should be implemented in a subclass')

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
    def __init__(self, instrument_name, name, id=None, input=False,
                 output=False,
                 input_trigger=False, input_TTL=False, output_TTL=False):
        self.instrument = instrument_name
        self.name = name
        self.id = id
        self.input = input
        self.output = output
        self.input_trigger = input_trigger
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
        if self.input_TTL:
            output_str += ', input_TTL'
        if self.output_TTL is not None:
            output_str += ', output_TTL: {}'.format(self.output_TTL)
        return output_str


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))