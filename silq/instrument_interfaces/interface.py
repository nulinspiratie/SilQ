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

        self.pulse_sequence = PulseSequence(allow_untargeted_pulses=False,
                                            allow_pulse_overlap=False)
        self.input_pulse_sequence = PulseSequence(
            allow_untargeted_pulses=False, allow_pulse_overlap=False)
        self.add_parameter('instrument_name',
                           parameter_class=ManualParameter,
                           initial_value=instrument_name,
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

    def has_pulse_implementation(self, pulse):
        """
        Checks if the interface has a pulse implementation that satisfies the pulse requirements.

        Args:
            pulse (Pulse): pulse for which an implementation is requested

        Returns:
            PulseImplementation if found. Otherwise None.
        """
        for pulse_implementation in self.pulse_implementations:
            if pulse_implementation.satisfies_requirements(pulse):
                return pulse_implementation
        else:
            return None

    def get_additional_pulses(self, **kwargs):
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def initialize(self):
        """
        This method gets called at the start of targeting a pulse sequence
        Returns:
            None
        """
        self.pulse_sequence.clear()
        self.input_pulse_sequence.clear()

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