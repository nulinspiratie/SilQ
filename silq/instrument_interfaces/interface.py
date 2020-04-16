from typing import Union, List, Dict, Any, Tuple
import logging

from qcodes import Instrument
from qcodes.utils import validators as vals

from silq.pulses.pulse_sequences import PulseSequence
from silq.pulses.pulse_types import Pulse


logger = logging.getLogger(__name__)


class Channel:
    """Instrument channel, specified in :class:`InstrumentInterface`

    A channel usually corresponds to a physical channel in the instrument,
    such as an input/output channel, triggering channel, etc.

    Args:
        instrument_name: Name of instrument.
        name: Channel name, usually specified on the instrument.
        id: Channel id, usually zero-based index.
        input: Channel is an input channel.
        output: Channel is an output channel.
        input_trigger: Channel is used as instrument trigger.
        input_TTL: Channel input signal must be TTL
        output_TTL: Channel output signal is TTL with (low, high) voltage
        invert: Channel signal is inverted: on is low signal, off is high signal
    """
    def __init__(self,
                 instrument_name: str,
                 name: str,
                 id: int = None,
                 input: bool = False,
                 output: bool = False,
                 input_trigger: bool = False,
                 input_TTL: bool = False,
                 output_TTL: Tuple[float, float] = False,
                 invert: bool = False):
        self.instrument = instrument_name
        self.name = name
        self.id = id
        self.input = input
        self.output = output
        self.input_trigger = input_trigger
        self.input_TTL = input_TTL
        self.output_TTL = output_TTL
        self.invert = invert

    def __repr__(self):
        output_str = f"Channel {self.name} (id={self.id})"
        if self.input:
            output_str += ', input'
        if self.output:
            output_str += ', output'
        if self.input_trigger:
            output_str += ', input_trigger'
        if self.input_TTL:
            output_str += ', input_TTL'
        if self.output_TTL is not None:
            output_str += f', output_TTL: {self.output_TTL}'
        if self.invert:
            output_str += ', invert'
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


class InstrumentInterface(Instrument):
    """ Interface between the :class:`.Layout` and instruments

    When a :class:`.PulseSequence` is targeted in the :class:`.Layout`, the
    pulses are directed to the appropriate interface. Each interface is
    responsible for translating all pulses directed to it into instrument
    commands. During the actual measurement, the instrument's operations will
    correspond to that required by the pulse sequence.

    The interface also contains a list of all available channels in the
    instrument.

    Args:
        instrument_name: name of instrument for which this is an interface

    Note:
        For a given instrument, its associated interface can be found using
            :func:`get_instrument_interface`

    """
    def __init__(self,
                 instrument_name: str,
                 **kwargs):
        # TODO: pass along actual instrument instead of name
        super().__init__(name=instrument_name + '_interface', **kwargs)
        self.instrument = self.find_instrument(name=instrument_name)

        self._input_channels = {}
        self._output_channels = {}

        self._channels = {}

        self.pulse_sequence = PulseSequence(allow_untargeted_pulses=False,
                                            allow_pulse_overlap=False)
        self.input_pulse_sequence = PulseSequence(
            allow_untargeted_pulses=False, allow_pulse_overlap=False)
        self.targeted_pulse_sequence = None
        self.targeted_input_pulse_sequence = None

        self.add_parameter('instrument_name',
                           set_cmd=None,
                           initial_value=instrument_name,
                           vals=vals.Anything())
        self.add_parameter('is_primary',
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())

        self.pulse_implementations = []

    def __repr__(self):
        return f'{self.name} interface'

    def get_channel(self, channel_name: str) -> Channel:
        """Get channel by its name.

        Args:
            channel_name: name of channel

        Returns:
            Channel whose name corresponds to channel_name
        """

        return self._channels[channel_name]

    def get_pulse_implementation(self,
                                 pulse: Pulse,
                                 connections: list = None) -> Union[Pulse, None]:
        """Get a target implementation of a pulse if it exists.

        If no implementation can be found for the pulse, or if the pulse
        properties are out of the implementation's bounds, None is returned.

        Args:
            pulse: pulse to be targeted
            connections: List of all connections in Layout, which might be
                used by the implementation.

        Returns:
            Targeted pulse if it can be implemented. Otherwise None
        """
        for pulse_implementation in self.pulse_implementations:
            if pulse_implementation.satisfies_requirements(pulse):
                return pulse_implementation.target_pulse(
                    pulse, interface=self, connections=connections)
        else:
            try:
                pulse_implementation = next(
                    pulse_implementation for pulse_implementation
                    in self.pulse_implementations
                if pulse_implementation.pulse_class == pulse.__class__)
                logger.warning(f'Pulse requirements not satisfied.\n'
                               f'Requirements: {pulse_implementation.pulse_requirements}\n'
                               f'Pulse: {repr(pulse)}')
            except:
                logger.warning(f'Could not target pulse {repr(pulse)}')
            return None

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses, empty by default.
        """
        return []

    def initialize(self):
        """
        This method gets called at the start of targeting a pulse sequence
        Returns:
            None
        """
        self.pulse_sequence = PulseSequence()
        self.input_pulse_sequence = PulseSequence()

    def setup(self,
              samples: Union[int, None] = None,
              input_connections: list = [],
              output_connections: list = [],
              repeat: bool = True,
              **kwargs) -> Dict[str, Any]:
        """Set up instrument after layout has been targeted by pulse sequence.

        Needs to be implemented in subclass.

        Args:
            samples: Number of acquisition samples.
                If None, it will use the previously set value.
            input_connections: Input :class:`.Connection` list of
                instrument, needed by some interfaces to setup the instrument.
            output_connections: Output :class:`.Connection` list of
                instrument, needed by some interfaces to setup the instrument.
            repeat: Repeat the pulse sequence indefinitely. If False, calling
                :func:`Layout.start` will only run the pulse sequence once.
            **kwargs: Additional interface-specific kwarg.

        Returns:
            setup flags (see :attr:`.Layout.flags`)

        """
        raise NotImplementedError(
            'InstrumentInterface.setup should be implemented in a subclass')

    def requires_setup(self, **kwargs) -> bool:
        if self.pulse_sequence != self.targeted_pulse_sequence:
            return True
        elif self.input_pulse_sequence != self.targeted_input_pulse_sequence:
            return True
        else:
            return False

    def start(self):
        """Start instrument

        Note:
            Acquisition instruments usually don't need to be started
        """
        raise NotImplementedError(
            'InstrumentInterface.start should be implemented in a subclass')

    def stop(self):
        """Stop instrument"""
        raise NotImplementedError(
            'InstrumentInterface.stop should be implemented in a subclass')
