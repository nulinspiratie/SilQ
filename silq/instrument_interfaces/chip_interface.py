from silq.instrument_interfaces.interface import InstrumentInterface, Channel
from silq.pulses.pulse_modules import  PulseImplementation
from silq.pulses.pulse_types import MeasurementPulse


class ChipInterface(InstrumentInterface):
    """ Interface for the Chip meta-instrument.

    The Chip and its interface don't have real functionality, but are used for
    connections that lead to/from the chip.

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
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'output': Channel(instrument_name=self.instrument_name(),
                              name='output',
                              output=True)}
        self._input_channels = {
            channel_name: Channel(instrument_name=self.instrument_name(),
                                  name=channel_name,
                                  input=True)
                               for channel_name in self.instrument.channels()}
        self._channels = {**self._input_channels, **self._output_channels}

        self.pulse_implementations = [
            MeasurementPulseImplementation(
                pulse_requirements=[]
            )
        ]

    def setup(self, **kwargs):
        """Set up instrument after layout has been targeted by pulse sequence.

        Does nothing in this case.

        Args:
            **kwargs: Ignored kwargs passed by layout.
        """
        pass

    def start(self):
        """Start instrument (ignored)."""
        pass

    def stop(self):
        """Stop instrument (ignored)."""
        pass


class MeasurementPulseImplementation(PulseImplementation):
    pulse_class = MeasurementPulse


    def implement(self, instrument, sampling_rates, threshold):
        pass
