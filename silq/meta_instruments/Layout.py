from silq.meta_instruments.instrument_interfaces import get_instrument_interface

# from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils import validators as vals


class Layout:
    # TODO Should make into an instrument
    def __init__(self, instruments):
        self.instruments = instruments
        self.instrument_interfaces = [self.add_instrument(instrument)
                                      for instrument in instruments]
        self.connections = []

        self.add_parameter('trigger_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self.instruments.keys()))

        self.add_parameter('acquisition_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self.instruments.keys()))

    def add_connection(self, output_instrument, output_channel,
                       input_instrument, input_channel, **kwargs):
        connection = Connection(output_instrument, output_channel,
                                input_instrument, input_channel, **kwargs)
        self.connections += [connection]
        return connection

    def add_instrument(self, instrument):
        instrument_interface = get_instrument_interface(instrument)
        self.instruments += [instrument]
        self.instrument_interfaces += [instrument_interface]
        return instrument_interface


    def get_pulse_instrument(self, pulse):
        pass

    def target_pulse_sequence(self, pulse_sequence):
        # Clear pulse sequences of all instruments
        for instrument in self.instruments:
            instrument.pulse_sequence.clear()

        # Add pulses in pulse_sequence to pulse_sequences of instruments
        for pulse in pulse_sequence:
            instrument = self.get_pulse_instrument(pulse)

            instrument.pulse_sequence.add(pulse, connection=instrument.trigger)

            # If instrument is not the main triggering instrument, add triggers
            # to each of the triggering instruments until you reach the main
            # triggering instrument.
            while instrument != self.trigger_instrument():
                connection = instrument.trigger
                # Replace instrument by its triggering instrument
                instrument = connection.output_instrument
                instrument.pulse_sequence.add('trigger', connection=connection)

        # Setup each of the instruments using its pulse_sequence
        for instrument in self.instruments:
            instrument.setup()

        # TODO setup acquisition instrument


class Connection:
    def __init__(self, output_instrument, output_channel,
                 input_instrument, input_channel,
                 trigger=False, acquire=False):
        """

        Args:
            output_instrument:
            output_channel:
            input_instrument:
            input_channel:
            trigger (bool): Sets the output channel to trigger the input
                instrument
            acquire (bool): Sets if this channel is used for acquisition
        """
        self.output_instrument = output_instrument
        self.output_channel = output_channel

        self.input_instrument = input_instrument
        self.input_channel = input_channel

        self.trigger = trigger
        if self.trigger:
            self.input_instrument.trigger = self

        self.acquire = acquire