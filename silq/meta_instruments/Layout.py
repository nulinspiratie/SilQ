from silq.meta_instruments import instrument_interfaces

# from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils import validators as vals


class Layout:
    # Should make into an instrument
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
                       input_instrument, input_channel,
                       delay=0):
        connection = Connection(output_instrument, output_channel,
                                input_instrument, input_channel,
                                delay)
        self.connections += [connection]

    def add_instrument(self, instrument):
        return instrument_interfaces.get_instrument_interface(instrument)

    def get_pulse_instrument(self, pulse):
        pass

    def setup(self, pulse_sequence):
        for instrument in self.instruments:
            instrument.pulse_sequence.clear()
        for pulse in pulse_sequence:
            instrument = self.get_pulse_instrument(pulse)

            # Determine
            if instrument == self.trigger_instrument():
                instrument.pulse_sequence.add(pulse, connection=None)
            else:
                connection = instrument.trigger
                instrument.pulse_sequence.add(pulse, connection=connection)

                triggering_instrument = connection.output_instrument
                while triggering_instrument != self.trigger_instrument():
                    instrument = triggering_instrument
                    instrument.pulse_sequence.add('trigger', )




class Connection:
    def __init__(self, output_instrument, output_channel,
                 input_instrument, input_channel,
                 trigger=False,
                 delay=0):
        self.output_instrument = output_instrument
        self.output_channel = output_channel

        self.input_instrument = input_instrument
        self.input_channel = input_channel

        self.trigger = trigger
        if self.trigger:
            self.input_instrument.trigger = self

        self.delay = delay
