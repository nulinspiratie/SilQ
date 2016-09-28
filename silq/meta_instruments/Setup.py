from silq.meta_instruments import instrument_interfaces
# from qcodes import Instrument


class Setup:
    # Should make into an instrument
    def __init__(self, instruments):
        self.instruments = instruments
        self.instrument_interfaces = [self.add_instrument(instrument)
                                      for instrument in instruments]
        self.connections = []

    def add_connection(self, output_instrument, output_channel,
                       input_instrument, input_channel,
                       delay=0):
        connection = Connection(output_instrument, output_channel,
                                input_instrument, input_channel,
                                delay)
        self.connections += [connection]

    def add_instrument(self, instrument):
        return instrument_interfaces.get_instrument_interface(instrument)


class Connection:
    def __init__(self, output_instrument, output_channel,
                 input_instrument, input_channel,
                 delay=0):
        self.output_instrument = output_instrument
        self.output_channel = output_channel

        self.input_instrument = input_instrument
        self.input_channel = input_channel

        self.delay = delay
