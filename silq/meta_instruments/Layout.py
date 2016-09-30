from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from silq.pulses import TriggerPulse

class Layout(Instrument):
    shared_kwargs=['instruments']
    # TODO Should make into an instrument
    def __init__(self, name, instruments, **kwargs):
        super().__init__(name, **kwargs)
        self.instruments = {}
        self.instrument_interfaces = {}

        for instrument in instruments:
            self.add_instrument(instrument)

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

    def get_connections(self, output_instrument=None, output_channel=None,
                        input_instrument=None, input_channel=None):
        """
        Returns all connections that satisfy given kwargs
        Args:
            output_instrument: Connections must have output_instrument
            output_channel: Connections must have output_channel
            input_instrument: Connections must have input_instrument
            input_channel: Connections must have input_channel

        Returns:
            Connections that satisfy kwarg constraints
        """
        filtered_connections = self.connections
        if output_instrument is not None:
            filtered_connections = filter(
                lambda c: c.output_instrument == output_instrument,
                filtered_connections
            )
        if output_channel is not None:
            filtered_connections = filter(
                lambda c: c.output_channel == output_channel,
                filtered_connections
            )
        if input_instrument is not None:
            filtered_connections = filter(
                lambda c: c.input_instrument == input_instrument,
                filtered_connections
            )
        if input_channel is not None:
            filtered_connections = filter(
                lambda c: c.input_channel == input_channel,
                filtered_connections
            )
        return filtered_connections

    def add_instrument(self, instrument):
        from silq.instrument_interfaces import \
            get_instrument_interface
        instrument_interface = get_instrument_interface(instrument)
        self.instruments[instrument.name] = instrument
        self.instrument_interfaces[instrument.name] = instrument_interface
        return instrument_interface

    def get_instrument_interface(self, instrument_name):
        return self.instrument_interfaces[instrument_name]

    def get_instrument_interfaces(self):
        return self.instrument_interfaces

    def print_instrument_interfaces(self):
        # print(self.instrument_interfaces)
        # for interface in self.instrument_interfaces.values():
        #     print(interface.instrument)
        for interface in self.instrument_interfaces.values():
            print('interface: {}'.format(interface))
            instrument = interface.instrument
            print('instrument: {}'.format(instrument))
            return interface

    def get_pulse_instrument(self, pulse):
        instruments = [instrument_interface for instrument_interface in
                       self.instrument_interfaces.values() if
                       instrument_interface.get_pulse_implementation(pulse)]
        if not instruments:
            raise Exception('No instruments have an implementation for pulses '
                            '{}'.format(pulse))
        elif len(instruments) > 1:
            raise Exception('More than one instrument have an implementation '
                            'for pulses {}. Functionality to choose instrument '
                            'not yet implemented'.format(pulse))
        else:
            return instruments[0]

    def get_pulse_connection(self, pulse, instrument=None, **kwargs):
        if instrument is None:
            instrument = self.get_pulse_instrument(pulse)

        connections = self.get_connections(output_instrument=instrument,
                                           **kwargs)

        default_connections = [connection for connection in connections
                               if connection.default]
        if not default_connections:
            raise Exception('Instrument {} has connections {}, but none are '
                            'set as default'.format(instrument, connections))
        elif len(default_connections) > 1:
            raise Exception('Instrument {} has connections {}, and more than'
                            'one are set as default'.format(instrument,
                                                            connections))
        else:
            return default_connections[0]


    def target_pulse_sequence(self, pulse_sequence):
        # Clear pulses sequences of all instruments
        for instrument in self.instruments.values():
            instrument.pulse_sequence.clear()

        # Add pulses in pulse_sequence to pulse_sequences of instruments
        for pulse in pulse_sequence:
            # Get default output instrument
            instrument = self.get_pulse_instrument(pulse)
            connection = self.get_pulse_connection(pulse, instrument=instrument)
            instrument.pulse_sequence.add(pulse, connection=connection)

            # If instrument is not the main triggering instrument, add triggers
            # to each of the triggering instruments until you reach the main
            # triggering instrument.
            # TODO this should only be done in some cases, for instance if an
            # Arbstudio is the input instrument and is in stepped mode
            while instrument != self.trigger_instrument():
                connection = instrument.trigger
                # Replace instrument by its triggering instrument
                instrument = connection.output_instrument
                instrument.pulse_sequence.add(
                    TriggerPulse(t_start=pulse.t_start,
                                             connection=connection))

        # Setup each of the instruments using its pulse_sequence
        for instrument in self.instruments.values():
            instrument.setup()

        # TODO setup acquisition instrument


class Connection:
    def __init__(self, default=False):
        self.input = {}
        self.output = {}

        # TODO make default dependent on implementation
        self.default = default

class SingleConnection(Connection):
    def __init__(self, output_channel, input_channel,
                 trigger=False, acquire=False, **kwargs):
        """
        Class representing a connection between instrument channels.

        Args:
            output_channel:
            input_channel:
            trigger (bool): Sets the output channel to trigger the input
                instrument
            acquire (bool): Sets if this connection is used for acquisition
            default (bool): Sets if this connection is the default for pulses
        """
        # TODO add optionality of having multiple channels connected.
        # TODO Add mirroring of other channel.
        super().__init__(**kwargs)

        self.output['channel'] = output_channel
        self.output['instrument'] = output_channel.instrument

        self.input['channel'] = input_channel
        self.input['instrument'] = input_channel.instrument

        self.trigger = trigger
        if self.trigger:
            self.input_instrument.trigger = self

        self.acquire = acquire

class CombinedConnection(Connection):
    def __init__(self, connections, scaling_factors=None,**kwargs):
        super().__init__(**kwargs)
        self.connections = connections
        self.output['instruments'] = set([connection.output['instrument']
                                          for connection in connections])
        if len(self.output['instruments']) == 1:
            self.output['instrument'] = self.output['instruments'][0]
        else:
            raise Exception('Connections with multiple output instruments not'
                            'yet supported')
        self.output['channels'] = set([connection.output['channels']
                                       for connection in connections])
        self.input['instruments'] = set([connection.input['instrument']
                                         for connection in connections])
        self.input['channels'] = set([connection.input['channels']
                                      for connection in connections])

        if scaling_factors is None:
            scaling_factors = {connection.input['channel']: 1
                               for connection in connections}
        elif type(scaling_factors) is list:
            # Convert scaling factors to dictionary with channel keys
            scaling_factors = {connection.input['channel']: scaling_factor
                               for (connection, scaling_factor)
                               in zip(connections, scaling_factors)}
        self.scaling_factors = scaling_factors
