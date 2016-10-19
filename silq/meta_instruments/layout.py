from silq.pulses import TriggerPulse
from silq.instrument_interfaces import Channel

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class Layout(Instrument):
    shared_kwargs = ['instrument_interfaces']

    def __init__(self, name, instrument_interfaces, **kwargs):
        super().__init__(name, **kwargs)

        # Add interfaces for each instrument to self.instruments
        self._interfaces = {interface.instrument_name(): interface
                            for interface in instrument_interfaces}

        self.connections = []

        self.add_parameter('primary_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self._interfaces.keys()))

        self.add_parameter('acquisition_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self._interfaces.keys()))
        self.add_parameter('instruments',
                           get_cmd=lambda: list(self._interfaces.keys()))

    def add_connection(self, output_arg, input_arg, **kwargs):
        connection = SingleConnection(output_arg, input_arg, **kwargs)
        self.connections += [connection]
        return connection

    def combine_connections(self, *connections, **kwargs):
        connection = CombinedConnection(connections=connections, **kwargs)
        self.connections += [connection]
        return connection

    def get_connections(self, output_interface=None, output_instrument=None,
                        output_channel=None, input_interface=None,
                        input_instrument=None, input_channel=None,
                        trigger=None):
        """
        Returns all connections that satisfy given kwargs
        Args:
            output_interface: Connections must have output_interface
            output_instrument: Connections must have output_instrument name
            output_channel: Connections must have output_channel
            input_interface: Connections must have input_interface
            input_instrument: Connections must have input_instrument name
            input_channel: Connections must have input_channel
            trigger: Connection must be a triggering connection
        Returns:
            Connections that satisfy kwarg constraints
        """
        filtered_connections = self.connections
        if output_interface is not None:
            output_instrument = output_interface.instrument_name()
        if output_instrument is not None:
            filtered_connections = filter(
                lambda c: c.output['instrument'] == output_instrument,
                filtered_connections
            )

        if output_channel is not None:
            if isinstance(output_instrument, Channel):
                output_channel = output_channel.name
            filtered_connections = filter(
                lambda c: c.output['channel'] == output_channel,
                filtered_connections
            )

        if input_interface is not None:
            input_instrument = input_interface.instrument_name()
        if input_instrument is not None:
            filtered_connections = filter(
                lambda c: c.input['instrument'] == input_instrument,
                filtered_connections
            )
        if input_channel is not None:
            if isinstance(input_instrument, Channel):
                input_channel = input_channel.name
            filtered_connections = filter(
                lambda c: c.input['channel'] == input_channel,
                filtered_connections
            )
        if trigger is not None:
            filtered_connections = filter(
                lambda c: c.trigger == trigger,
                filtered_connections
            )

        return list(filtered_connections)

    def _get_pulse_interface(self, pulse):
        """
        Retrieves the instrument interface to output pulse
        Args:
            pulse: Pulse for which to find the default instrument interface

        Returns:
            Instrument interface for pulse
        """
        #
        print('Getting interface for pulse: {}'.format(pulse))

        # Only look at interfaces that are the output instrument for a
        # connection that satisfies pulse.connection_requirements
        connections = self.get_connections(**pulse.connection_requirements)
        print('All connections: {}'.format(self.connections))
        print('Available connections: {}'.format(connections))

        output_instruments = set([connection.output['instrument']
                                for connection in connections])
        print('Output instruments: {}'.format(output_instruments))

        interfaces = {instrument: self._interfaces[instrument]
                      for instrument in output_instruments}
        print('potential interfaces: {}'.format(interfaces))

        matched_interfaces = []
        for instrument_name, interface in interfaces.items():
            pulse_implementation = interface.get_pulse_implementation(pulse)

            # Skip to next interface if there is no pulse implementation
            if pulse_implementation is None:
                continue

            # Test if all pulse requirements of this pulse also have a matching
            # interface. Note that this is recursive.
            if not all([self._get_pulse_interface(pulse_requirement) is not None
                        for pulse_requirement
                        in pulse_implementation.pulse_requirements]):
                continue
            else:
                matched_interfaces.append(interface)

        if not matched_interfaces:
            raise Exception('No instruments have an implementation for pulses '
                            '{}'.format(pulse))
        elif len(matched_interfaces) > 1:
            raise Exception('More than one instrument have an implementation '
                            'for pulses {}. Functionality to choose instrument '
                            'not yet implemented'.format(pulse))
        else:
            return matched_interfaces[0]

    def get_pulse_instrument(self, pulse):
        """
        Retrieves the instrument name to output pulse
        Args:
            pulse: Pulse for which to find the default instrument name

        Returns:
            Instrument name for pulse
        """
        interface = self._get_pulse_interface(pulse)
        return interface.instrument_name()

    def get_pulse_connection(self, pulse, interface=None, instrument=None,
                             **kwargs):
        """
        Obtain default connection for a given pulse. If no instrument or
        instrument_name is given, the instrument is determined from
        self.get_pulse_instrument.
        Args:
            pulse: Pulse for which to find default connection
            interface (optional): Output instrument interface of pulse
            instrument (optional): Output instrument name of pulse
            **kwargs: Additional kwargs to specify connection

        Returns:
            Connection object for pulse
        """
        if interface is not None:
            connections = self.get_connections(
                output_interface=interface, **kwargs)
        elif instrument is not None:
            connections = self.get_connections(
                output_instrument=instrument, **kwargs)
        else:
            interface = self._get_pulse_interface(pulse)
            connections = self.get_connections(
                output_interface=interface, **kwargs)

        default_connections = [connection for connection in connections
                               if connection.default]
        if not default_connections:
            raise Exception('Instrument {} has connections {}, but none are '
                            'set as default'.format(interface, connections))
        elif len(default_connections) > 1:
            raise Exception('Instrument {} has connections {}, and more than'
                            'one are set as default'.format(interface,
                                                            connections))
        else:
            return default_connections[0]

    def _target_pulse(self, pulse):
        # Get default output instrument
        interface = self._get_pulse_interface(pulse)
        connection = self.get_pulse_connection(pulse, interface=interface)

        targeted_pulse = pulse.copy()
        targeted_pulse.connection = connection
        interface.pulse_sequence(('add', targeted_pulse))

        pulse_implementation = interface.get_pulse_implementation(targeted_pulse)

        # Also target any pulses that are in the pulse_requirements
        for pulse_requirement in pulse_implementation.pulse_requirements:
            self._target_pulse(pulse_requirement)

    def target_pulse_sequence(self, pulse_sequence):
        # Clear pulses sequences of all instruments
        for interface in self._interfaces.values():
            interface.pulse_sequence('clear')

        # Add pulses in pulse_sequence to pulse_sequences of instruments
        for pulse in pulse_sequence:
            self._target_pulse(pulse)

        # Setup each of the instruments using its pulse_sequence
        for interface in self._interfaces.values():
            interface.pulse_sequence(('duration', pulse_sequence.duration))
            interface.setup()

        # TODO setup acquisition instrument


class Connection:
    def __init__(self, default=False):
        self.input = {}
        self.output = {}

        # TODO make default dependent on implementation
        self.default = default


class SingleConnection(Connection):
    def __init__(self, output_arg, input_arg,
                 trigger=False, acquire=False, **kwargs):
        """
        Class representing a connection between instrument channels.

        Args:
            output_arg: Specification of output channel.
                Can be:
                    str "{instrument_name}.{output_channel_name}"
                    tuple ({instrument_name}, {output_channel_name})
            input_channel:
            trigger (bool): Sets the output channel to trigger the input
                instrument
            acquire (bool): Sets if this connection is used for acquisition
            default (bool): Sets if this connection is the default for pulses
        """
        # TODO add optionality of having multiple channels connected.
        # TODO Add mirroring of other channel.
        super().__init__(**kwargs)

        if type(output_arg) is str:
            output_instrument, output_channel = output_arg.split('.')
        elif type(output_arg) is tuple:
            output_instrument, output_channel = output_arg
        else:
            raise TypeError('Connection output must be a string or tuple')
        self.output['instrument'] = output_instrument
        self.output['channel'] = output_channel

        if type(input_arg) is str:
            input_instrument, input_channel = input_arg.split('.')
        elif type(input_arg) is tuple:
            input_instrument, input_channel = input_arg
        else:
            raise TypeError('Connection input must be a string or tuple')
        self.input['instrument'] = input_instrument
        self.input['channel'] = input_channel

        self.trigger = trigger
        # TODO add this connection to input_instrument.trigger

        self.acquire = acquire

    def __repr__(self):
        output_str = "Connection{{{}.{}->{}.{}}}(".format(
            self.output['instrument'], self.output['channel'],
            self.input['instrument'], self.input['channel'])
        if self.trigger:
            output_str += ', trigger'
        if self.default:
            output_str += ', default'
        if self.acquire:
            output_str += ', acquire'
        output_str += ')'
        return output_str


class CombinedConnection(Connection):
    def __init__(self, connections, scaling_factors=None, **kwargs):
        super().__init__(**kwargs)
        self.connections = connections
        self.output['instruments'] = list(set([connection.output['instrument']
                                          for connection in connections]))
        if len(self.output['instruments']) == 1:
            self.output['instrument'] = self.output['instruments'][0]
        else:
            raise Exception('Connections with multiple output instruments not'
                            'yet supported')
        self.output['channels'] = list(set([connection.output['channel']
                                       for connection in connections]))
        self.input['instruments'] = list(set([connection.input['instrument']
                                         for connection in connections]))
        self.input['channels'] = list(set([connection.input['channel']
                                      for connection in connections]))

        if scaling_factors is None:
            scaling_factors = {connection.input['channel']: 1
                               for connection in connections}
        elif type(scaling_factors) is list:
            # Convert scaling factors to dictionary with channel keys
            scaling_factors = {connection.input['channel']: scaling_factor
                               for (connection, scaling_factor)
                               in zip(connections, scaling_factors)}
        self.scaling_factors = scaling_factors
