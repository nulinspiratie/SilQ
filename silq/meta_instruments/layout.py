import inspect

from silq.instrument_interfaces import Channel

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
connection_conditions = ['input_arg', 'input_instrument', 'input_channel',
                         'input_interface','output_arg', 'output_instrument',
                         'output_channel', 'output_interface','trigger']

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
        self.add_parameter('acquisition_outputs',
                           parameter_class=ManualParameter,
                           initial_value=(['chip.output'] if 'chip' in
                                          self._interfaces.keys() else []),
                           vals=vals.Anything())
        self.add_parameter('instruments',
                           get_cmd=lambda: list(self._interfaces.keys()))

    @property
    def acquisition_interface(self):
        if self.acquisition_instrument() is not None:
            return self._interfaces[self.acquisition_instrument()]
        else:
            return None

    def add_connection(self, output_arg, input_arg, **kwargs):
        connection = SingleConnection(output_arg, input_arg, **kwargs)
        self.connections += [connection]
        return connection

    def combine_connections(self, *connections, **kwargs):
        connection = CombinedConnection(connections=connections, **kwargs)
        self.connections += [connection]
        return connection

    def get_connections(self, **conditions):
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
        return [connection for connection in self.connections
                if connection.satisfies_conditions(**conditions)]

    def _get_interfaces_hierarchical(self, sorted_interfaces=[]):
        """
        Determines the hierarchy for instruments, ensuring that instruments
        in the list never trigger instruments later in the list.
        This function is recursive.
        Args:
            sorted_interfaces: should start empty. This gets filled recursively

        Returns:
            Hierarchically sorted list of interfaces
        """
        # Find all interfaces that have not been sorted yet
        remaining_interfaces = {
            instrument: interface
            for instrument, interface in self._interfaces.items()
            if interface not in sorted_interfaces}

        # All interfaces are in sorted_interfaces. Finishing recursion
        if not remaining_interfaces:
            return sorted_interfaces

        for instrument, interface in remaining_interfaces.items():
            trigger_connections = self.get_connections(
                 output_interface=interface, trigger=True)

            # Find instruments that are triggered by interface
            trigger_instruments = set(connection.input['instrument']
                                      for connection in trigger_connections)
            # Add interface to sorted interface if it does not trigger any of
            # the remaining interfaces
            if all(instrument not in remaining_interfaces
                   for instrument in trigger_instruments):
                sorted_interfaces.append(interface)

        # Ensure that we are not in an infinite loop
        if not any(interface in sorted_interfaces for interface in
                   remaining_interfaces.values()):
            raise RecursionError("Could not find hierarchy for instruments."
                                 " This likely means that instruments are "
                                 "triggering each other")

        # Go to next level in recursion
        return self._get_interfaces_hierarchical(sorted_interfaces)

    def _get_pulse_interface(self, pulse):
        """
        Retrieves the instrument interface to output pulse
        Args:
            pulse: Pulse for which to find the default instrument interface

        Returns:
            Instrument interface for pulse
        """
        #
        # print('Getting interface for pulse: {}'.format(pulse))

        # Only look at interfaces that are the output instrument for a
        # connection that satisfies pulse.connection_requirements
        connections = self.get_connections(**pulse.connection_requirements)
        # print('All connections: {}'.format(self.connections))
        # print('Available connections: {}'.format(connections))

        output_instruments = set([connection.output['instrument']
                                  for connection in connections])
        # print('Output instruments: {}'.format(output_instruments))

        interfaces = {instrument: self._interfaces[instrument]
                      for instrument in output_instruments}
        # print('potential interfaces: {}'.format(interfaces))

        matched_interfaces = []
        for instrument_name, interface in interfaces.items():
            is_primary = self.primary_instrument() == \
                         interface.instrument_name()
            pulse_implementation = interface.get_pulse_implementation(
                pulse, is_primary=is_primary)

            # Skip to next interface if there is no pulse implementation
            if pulse_implementation is None:
                continue

            # Test if all additional pulses of this pulse also have a matching
            # interface. Note that this is recursive.
            if not all([self._get_pulse_interface(additional_pulse) is not None
                        for additional_pulse
                        in pulse_implementation.additional_pulses]):
                continue
            else:
                matched_interfaces.append(interface)

        if not matched_interfaces:
            raise Exception('No instruments have an implementation for pulses '
                            '{}'.format(pulse))
        elif len(matched_interfaces) > 1:
            raise Exception('More than one interface has an implementation for '
                            'pulse. Functionality to choose instrument not yet '
                            'implemented.\nInterfaces: {}, pulse: {}'.format(
                matched_interfaces, pulse))
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
        connection_requirements = pulse.connection_requirements.copy()

        if interface is not None:
            connection_requirements['output_interface'] = interface
        elif instrument is not None:
            connection_requirements['output_instrument'] = instrument
        else:
            connection_requirements['output_interface'] = \
                self._get_pulse_interface(pulse)

        connections = self.get_connections(**connection_requirements, **kwargs)

        if len(connections) > 1:
            connections = [connection for connection in connections
                           if connection.default]

        if len(connections) != 1:
            raise Exception('Instrument {} did not have suitable connection out'
                            ' of connections {}'.format(interface, connections))
        else:
            return connections[0]

    def _target_pulse(self, pulse, **kwargs):
        # Get default output instrument
        interface = self._get_pulse_interface(pulse)
        connection = self.get_pulse_connection(pulse, interface=interface)

        is_primary = self.primary_instrument() == interface.instrument_name()
        targeted_pulse = interface.get_pulse_implementation(
            pulse, is_primary=is_primary)
        targeted_pulse.connection = connection
        interface.pulse_sequence(('add', targeted_pulse))

        # Add pulse to acquisition instrument if it must be acquired
        if pulse.acquire:
            self.acquisition_interface.pulse_sequence(('add', targeted_pulse))

        # Also target any pulses that are in additional_pulses, such as triggers
        for additional_pulse in targeted_pulse.additional_pulses:
            self._target_pulse(additional_pulse)

    def target_pulse_sequence(self, pulse_sequence):
        # Clear pulses sequences of all instruments
        for interface in self._interfaces.values():
            interface.pulse_sequence('clear')

        # Add pulses in pulse_sequence to pulse_sequences of instruments
        for pulse in pulse_sequence:
            self._target_pulse(pulse)

        # Setup each of the instruments hierarchically using its pulse_sequence
        # The ordering is because instruments might need final pulses from
        # triggering instruments (e.g. triggering pulses that can only be
        # defined once all other pulses have been given)
        for interface in self._get_interfaces_hierarchical():
            additional_pulses = interface.get_final_additional_pulses()
            for pulse in additional_pulses:
                self._target_pulse(pulse)

            interface.pulse_sequence(('duration', pulse_sequence.duration))
            interface.setup()


class Connection:
    def __init__(self, default=False):
        self.input = {}
        self.output = {}

        # TODO make default dependent on implementation
        self.default = default

    def satisfies_conditions(self, input_arg=None, input_instrument=None,
                             input_channel=None, input_interface=None,
                             output_arg=None, output_instrument=None,
                             output_channel=None, output_interface=None,
                             **kwargs):
        """
        Checks if this connection satisfies conditions
        Args:
            output_arg: Connection must have output 'instrument.channel'
            output_interface: Connection must have output_interface
            output_instrument: Connection must have output_instrument name
            output_channel: Connection must have output_channel
            input_arg: Connection must have input 'instrument.channel'
            input_interface: Connection must have input_interface
            input_instrument: Connection must have input_instrument name
            input_channel: Connection must have input_channel
        Returns:
            Bool depending on if the connection satisfies conditions
        """

        if output_interface is not None:
            output_instrument = output_interface.instrument_name()
        if input_interface is not None:
            input_instrument = input_interface.instrument_name()

        if isinstance(output_channel, Channel):
            output_channel = output_channel.name
        if isinstance(input_channel, Channel):
            input_channel = input_channel.name

        # Test conditions
        if output_instrument is not None and \
                    self.output.get('instrument', None) != output_instrument:
            return False
        elif output_channel is not None and \
                        self.output.get('channel', None) != output_channel:
            return False
        elif input_instrument is not None and \
                        self.input.get('instrument', None) != input_instrument:
            return False
        elif input_channel is not None and \
                        self.input.get('channel', None) != input_channel:
            return False
        else:
            return True


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
        self.output['str'] = '{}.{}'.format(output_instrument, output_channel)

        if type(input_arg) is str:
            input_instrument, input_channel = input_arg.split('.')
        elif type(input_arg) is tuple:
            input_instrument, input_channel = input_arg
        else:
            raise TypeError('Connection input must be a string or tuple')
        self.input['instrument'] = input_instrument
        self.input['channel'] = input_channel
        self.input['str'] = '{}.{}'.format(input_instrument, input_channel)

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

    def satisfies_conditions(self, trigger=None, **kwargs):
        """
        Checks if this connection satisfies conditions
        Args:
            output_interface: Connection must have output_interface
            output_instrument: Connection must have output_instrument name
            output_channel: Connection must have output_channel
            input_interface: Connection must have input_interface
            input_instrument: Connection must have input_instrument name
            input_channel: Connection must have input_channel
        Returns:
            Bool depending on if the connection satisfies conditions
        """
        if not super().satisfies_conditions(**kwargs):
            return False

        if trigger is not None and self.trigger != trigger:
            return False

        return True



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
