from functools import partial
from silq.pulses import TriggerPulse
from silq.instrument_interfaces import InstrumentInterface, Channel

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class Layout(Instrument):
    shared_kwargs=['instruments']
    # TODO Should make into an instrument
    def __init__(self, name, instruments, **kwargs):
        super().__init__(name, **kwargs)

        # Add interfaces for each instrument to self.instruments
        self._instruments = {}
        for instrument in instruments:
            self._add_instrument(instrument)

        self.connections = []

        self.add_parameter('trigger_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self._instruments.keys()))

        self.add_parameter('acquisition_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self._instruments.keys()))
        self.add_parameter('instruments',
                           get_cmd=lambda: list(self._instruments.keys()))

    def add_connection(self, output, input, **kwargs):
        connection = SingleConnection(output, input, **kwargs)
        self.connections += [connection]
        return connection

    def combine_connections(self, *connections, **kwargs):
        connection = CombinedConnection(connections=connections, **kwargs)
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
            if isinstance(output_instrument, InstrumentInterface):
                output_instrument = output_instrument.name
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

        if input_instrument is not None:
            if isinstance(input_instrument, InstrumentInterface):
                input_instrument = input_instrument.name
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
        return list(filtered_connections)

    def _add_instrument(self, instrument):
        from silq.instrument_interfaces import \
            get_instrument_interface
        instrument_interface = get_instrument_interface(instrument)
        self._instruments[instrument.name] = instrument_interface

    def _get_pulse_instrument(self, pulse):
        """
        Retrieves the instrument interface to output pulse
        Args:
            pulse: Pulse for which to find the default instrument interface

        Returns:
            Instrument interface for pulse
        """
        instruments = [instrument for instrument in
                       self._instruments.values() if
                       instrument.get_pulse_implementation(pulse)]
        if not instruments:
            raise Exception('No instruments have an implementation for pulses '
                            '{}'.format(pulse))
        elif len(instruments) > 1:
            raise Exception('More than one instrument have an implementation '
                            'for pulses {}. Functionality to choose instrument '
                            'not yet implemented'.format(pulse))
        else:
            return instruments[0]

    def get_pulse_instrument_name(self, pulse):
        instrument = self._get_pulse_instrument(pulse)
        return instrument.name

    def get_pulse_connection(self, pulse, instrument=None, instrument_name=None,
                             **kwargs):
        """
        Obtain default connection for a given pulse. If no instrument or
        instrument_name is given, the instrument is determined from
        self.get_pulse_instrument.
        Args:
            pulse: Pulse for which to find default connection
            instrument (optional): Output instrument of pulse
            instrument_name (optional): Output instrument name of pulse
            **kwargs: Additional kwargs to specify connection

        Returns:
            Connection object for pulse
        """
        if instrument is not None:
            connections = self.get_connections(
                output_instrument=instrument, **kwargs)
        elif instrument_name is not None:
            connections = self.get_connections(
                output_instrument_name=instrument_name, **kwargs)
        else:
            instrument = self._get_pulse_instrument(pulse)
            connections = self.get_connections(
                output_instrument=instrument, **kwargs)


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
        for instrument in self._instruments.values():
            instrument.pulse_sequence.clear()

        # Add pulses in pulse_sequence to pulse_sequences of instruments
        for pulse in pulse_sequence:
            # Get default output instrument
            instrument = self._get_pulse_instrument(pulse)
            connection = self.get_pulse_connection(pulse, instrument=instrument)

            targeted_pulse = pulse.copy()
            targeted_pulse.connection = connection
            instrument.pulse_sequence.add(targeted_pulse)

            # If instrument is not the main triggering instrument, add triggers
            # to each of the triggering instruments until you reach the main
            # triggering instrument.
            # TODO this should only be done in some cases, for instance if an
            # Arbstudio is the input instrument and is in stepped mode
            while instrument.name != self.trigger_instrument():
                connection = instrument.trigger
                # Replace instrument by its triggering instrument
                instrument = self.instruments[
                    connection.output['instrument']]
                instrument.pulse_sequence.add(
                    TriggerPulse(t_start=pulse.t_start, connection=connection))

        # Setup each of the instruments using its pulse_sequence
        for instrument in self._instruments.values():
            instrument.setup()

        # TODO setup acquisition instrument


class Connection:
    def __init__(self, default=False):
        self.input = {}
        self.output = {}

        # TODO make default dependent on implementation
        self.default = default

class SingleConnection(Connection):
    def __init__(self, output, input,
                 trigger=False, acquire=False, **kwargs):
        """
        Class representing a connection between instrument channels.

        Args:
            output: Specification of output channel.
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

        if type(output) is str:
            output_instrument, output_channel = output.split('.')
        elif type(output) is tuple:
            output_instrument, output_channel = output
        self.output['instrument'] = output_instrument
        self.output['channel'] = output_channel

        if type(input) is str:
            input_instrument, input_channel = input.split('.')
        elif type(input) is tuple:
            input_instrument, input_channel = input
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
