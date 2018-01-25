import os
from collections import OrderedDict as od, Iterable
import logging
from copy import copy
import pickle
from time import sleep
from typing import Union, List

import silq
from silq import config
from silq.instrument_interfaces import Channel, InstrumentInterface
from silq.pulses.pulse_modules import PulseSequence

from qcodes import Instrument, FormatLocation
from qcodes.utils import validators as vals
from qcodes.data.io import DiskIO

logger = logging.getLogger(__name__)

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
connection_conditions = ['input_arg', 'input_instrument', 'input_channel',
                         'input_interface', 'output_arg', 'output_instrument',
                         'output_channel', 'output_interface', 'trigger']


class Layout(Instrument):
    shared_kwargs = ['instrument_interfaces']

    def __init__(self, name: str = 'layout',
                 instrument_interfaces: List[InstrumentInterface] = [],
                 store_pulse_sequences_folder: Union[bool, None] = None,
                 **kwargs):
        """ Acquisition controller of instruments via their interfaces.

        The Layout is provided with the instruments and their associated
        interfaces, as well as connectivity between instrument channels.


        Args:
            name: Name of Layout instrument.
            instrument_interfaces: List of all instrument interfaces
            store_pulse_sequences_folder: Folder in which to store a copy of
                any pulse sequence that is targeted. Pulse sequences are
                stored as pickles, and can be used to trace back measurement
                parameters.
            **kwargs: Additional kwargs passed to Instrument
        """
        super().__init__(name, **kwargs)

        # Add interfaces for each instrument to self.instruments
        self._interfaces = {interface.instrument_name(): interface
                            for interface in instrument_interfaces}

        # TODO remove this and methods once parameters are improved
        self._primary_instrument = None

        self.connections = []

        self.add_parameter('instruments',
                           get_cmd=lambda: list(self._interfaces.keys()))
        self.add_parameter('primary_instrument',
                           get_cmd=lambda: self._primary_instrument,
                           set_cmd=self._set_primary_instrument,
                           vals=vals.Enum(*self._interfaces.keys()))

        self.add_parameter('acquisition_instrument',
                           set_cmd=None,
                           initial_value=None,
                           vals=vals.Enum(*self._interfaces.keys()))
        self.add_parameter('acquisition_outputs',
                           set_cmd=None,
                           initial_value=([('chip.output', 'output')]
                                          if 'chip' in self._interfaces.keys()
                                          else []),
                           vals=vals.Anything())

        self.add_parameter(name='samples',
                           set_cmd=None,
                           initial_value=1)

        self.add_parameter(name='active',
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())

        # Untargeted pulse_sequence, can be set via layout.pulse_sequence
        self._pulse_sequence = None

        # Targeted pulse sequence, which is generated after targeting
        # layout.pulse_sequence
        self.targeted_pulse_sequence = None

        # Handle saving of pulse sequence
        if store_pulse_sequences_folder is not None:
            self.store_pulse_sequences_folder = store_pulse_sequences_folder
        elif config.properties.get('store_pulse_sequences_folder') is not None:
            self.store_pulse_sequences_folder = \
                config.properties.store_pulse_sequences_folder
        else:
            self.store_pulse_sequences_folder = None
        self._pulse_sequences_folder_io = DiskIO(store_pulse_sequences_folder)

        self.acquisition_shapes = {}

    @property
    def pulse_sequence(self):
        return self._pulse_sequence

    @pulse_sequence.setter
    def pulse_sequence(self, pulse_sequence):
        """ Set target pulse sequence to layout, which distributes pulses to
        interfaces

        Args:
            pulse_sequence: Pulse sequence to be targeted
        """
        assert isinstance(pulse_sequence, PulseSequence), \
            "Can only set layout.pulse_sequence to a PulseSequence"

        # Target pulse_sequence, distributing pulses to interfaces
        self._target_pulse_sequence(pulse_sequence)


    @property
    def acquisition_interface(self):
        """ Obtain interface for acquisition system

        Returns:
            Interface instrument
        """
        if self.acquisition_instrument() is not None:
            return self._interfaces[self.acquisition_instrument()]
        else:
            return None

    @property
    def acquisition_channels(self):
        """ Returns a dictionary acquisition_label: acquisition_channel_name pairs.
        The acquisition_label is the label associated with a certain
        acquisition channel. This is settable via layout.acquisition_outputs
        The acquisition_channel_name is the actual channel name of the
        acquisition controller.
        """

        acquisition_channels = od()
        for output_arg, output_label in self.acquisition_outputs():
            # Use try/except in case not all connections exist
            try:
                connection = self.get_connection(
                    output_arg=output_arg,
                    input_instrument=self.acquisition_instrument())
                acquisition_channels[output_label] = \
                    connection.input['channel'].name
            except:
                return None
        return acquisition_channels

    @property
    def sample_rate(self):
        if self.acquisition_interface is not None:
            return self.acquisition_interface.sample_rate()
        else:
            return None

    def add_connection(self, output_arg, input_arg, **kwargs):
        """
        Creates a SingleConnection between two instrument interface channels.
        Note that both the output and input instruments must already have a
        corresponding interface and should have been passed when creating the
        Layout.

        Args:
            output_arg: "{instrument}_{channel}" string for the connection
                output
            input_arg: "{instrument}_{channel}" string for the connection input
            **kwargs: Additional options for the SingleConnection

        Returns:
            SingleConnection object
        """
        output_instrument, output_channel_name = output_arg.split('.')
        output_interface = self._interfaces[output_instrument]
        output_channel = output_interface.get_channel(output_channel_name)

        input_instrument, input_channel_name = input_arg.split('.')
        input_interface = self._interfaces[input_instrument]
        input_channel = input_interface.get_channel(input_channel_name)

        connection = SingleConnection(output_instrument=output_instrument,
                                      output_channel=output_channel,
                                      input_instrument=input_instrument,
                                      input_channel=input_channel,
                                      **kwargs)
        self.connections += [connection]
        return connection

    def combine_connections(self, *connections, **kwargs):
        """
        Combines multiple SingleConnections into a CombinedConnection.
        This is useful for cases such as when pulses are by default sent
        through multiple connections simultaneously.
        Args:
            *connections: list of SingleConnections
            **kwargs: Additional kwargs for CombinedConnection

        Returns:
            CombinedConnection object
        """
        connection = CombinedConnection(connections=connections, **kwargs)
        self.connections += [connection]
        return connection

    def load_connections(self, filepath=None):
        """
        Load connections from qcodes.config.user.connections
        Returns:
            None
        """
        self.connections.clear()
        if filepath is not None:
            import os, json
            if not os.path.isabs(filepath):
                # relative path is wrt silq base directory
                filepath = os.path.join(silq.get_SilQ_folder(), filepath)
            with open(filepath, "r") as fp:
                connections = json.load(fp)
        else:
            from silq import config
            connections = config.get('connections', None)
            if connections is None:
                raise RuntimeError('No connections found in config.user')

        for connection in connections:
            # Create a copy of connection, else it changes the config
            connection = connection.copy()
            if 'combine' in connection:
                # Create CombinedConnection. connection['combine'] consists of
                # output args of the SingleConnections
                output_args = connection.pop('combine')
                # Must add actual Connection objects, so retrieving them from
                #  Layout
                nested_connections = [self.get_connection(output_arg=output_arg)
                                      for output_arg in output_args]
                # Remaining properties in connection dict are kwargs
                self.combine_connections(*nested_connections, **connection)
            else:
                # Properties in connection dict are kwargs
                self.add_connection(**connection)

    def get_connections(self, connection=None, **conditions):
        """ Returns all connections that satisfy given conditions

        Args:
            connection: Specific connection to be checked. If the connection
                is in layout.connections, it returns a list with the connection.
                Can be useful when pulse.connection_requirements needs a
                specific connection
            connection: Specific connection to be checked. If the connection
                is in layout.connections, it returns a list with the connection.
                Can be useful when pulse.connection_requirements needs a
                specific connection
            output_arg: string representation of output.

                * SingleConnection has form '{instrument}.{channel}'
                * CombinedConnection is list of SingleConnection output_args,
                  which must have equal number of elements as underlying
                  connections. Can be combined with input_arg, in which
                  case the first element of output_arg and input_arg are
                  assumed to belong to the same underlying connection.

            output_interface: Connections must have output_interface object
            output_instrument: Connections must have output_instrument name
            output_channel: Connections must have output_channel
                (either Channel object, or channel name)
            input_arg: string representation of input.

                * SingleConnection has form '{instrument}.{channel}'
                * CombinedConnection is list of SingleConnection output_args,
                  which must have equal number of elements as underlying
                  connections. Can be combined with output_arg, in which
                  case the first element of output_arg and input_arg are
                  assumed to belong to the same underlying connection.

            input_interface: Connections must have input_interface object
            input_instrument: Connections must have input_instrument name
            input_channel: Connections must have input_channel
                (either Channel object, or channel name)
            trigger: Connection is used for triggering
            acquire: Connection is used for acquisition
            software: Connection is not an actual hardware connection. Used
                when a software trigger needs to be sent

        Returns:
            Connections that satisfy kwarg constraints
        """
        if connection is not None:
            # Check if connection is in connections.
            if connection in self.connections:
                return [connection]
            else:
                # raise RuntimeError(f"{connection} not found in connections")
                raise RuntimeError("{} not found in connections".format(
                    connection))
        else:
            return [connection for connection in self.connections
                    if connection.satisfies_conditions(**conditions)]

    def get_connection(self, environment=None,
                       connection_label=None, **conditions):
        """
        Returns connection that satisfy given kwargs.
        If not exactly one was found, it raises an error.
        Args:
            output_interface: Connections must have output_interface
            output_instrument: Connections must have output_instrument name
            output_channel: Connections must have output_channel
            input_interface: Connections must have input_interface
            input_instrument: Connections must have input_instrument name
            input_channel: Connections must have input_channel
            trigger: Connection must be a triggering connection
        Returns:
            Connection that satisfies kwarg constraints
        """
        if connection_label is not None:
            if environment is None:
                # Determine environment, either from connection label or
                # default_environment
                if '.' in connection_label:
                    # connection label has form {environment}.{connection_label}
                    environment, connection_label = connection_label.split('.')
                else:
                    # Use default environment defined in config
                    assert 'default_environment' in config.properties, \
                        "No environment nor default environment provided"
                    environment = config.properties.default_environment

            # Obtain list of connections from environment
            environment_connections = config[environment].connections

            # Find connection from environment connections
            assert connection_label in environment_connections, \
                f"Could not find connection {connection_label} in " \
                f"environment {environment}"
            connection_attrs = environment_connections[connection_label]

            connection_output_str, connection_input_str = connection_attrs

            connection = self.get_connection(output_arg=connection_output_str,
                                             input_arg=connection_input_str)
            return connection
        else:
            # Extract from conditions other than environment and
            # connection_label
            connections = self.get_connections(**conditions)
            assert len(connections) == 1, \
                f"Found {len(connections)} connections " \
                f"instead of one satisfying {conditions}"
            return connections[0]

    def _set_primary_instrument(self, primary_instrument):
        # TODO remove once parameters are improved
        for instrument_name, interface in self._interfaces.items():
            interface.is_primary(instrument_name == primary_instrument)

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

        # Set acquisition interface as first interface
        if not sorted_interfaces:
            sorted_interfaces = [self.acquisition_interface]

        # Find all interfaces that have not been sorted yet
        remaining_interfaces = {
            instrument: interface
            for instrument, interface in self._interfaces.items()
            if interface not in sorted_interfaces}

        # All interfaces are in sorted_interfaces. Finishing recursion
        if not remaining_interfaces:
            return sorted_interfaces

        for instrument, interface in remaining_interfaces.items():
            output_connections = self.get_connections(
                output_interface=interface)

            # Find instruments that have this instrument as an input
            input_instruments = {connection.input['instrument']
                                 for connection in output_connections
                                 if 'instrument' in connection.input}

            # Add interface to sorted interface if it does not trigger any of
            # the remaining interfaces
            if all(instrument not in remaining_interfaces
                   for instrument in input_instruments):
                sorted_interfaces.append(interface)

        # Ensure that we are not in an infinite loop
        if not any(interface in sorted_interfaces
                   for interface in remaining_interfaces.values()):
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
            pulse_implementation = interface.has_pulse_implementation(pulse)

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
        Retrieves the name of the instrument that can output the given pulse

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

        if pulse.connection_label is not None:
            connection = self.get_connection(
                environment=pulse.environment,
                connection_label=pulse.connection_label,
                **connection_requirements)
        else:
            connection = self.get_connection(**connection_requirements)
        return connection

    def _target_pulse(self, pulse, **kwargs):
        """
        Add pulse to default instrument that can output the pulse.
        Often the instrument requires additional pulses, such as a triggering
        pulse. These pulses are also targeted by recursively calling this
        function.
        During targeting, all the properties of pulses are fixed, meaning
        that they do not depend on other pulses anymore. By contrast,
        untargeted pulses can be dependent on other pulses, such as starting
        at the end of the previous pulse, even if the previous pulse is
        modified.
        Args:
            pulse: pulse to be targeted
            **kwargs: No function yet  (TODO remove)

        Returns:
            None
        """
        # Get default output instrument
        connection = self.get_pulse_connection(pulse)
        interface = self._interfaces[connection.output['instrument']]

        # Add pulse to acquisition instrument if it must be acquired
        if pulse.acquire:
            self.acquisition_interface.pulse_sequence.add(pulse)

        pulses = connection.target_pulse(pulse)
        if not isinstance(pulses, list):
            pulses = [pulses]

        for pulse in pulses:
            targeted_pulse = interface.get_pulse_implementation(
                pulse, connections=self.connections)

            self.targeted_pulse_sequence.add(targeted_pulse)

            interface.pulse_sequence.add(targeted_pulse)

            # Also add pulse to input interface pulse sequence
            input_interface = self._interfaces[
                pulse.connection.input['instrument']]
            input_interface.input_pulse_sequence.add(targeted_pulse)

    def _target_pulse_sequence(self, pulse_sequence):
        """
        Targets a pulse sequence.
        For each of the pulses, it finds the instrument that can output it,
        and adds the pulse to its respective interface. It also takes care of
        any additional necessities, such as additional triggering pulses.

        Args:
            pulse_sequence: (Untargeted) pulse sequence that is to be targeted.

        Returns:
            None
        """
        logger.info(f'Targeting pulse sequence {pulse_sequence}')

        if self.active():
            self.stop()

        # Copy untargeted pulse sequence so none of its attributes are modified
        self.targeted_pulse_sequence = PulseSequence()
        self.targeted_pulse_sequence.duration = pulse_sequence.duration

        # Clear pulses sequences of all instruments
        for interface in self._interfaces.values():
            logger.debug(f'Initializing interface {interface.name}')
            interface.initialize()
            interface.pulse_sequence.duration = pulse_sequence.duration
            interface.input_pulse_sequence.duration = pulse_sequence.duration

        # Add pulses in pulse_sequence to pulse_sequences of instruments
        for pulse in pulse_sequence:
            self._target_pulse(pulse)

        # Setup each of the instruments hierarchically using its pulse_sequence
        # The ordering is because instruments might need final pulses from
        # triggering instruments (e.g. triggering pulses that can only be
        # defined once all other pulses have been given)
        for interface in self._get_interfaces_hierarchical():
            additional_pulses = interface.get_additional_pulses(interface=self)
            for pulse in additional_pulses:
                self._target_pulse(pulse)

        # Update pulse sequence
        self._pulse_sequence = copy(pulse_sequence)

        # Store pulse sequence
        if self.store_pulse_sequences_folder:
            logger.debug('Storing pulse sequence to')
            try:
                self._pulse_sequences_folder_io.base_location = \
                    self.store_pulse_sequences_folder
                location = FormatLocation()(self._pulse_sequences_folder_io,
                                            {"name": 'pulse_sequence'})
                location += '.pickle'

                filepath = self._pulse_sequences_folder_io.to_path(location)

                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                logger.debug(f'Storing pulse sequence to {filepath}')
                with open(filepath, 'wb') as f:
                    pickle.dump(self._pulse_sequence, f)
            except:
                logger.exception('Could not save pulse sequence')

    def update_flags(self, new_flags):
        """
        Updates existing flags with new flags. Flags are instructions sent
        to interfaces, usually from other interfaces to modify the usual
        operations. Examples are skip_start, setup kwargs, etc.
        Args:
            new_flags: {instrument: {flag: val}} dict

        Returns:
            None
        """
        if 'skip_start' in new_flags:
            self.flags['skip_start'].add(new_flags['skip_start'])

        if 'post_start_actions' in new_flags:
            self.flags['post_start_actions'] += new_flags['post_start_actions']

        if 'start_last' in new_flags:
            self.flags['start_last'].add(new_flags['start_last'])

        if 'setup' in new_flags:
            for instrument_name, new_instrument_flags in new_flags['setup'].items():
                instrument_flags = self.flags['setup'].get(instrument_name, {})
                for flag, val in new_instrument_flags.items():
                    if flag not in instrument_flags:
                        # New instrument flag is not yet in existing flags,
                        # add it to the existing flags
                        instrument_flags[flag] = val
                    elif type(val) is dict:
                        # New instrument flag is already in existing flags,
                        # but the value is a dict. Update existing flag dict with
                        #  new dict
                        instrument_flags[flag].update(val)
                    elif not instrument_flags[flag] == val:
                        raise RuntimeError(
                            f"Instrument {instrument_name} flag {flag} already "
                            f"exists, but val {val} does not match existing "
                            f"val {instrument_flags[flag]}")
                    else:
                        # Instrument Flag exists, and values match
                        pass
                # Set dict since it may not have existed previously
                self.flags['setup'][instrument_name] = instrument_flags

    def setup(self, samples=None, repeat=True, ignore=[], **kwargs):
        """
        Sets up all the instruments after having targeted a pulse sequence.
        Instruments are setup through their respective interfaces, and only
        the instruments that have a pulse sequence are setup.
        The interface setup order is by hierarchy, i.e. instruments that trigger
        other instruments are never setup before the triggered instruments.
        Any flags, such as to skip starting an instrument, are also collected
        and applied at this stage.

        Interfaces can return a dict of flags. The following flags are accepted:
        - setup (dict): key (instrument_name) value setup_flags
            When the interface with instrument_name (key) is setup, the
            setup flags (value) will be passed along
        - skip_start (str): instrument name that should be skipped when
            calling layout.start()
        - post_start_actions (list(callable)): callables to perform after all
            interfaces are started
        - start_last (interface) interface that should be started after others


        Args:
            samples: Number of samples (by default uses previous value)
            repeat: Whether to repeat pulse sequence indefinitely or stop at end
            **kwargs: additional kwargs sent to all interfaces being setup.

        Returns:
            None
        """

        logger.info(f'Layout setup with {samples} samples and kwargs: {kwargs}')

        if not self.pulse_sequence:
            raise RuntimeError("Cannot setup with an empty PulseSequence.")

        if self.active():
            self.stop()

        # Initialize with empty flags, used for instructions between interfaces
        self.flags = {'setup': {},
                      'skip_start': set(),
                      'post_start_actions': [],
                      'start_last': set(),
                      'auto_stop': None}

        if samples is not None:
            self.samples(samples)

        if self.acquisition_interface is not None:
            self.acquisition_interface.acquisition_channels(
                [ch_name for _, ch_name in self.acquisition_channels.items()])

        for interface in self._get_interfaces_hierarchical():
            if interface.pulse_sequence and interface.instrument_name() not in ignore:
                # Get existing setup flags (if any)
                setup_flags = self.flags['setup'].get(interface.instrument_name(), {})
                if setup_flags:
                    logger.debug(f'{interface.name} setup flags: {setup_flags}')

                is_primary = self.primary_instrument() == interface.instrument.name
                output_connections = self.get_connections(
                    output_interface=interface)

                flags = interface.setup(samples=self.samples(),
                                        is_primary=is_primary,
                                        output_connections=output_connections,
                                        repeat=repeat,
                                        **setup_flags, **kwargs)
                if flags:
                    logger.debug(f'Received flags {flags} from interface {interface}')
                    self.update_flags(flags)

        if self.flags['auto_stop'] is None:
            # auto_stop flag hasn't been specified.
            # Set to True if the pulse sequence doesn't repeat, False otherwise
            self.flags['auto_stop'] = not repeat

        # Create acquisition shapes
        trace_shapes = self.pulse_sequence.get_trace_shapes(
            sample_rate=self.sample_rate, samples=self.samples())
        self.acquisition_shapes = {}
        output_labels = [output[1] for output in self.acquisition_outputs()]
        for pulse_name, shape in trace_shapes.items():
            self.acquisition_shapes[pulse_name] = {
                label: shape for label in output_labels}

    def start(self, auto_stop=None):
        """
        Starts all the instruments except the acquisition instrument
        The interface start order is by hierarchy, i.e. instruments that trigger
        other instruments are never setup before the triggered instruments.
        Does not start instruments that have the flag skip_start
        Returns:

        """
        self.active(True)
        for interface in self._get_interfaces_hierarchical():
            if interface == self.acquisition_interface:
                continue
            elif interface.instrument_name() in self.flags['skip_start']:
                logger.info('Skipping starting {interface.name} (flag skip_start)')
                continue
            elif interface in self.flags['start_last']:
                logger.info('Delaying starting {interface.name} (flag start_last)')
                continue
            elif interface.pulse_sequence:
                interface.start()
                logger.debug(f'{interface} started')
            else:
                logger.debug(f'Skipping starting {interface} (no pulse sequence)')

        for interface in self.flags['start_last']:
            interface.start()
            logger.debug(f'Started {interface} after others (flag start_last)')

        for action in self.flags['post_start_actions']:
            action()
            logger.debug(f'Called post_start_action {action}')

        logger.debug('Layout started')

        if auto_stop is None:
            auto_stop = self.flags['auto_stop']

        if auto_stop is not False:
            if auto_stop is True:
                # Wait for three times the duration of the pulse sequence, then stop instruments
                sleep(self.pulse_sequence.duration * 3 / 1000)
            elif isinstance(auto_stop, (int, float)):
                # Wait for duration specified in auto_stop
                sleep(auto_stop)
            else:
                raise SyntaxError('auto_stop must be either True or a number')
                self.stop()


    def stop(self):
        """
        Stops all instruments.
        Returns:
            None
        """
        for interface in self._get_interfaces_hierarchical():
            interface.stop()
            logger.debug(f'{interface} stopped')
        self.active(False)
        logger.debug('Layout stopped')

    def acquisition(self, stop=True):
        """
        Performs an acquisition.
        By default this includes starting and stopping of all the instruments.
        Args:
            stop (Bool): Whether to stop instruments after finishing
                measurements (True by default)

        Returns:
            data (Dict): Dictionary where every element is of the form
                acquisition_channel: acquisition_signal.
        """
        try:
            logger.info(f'Performing acquisition, stop when finished: {stop}')
            if not self.active():
                self.start()

            # Obtain traces from acquisition interface as dict
            pulse_traces = self.acquisition_interface.acquisition()

            if stop:
                self.stop()

            # current output is pulse_traces[pulse_name][acquisition_channel]
            # needs to be converted to data[pulse_name][output_label]
            # where output_label is taken from self.acquisition_channels()
            data = {}
            for pulse, channel_traces in pulse_traces.items():
                data[pulse] = {}
                for channel, trace in channel_traces.items():
                    # Find corresponding connection
                    connection = self.get_connection(
                        input_channel=channel,
                        input_instrument=self.acquisition_instrument())
                    # Get output arg (instrument.channel)
                    output_arg = connection.output['str']
                    # Find label corresponding to output arg
                    output_label = next(
                        item[1] for item in self.acquisition_outputs()
                        if item[0] == output_arg)
                    data[pulse][output_label] = trace
        except:
            # If any error occurs, stop all instruments
            self.stop()
            raise

        return data


class Connection:
    def __init__(self, default=False, scale=None):
        """
        Connection base class for connections between interfaces (instruments)

        Args:
            default (Bool): Whether this connection should be the default.
                This is used when multiple possible connection are found that
                can implement a pulse. In such a case, if any connection has
                default=True, it will be chosen over the others.
            scale: Whether there is a scaling factor between output and input.
                Scale 1/x means the signal at the input is x times lower than
                emitted from the output
        """
        self.input = {}
        self.output = {}
        self.scale = scale

        # TODO make default dependent on implementation
        self.default = default

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __hash__(self):
        # Define custom hash, used for creating a set of unique elements
        dict_items = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                # Dicts cannot be hashed, and must be converted
                v = tuple(sorted(v.items()))
            dict_items[k] = v
        return hash(tuple(sorted(dict_items)))

    def target_pulse(self, pulse):
        raise NotImplementedError();

    def satisfies_conditions(self, input_instrument=None,
                             input_channel=None, input_interface=None,
                             output_instrument=None, output_channel=None,
                             output_interface=None,
                             **kwargs):
        """
        Checks if this connection satisfies conditions. Note that all
        instrument/channel args can also be lists of elements. If so,
        condition is satisfied if connection property is in list
        Args:
            output_interface: Connection must have output_interface object
            output_instrument: Connection must have output_instrument name
            output_channel: Connection must have output_channel
                (either Channel object, or channel name)
            input_interface: Connection must have input_interface object
            input_instrument: Connection must have input_instrument name
            input_channel: Connection must have input_channel
                (either Channel object, or channel name)
        Returns:
            Bool depending on if the connection satisfies conditions
        """
        # Convert interfaces to their underlying instruments
        if output_interface is not None:
            output_instrument = output_interface.instrument_name()
        if input_interface is not None:
            input_instrument = input_interface.instrument_name()

        # Change instruments and channels into lists
        if not (output_instrument is None or type(output_instrument) is list):
            output_instrument = [output_instrument]
        if not (input_instrument is None or type(input_instrument) is list):
            input_instrument = [input_instrument]
        if not (output_channel is None or type(output_channel) is list):
            output_channel = [output_channel]
        if not (input_channel is None or type(input_channel) is list):
            input_channel = [input_channel]

        # If channel is an object, convert to its name
        if output_channel is not None:
            output_channel = [ch.name if isinstance(ch, Channel) else ch
                              for ch in output_channel]
        if input_channel is not None:
            input_channel = [ch.name if isinstance(ch, Channel) else ch
                             for ch in input_channel]

        # Test conditions
        if (output_instrument is not None) and \
                (self.output['instrument'] not in output_instrument):
            return False
        elif (output_channel is not None) and \
                (('channel' not in self.output) or
                     (self.output['channel'].name not in output_channel)):
            return False
        elif (input_instrument is not None) and \
                (('instrument' not in self.input) or
                     (self.input['instrument'] not in input_instrument)):
            return False
        elif (input_channel is not None) and \
                (('channel' not in self.input) or
                     (self.input['channel'].name not in input_channel)):
            return False
        else:
            return True


class SingleConnection(Connection):
    def __init__(self, output_instrument, output_channel,
                 input_instrument, input_channel,
                 trigger=False, acquire=False, software=False, **kwargs):
        """
        Class representing a single connection between instrument channels.

        Args:
            output_instrument (str): Name of output instrument
            output_channel (Channel): Output channel object
            input_instrument (str): Name of output instrument
            input_channel (Channel): Input channel object
            trigger (bool): Sets the output channel to trigger the input
                instrument
            acquire (bool): Sets if this connection is used for acquisition
            software (bool): Sets if this connection is a software connection.
                This is used for cases such as software triggering
            default (bool): Sets if this connection is the default for pulses
        """
        # TODO add optionality of having multiple channels connected.
        # TODO Add mirroring of other channel.
        super().__init__(**kwargs)

        self.output['instrument'] = output_instrument
        self.output['channel'] = output_channel
        self.output['str'] = '{}.{}'.format(output_instrument,
                                            output_channel.name)

        self.input['instrument'] = input_instrument
        self.input['channel'] = input_channel
        self.input['str'] = '{}.{}'.format(input_instrument,
                                           input_channel.name)

        self.trigger = trigger
        # TODO add this connection to input_instrument.trigger

        self.acquire = acquire
        self.software = software

    def __repr__(self):
        output_str = "Connection{{{}.{}->{}.{}}}(".format(
            self.output['instrument'], self.output['channel'].name,
            self.input['instrument'], self.input['channel'].name)
        if self.trigger:
            output_str += ', trigger'
        if self.default:
            output_str += ', default'
        if self.acquire:
            output_str += ', acquire'
        if self.software:
            output_str += ', software'
        output_str += ')'
        return output_str

    def target_pulse(self, pulse, apply_scale=True, copy_pulse=True):
        """
        Targets a pulse to this connection.
        This includes applying a possible scale to the pulse amplitude,
        and linking the property pulse.connection to this connection object.
        Args:
            pulse: Pulse to be targeted
            apply_scale (bool): Sets if the pulse amplitude should be divided by
                the scale
            copy_pulse (bool): Sets if the pulse should be copied.
                If set to True, a new object is created, and all properties
                are fixed (not dependent on other pulses).

        Returns:
            Targeted (copy of) pulse
        """
        if copy_pulse:
            targeted_pulse = copy(pulse)
        else:
            targeted_pulse = pulse
        targeted_pulse.connection = self
        if apply_scale and self.scale is not None:
            for attr in ['amplitude', 'amplitude_start', 'amplitude_stop']:
                if hasattr(pulse, attr):
                    val = getattr(pulse, attr)
                    setattr(targeted_pulse, attr, val / self.scale)
        return targeted_pulse

    def satisfies_conditions(self, output_arg=None, input_arg=None,
                             trigger=None, acquire=None, software=None,
                             **kwargs):
        """
        Checks if this connection satisfies conditions. Note that all
        instrument/channel args can also be lists of elements. If so,
        condition is satisfied if connection property is in list
        Args:
            output_arg: Connection must have output '{instrument}.{channel}'
            output_interface: Connection must have output_interface object
            output_instrument: Connection must have output_instrument name
            output_channel: Connection must have output_channel
                (either Channel object, or channel name)
            input_arg: Connection must have input '{instrument}.{channel}'
            input_interface: Connection must have input_interface object
            input_instrument: Connection must have input_instrument name
            input_channel: Connection must have input_channel
                (either Channel object, or channel name)
            trigger: Connection is used for triggering
            acquire: Connection is used for acquisition
            software: Connection is not an actual hardware connection. Used
                when a software trigger needs to be sent
        Returns:
            Bool depending on if the connection satisfies conditions
        """
        if output_arg is not None:
            if not isinstance(output_arg, str):
                # output_arg is of wrong type (probably CombinedConnection)
                return False
            if not self.output['str'] == output_arg:
                return False

        if input_arg is not None:
            if not isinstance(input_arg, str):
                # input_arg is of wrong type (probably CombinedConnection)
                return False
            if not self.input['str'] == input_arg:
                return False

        if not super().satisfies_conditions(**kwargs):
            return False
        elif trigger is not None and self.trigger != trigger:
            return False
        elif acquire is not None and self.acquire != acquire:
            return False
        elif software is not None and self.software != software:
            return False
        else:
            return True


class CombinedConnection(Connection):
    def __init__(self, connections, scale=None, **kwargs):
        """
        Class used to combine multiple SingleConnections.
        The CombinedConnection is most useful for when multiple connections
        are often used together.
        Args:
            connections: SingleConnections to combine
            scale (float list): List specifying the value by which the
                amplitude of a pulse should be scaled for each connection
                (by default no scaling).
            **kwargs: Additional Connection kwargs.
        """
        super().__init__(scale=scale, **kwargs)
        self.connections = connections

        self.output['str'] = [connection.output['str']
                              for connection in connections]
        self.output['instruments'] = list(set([connection.output['instrument']
                                               for connection in connections]))
        if len(self.output['instruments']) == 1:
            self.output['instrument'] = self.output['instruments'][0]
            self.output['channels'] = list(set([connection.output['channel']
                                                for connection in connections]))

        self.input['str'] = [connection.input['str']
                             for connection in connections]
        self.input['instruments'] = list(set([connection.input['instrument']
                                              for connection in connections]))
        if len(self.input['instruments']) == 1:
            self.input['instrument'] = self.input['instruments'][0]
            self.input['channels'] = list(set([connection.input['channel']
                                               for connection in connections]))

    def __repr__(self):
        output = 'CombinedConnection\n'
        for connection in self.connections:
            output += '\t' + repr(connection) + '\n'
        return output

    def target_pulse(self, pulse):
        """
        Targets a pulse to the combined connection.
        This creates a copy of the pulse for each connection, applies the
        respective scale, and further lets each of the connections target the
        respective pulse.
        Args:
            pulse: Pulse to be targeted

        Returns:
            pulses: List of pulses for each of the connections
        """
        pulses = []
        for k, connection in enumerate(self.connections):
            targeted_pulse = copy(pulse)
            for attr in ['amplitude', 'amplitude_start', 'amplitude_stop']:
                if hasattr(pulse, attr):
                    val = getattr(pulse, attr)
                    if isinstance(val, Iterable):
                        if k < len(getattr(pulse, attr)):
                            setattr(targeted_pulse, attr, val[k])
                        else:
                            setattr(targeted_pulse, attr, val[0])
                    elif self.scale is not None:
                        setattr(targeted_pulse, attr, val / self.scale[k])
            targeted_pulse = connection.target_pulse(targeted_pulse,
                                                     copy_pulse=False)
            pulses.append(targeted_pulse)
        return pulses

    def satisfies_conditions(self, output_arg=None, input_arg=None,
                             trigger=None, acquire=None, **kwargs):
        """ Checks if this connection satisfies conditions
        Args:
            output_arg: list of SingleConnection output_args, each of which
                has the form '{instrument}.{channel}. Must have equal number
                of elements as underlying connections. Can be combined with
                input_arg, in which case the first element of output_arg and
                input_arg are assumed to belong to the same underlying
                connection.
            output_interface: Connection must have output_interface
            output_instrument: Connection must have output_instrument name
            output_channel: Connection must have output_channel
                (either Channel object, or channel name)
            input_arg: list of SingleConnection input_args, each of which
                has the form '{instrument}.{channel}. Must have equal number
                of elements as underlying connections. Can be combined with
                output_arg, in which case the first element of output_arg and
                input_arg are assumed to belong to the same underlying
                connection.
            input_interface: Connection must have input_interface
            input_instrument: Connection must have input_instrument name
            input_channel: Connection must have input_channel.
                (either Channel object, or channel name)
            trigger: Connection is used for triggering
            acquire: Connection is used for acquisition
        Returns:
            Bool depending on if the connection satisfies conditions
        """

        if output_arg is not None:
            if not isinstance(output_arg, list):
                # output_arg is not a list (probably str for SingleConnection)
                return False
            if not len(output_arg) == len(self.connections):
                return False

        if input_arg is not None:
            if not isinstance(input_arg, list):
                # input_arg is not a list (probably str for SingleConnection)
                return False
            if not len(input_arg) == len(self.connections):
                return False

        if output_arg is not None and input_arg is not None:
            for connection in self.connections:
                # Check for each connection if there is an output/input
                # combination that satisfies conditions
                if not any(connection.satisfies_conditions(output_arg=output,
                                                           input_arg=input)
                           for output, input in zip(output_arg, input_arg)):
                    return False
        elif output_arg is not None:
            for connection in self.connections:
                # Check for each connection if there is an output that satisfies conditions
                if not any(connection.satisfies_conditions(output_arg=output)
                           for output in output_arg):
                    return False
        elif input_arg is not None:
            for connection in self.connections:
                # Check for each connection if there is an input that satisfies conditions
                if not any(connection.satisfies_conditions(input_arg=input)
                           for input in input_arg):
                    return False

        if not super().satisfies_conditions(**kwargs):
            return False
        elif trigger is not None:
            return False
        elif acquire is not None and self.acquire != acquire:
            return False
        else:
            return True