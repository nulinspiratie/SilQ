import os
import numpy as np
from collections import OrderedDict as od, Iterable
import logging
from copy import copy
import traceback
import pickle, dill
import time
from typing import Union, List, Sequence, Dict, Any
import h5py
from pathlib import Path

import silq
from silq.instrument_interfaces.interface import InstrumentInterface, Channel
from silq.pulses.pulse_modules import PulseSequence, find_matching_pulse_sequence
from silq.pulses.pulse_types import Pulse, MeasurementPulse

import qcodes as qc
from qcodes.instrument.parameter_node import parameter
from qcodes.instrument.parameter import Parameter
from qcodes import Instrument, FormatLocation, MatPlot
from qcodes.loops import ActiveLoop
from qcodes.utils import validators as vals
from qcodes.data.io import DiskIO
from qcodes.data.hdf5_format import HDF5Format
from qcodes.utils import PerformanceTimer

logger = logging.getLogger(__name__)

# Set of valid connection conditions for satisfies_conditions. These are
# useful when multiple objects have distinct satisfies_conditions kwargs
connection_conditions = ['input_arg', 'input_instrument', 'input_channel',
                         'input_interface', 'output_arg', 'output_instrument',
                         'output_channel', 'output_interface', 'trigger',
                         'trigger_start']


class Connection:
    """Connection base class for connections between interfaces (instruments).

    Args:
        scale: Whether there is a scaling factor between output and input.
            Scale 1/x means the signal at the input is x times lower than
            emitted from the output
    """
    def __init__(self, scale: float = None, label: str = None):

        self.input = {}
        self.output = {}
        self.scale = scale
        self.label = label

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        """Define a non-equality test"""
        return not self.__eq__(other)

    def __hash__(self):
        """Define custom hash, used for creating a set of unique elements."""
        dict_items = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                # Dicts cannot be hashed, and must be converted
                v = tuple(sorted(v.items()))
            dict_items[k] = v
        return hash(tuple(sorted(dict_items)))

    def target_pulse(self,
                     pulse: Pulse):
        """Target pulse to connection.

        Used to apply modifications such as an amplitude scale.

        Note:
            The pulse should be copied, and modifications applied to the copy.
        """
        raise NotImplementedError()

    def satisfies_conditions(self,
                             input_instrument: str = None,
                             input_channel: Union[Channel, str] = None,
                             input_interface: InstrumentInterface = None,
                             output_instrument: str = None,
                             output_channel: Union[Channel, str] = None,
                             output_interface: InstrumentInterface = None,
                             **kwargs) -> bool:
        """Checks if this connection satisfies conditions.

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
            True if the connection satisfies conditions

        Note:
            instrument/channel args can also be lists of elements. If so,
                condition is satisfied if connection property is in list.
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
                (('instrument' not in self.output) or
                (self.output['instrument'] not in output_instrument)):
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
    """Class representing a single connection between instrument channels.

    Args:
        output_instrument: Name of output instrument.
        output_channel: Output channel object.
        input_instrument: Name of output instrument.
        input_channel: Input channel object.
        trigger: Sets the output channel to trigger the input instrument.
        trigger_start: The output only triggers at the start of the
            `PulseSequence`. Ignored if trigger == False
        acquire: Sets if this connection is used for acquisition.
        software: Sets if this connection is a software connection.
            This is used for cases such as software triggering.
    """
    def __init__(self,
                 output_instrument: str,
                 output_channel: Channel,
                 input_instrument: str,
                 input_channel: Channel,
                 trigger: bool = False,
                 trigger_start: bool = False,
                 acquire: bool = False,
                 software: bool = False,
                 **kwargs):
        # TODO add optionality of having multiple channels connected.
        # TODO Add mirroring of other channel.
        super().__init__(**kwargs)

        self.output['instrument'] = output_instrument
        self.output['channel'] = output_channel
        self.output['str'] = f'{output_instrument}.{output_channel.name}'

        self.input['instrument'] = input_instrument
        self.input['channel'] = input_channel
        self.input['str'] = f'{input_instrument}.{input_channel.name}'

        self.trigger = trigger
        self.trigger_start = trigger_start
        # TODO add this connection to input_instrument.trigger

        self.acquire = acquire
        self.software = software

    def __repr__(self):
        output_str = "Connection("
        if self.label:
            output_str += f'{self.label}: '
        output_str += f"{{{self.output['str']}->{self.input['str']}}}"
        if self.trigger:
            output_str += ', trigger'
            if self.trigger_start:
                output_str += ' start'
        if self.acquire:
            output_str += ', acquire'
        if self.software:
            output_str += ', software'
        output_str += ')'
        return output_str

    def target_pulse(self,
                     pulse: Pulse,
                     apply_scale: bool = True,
                     copy_pulse: bool = True) -> Pulse:
        """Targets a pulse to this connection.

        During connection targeting, the pulse is copied and its properties
        modified for a specific connection. This includes applying a any scale
        to the pulse amplitude, and adding the connection to
        ``Pulse.connection``.

        Args:
            pulse: Pulse to be targeted
            apply_scale: Divide ``Pulse.amplitude`` by connection scale.
            copy_pulse: Copy pulse before targeting
                If set to True, a new object is created, and all properties
                are fixed (not dependent on other pulses).

        Returns:
            Targeted (copy of) pulse

        Note:
            If scale is applied, the attributes ``amplitude``,
            ``amplitude_start``, and ``amplitude_stop`` are scaled.
        """
        if copy_pulse:
            targeted_pulse = pulse.copy(connect_to_config=False)
        else:
            targeted_pulse = pulse
        targeted_pulse.connection = self
        if apply_scale and self.scale is not None:
            for attr in ['amplitude', 'amplitude_start', 'amplitude_stop']:
                if hasattr(pulse, attr):
                    val = getattr(pulse, attr)
                    setattr(targeted_pulse, attr, val / self.scale)
        return targeted_pulse

    def satisfies_conditions(self,
                             output_arg: str = None,
                             input_arg: str = None,
                             trigger: bool = None,
                             trigger_start: bool = None,
                             acquire: bool = None,
                             software: bool = None,
                             **kwargs) -> bool:
        """Checks if this connection satisfies conditions.

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
            True if connection satisfies conditions

        Note:
            All instrument/channel args can also be lists of elements. If so,
                condition is satisfied if connection property is in list

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
        elif trigger_start is not None and self.trigger_start != trigger_start:
            return False
        elif acquire is not None and self.acquire != acquire:
            return False
        elif software is not None and self.software != software:
            return False
        else:
            return True


class CombinedConnection(Connection):
    """Class used to combine multiple SingleConnections.

    The CombinedConnection is most useful for when multiple connections
    are often used together, such as when using compensated pulses.

    Args:
        connections: `SingleConnection` list to combine
        scale: List specifying the value by which the amplitude of a pulse
            should be scaled for each connection (by default no scaling).
        **kwargs: Additional Connection keyword arguments.
    """
    def __init__(self,
                 connections: List[SingleConnection],
                 label: str = None,
                 scale: List[float] = None,
                 **kwargs):
        super().__init__(scale=scale, label=label)
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

        self.trigger = False
        self.trigger_start = False

    def __repr__(self):
        output = 'CombinedConnection'
        if self.label:
            output += f' {self.label}: '
        for connection in self.connections:
            output += '\n\t' + repr(connection)
        return output

    def target_pulse(self,
                     pulse: Pulse) -> List[Pulse]:
        """Targets a pulse to the combined connection.

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
            targeted_pulse = pulse.copy(connect_to_config=False)
            for attr in ['amplitude', 'amplitude_start', 'amplitude_stop']:
                if hasattr(pulse, attr):
                    val = getattr(pulse, attr)
                    if isinstance(val, Iterable):
                        if k < len(getattr(pulse, attr)):
                            setattr(targeted_pulse, attr, val[k])
                        else:
                            setattr(targeted_pulse, attr, val[0])
                    elif self.scale is not None:
                        setattr(targeted_pulse, attr, val * self.scale[k])
            targeted_pulse = connection.target_pulse(targeted_pulse,
                                                     copy_pulse=False)
            pulses.append(targeted_pulse)
        return pulses

    def satisfies_conditions(self,
                             output_arg: str = None,
                             input_arg: str = None,
                             trigger: bool = None,
                             trigger_start: bool = None,
                             acquire: bool = None,
                             **kwargs):
        """Checks if this connection satisfies conditions

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
            trigger_start:
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
        elif trigger_start is not None:
            return False
        elif acquire is not None and self.acquire != acquire:
            return False
        else:
            return True


class Layout(Instrument):
    """Global pulse sequence controller of instruments via interfaces.

    The Layout contains a representation of the experimental layout, i.e. it
    has a list of the instruments in the experimental setup, and the physical
    connectivity between the instrument channels. Knowledge of the instruments
    and their connectivity allows the layout to target a generic
    setup-independent `PulseSequence` to one that is specific to the
    experimental layout. Each `Pulse` is passed along to the relevant
    `InstrumentInterface` of instruments, as well as additional pulses,
    such as triggering pulses.

    One the instrument interfaces has received the pulses it is supposed to
    apply, it can convert these into instrument commands during the setup phase.
    Once all instruments have been setup, the resulting output and acquisition
    of the instruments should match that of the pulse sequence.

    Args:
        name: Name of Layout instrument.
        instrument_interfaces: List of all instrument interfaces
        store_pulse_sequences_folder: Folder in which to store a copy of any
            pulse sequence that is targeted. Pulse sequences are stored as
            pickles, and can be used to trace back measurement parameters.
        **kwargs: Additional kwargs passed to Instrument
    """

    # Targeted pulse sequence whose duration exceeds this will raise an error
    maximum_pulse_sequence_duration = 25

    def __init__(
        self,
        name: str = "layout",
        instrument_interfaces: List[InstrumentInterface] = [],
        store_pulse_sequences_folder: Union[bool, None] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # Add interfaces for each instrument to self.instruments
        self._interfaces = {
            interface.instrument_name(): interface
            for interface in instrument_interfaces
        }

        self.connections = []

        self.instruments = Parameter(
            get_cmd=lambda: list(self._interfaces.keys()),
            docstring="List of instrument names. Can only be retrieved. "
            "To set, update layout._interfaces",
        )
        self.primary_instrument = Parameter(
            get_cmd=None,
            set_cmd=self._set_primary_instrument,
            vals=vals.Enum(*self._interfaces.keys()),
            docstring="Name of primary instrument, usually the instrument that "
            "performs triggering",
        )
        self.acquisition_instrument = Parameter(
            set_cmd=None,
            initial_value=None,
            vals=vals.Enum(*self._interfaces.keys()),
            docstring="Name of instrument that acquires data",
        )
        self.acquisition_channels = Parameter(
            set_cmd=None,
            vals=vals.Lists(),
            docstring="List of acquisition channels to acquire. "
            "Each element in the list should be a tuple (ch_name, ch_label), "
            "where ch_name is a channel of the acquisition interface, "
            "and ch_label is a given label for that channel (e.g. 'output')."
        )
        self.samples = Parameter(
            set_cmd=None,
            initial_value=1,
            docstring="Number of times to acquire the pulse sequence",
        )

        self.active = Parameter(
            set_cmd=None,
            initial_value=False,
            vals=vals.Bool(),
            docstring="Whether the pulse sequence is being executed. "
            "Can be started/stopped via layout.start/layout.stop",
        )

        self.force_setup = Parameter(
            set_cmd=None,
            initial_value=True,
            vals=vals.Bool(),
            docstring="Setup all instruments if the pulse sequence has changed. "
            "If False, only the instruments are setup if their "
            "respective pulses have changed",
        )

        self.save_trace_channels = Parameter(
            set_cmd=None,
            initial_value=["output"],
            vals=vals.Lists(vals.Strings()),
            docstring="List of channel labels to acquire. "
            "Channel labels are defined in layout.acquisition_channels",
        )
        self.is_acquiring = Parameter(
            set_cmd=None,
            initial_value=False,
            docstring="Whether or not the Layout is performing an acquisition. "
                      "Ensures no two acquisitions are run simultaneously."
        )
        self.sample_rate = Parameter(
            docstring='Acquisition sample rate'
        )

        # Untargeted pulse_sequence, can be set via layout.pulse_sequence
        self._pulse_sequence = None

        # Targeted pulse sequence, which is generated after targeting
        # layout.pulse_sequence
        self.targeted_pulse_sequence = None

        # Handle saving of pulse sequence
        if store_pulse_sequences_folder is not None:
            self.store_pulse_sequences_folder = store_pulse_sequences_folder
        elif silq.config.properties.get("store_pulse_sequences_folder") is not None:
            self.store_pulse_sequences_folder = (
                silq.config.properties.store_pulse_sequences_folder
            )
        else:
            self.store_pulse_sequences_folder = None
        self._pulse_sequences_folder_io = DiskIO(store_pulse_sequences_folder)

        # Shapes of pulse traces, dict of form {pulse_name: {ch_label: shape}}
        self.acquisition_shapes = {}

        # HDF5 files for saving of traces in a loop. One per AcquisitionParameter
        self.trace_files = {}

        # Dictionary for storing all performance timings
        self.timings = PerformanceTimer()

    @property
    def pulse_sequence(self):
        """Target pulse sequence by distributing its pulses to interfaces.

        Each pulse is directed to a connection, either provided by
        ``Pulse.connection`` or ``Pulse.connection_label``.

        Args:
            pulse_sequence: Pulse sequence to be targeted

        Raises:
            AssertionError
                Value passed is not valid pulse sequence
        """
        return self._pulse_sequence

    @pulse_sequence.setter
    def pulse_sequence(self, pulse_sequence):
        assert isinstance(pulse_sequence, PulseSequence), \
            "Can only set layout.pulse_sequence to a PulseSequence"

        # Target pulse_sequence, distributing pulses to interfaces
        self._target_pulse_sequence(pulse_sequence)


    @property
    def acquisition_interface(self):
        """AcquisitionInterface: interface for acquisition instrument

        Acquisition instrument given by parameter

        Returns:
            Acquisition InstrumentInterface
        """
        if self.acquisition_instrument() is not None:
            return self._interfaces[self.acquisition_instrument()]
        else:
            return None

    @parameter
    def sample_rate_get(self, parameter):
        """Union[float, None]: Acquisition sample rate

        If `Layout.acquisition_interface` is not setup, return None

        """
        if self.acquisition_interface is not None:
            return self.acquisition_interface.sample_rate.get_latest()
        else:
            return None

    @parameter
    def sample_rate_set(self, parameter, sample_rate: float):
        """Acquisition sample rate

        If `Layout.acquisition_interface` is not setup, return None

        """
        if self.acquisition_interface is None:
            raise RuntimeError('layout.acquisition_interface not defined')

        self.acquisition_interface.sample_rate(sample_rate)

    def add_connection(self,
                       output_arg: str,
                       input_arg: str,
                       **kwargs):
        """Creates a `SingleConnection` between two instrument interface
           channels.

        Args:
            output_arg: "{instrument}_{channel}" string for connection output.
            input_arg: "{instrument}_{channel}" string for connection input.
            **kwargs: Additional init options for the SingleConnection.

        Returns:
            SingleConnection object

        Notes:
            Both the output and input instruments must already have a
              corresponding interface which should have been passed when
              creating the Layout.

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

    def combine_connections(self,
                            *connections: Sequence[SingleConnection],
                            **kwargs):
        """Combines multiple `SingleConnection` into a `CombinedConnection`.

        This is useful for cases such as when pulses are by default directed to
        multiple connections, e.g. compensated pulses.
        Combined connection is also added to list of existing connections.

        Args:
            *connections: SingleConnections to be combined
            **kwargs: Additional kwargs for CombinedConnection

        Returns:
            CombinedConnection object combining single connections

        """
        connection = CombinedConnection(connections=connections, **kwargs)
        self.connections.append(connection)
        return connection

    def load_connections(self, filepath: Union[str, None] = None,
                         connections_dicts: list = None):
        """Load connections from config or file

        Args:
            filepath: Path to JSON file containing JSON representation of
                connections. If None, ``qc.config.user.connections`` is used.
            connections_dicts: dict to load connections from.
        Returns:
            List[Connection]: Loaded connections

        Raises:
            RuntimeError
                No filepath provided and no connections in config.

        Note:
            Please see other Experiment folder configs for JSON syntax.

        Todo:
            Write documentation/function for how to create JSON connection list.
        """
        self.connections.clear()
        if filepath is not None:
            import os, json
            if not os.path.isabs(filepath):
                # relative path is wrt silq base directory
                filepath = os.path.join(silq.get_SilQ_folder(), filepath)
            with open(filepath, "r") as fp:
                connections = json.load(fp)
        elif connections_dicts is not None:
            connections = connections_dicts
        else:
            connections = silq.config.get('connections', None)
            if connections is None:
                raise RuntimeError('No connections found in config.user')

        for connection in connections:
            # Create a copy of connection, else it changes the config
            connection = connection.copy()
            if 'combine' in connection:
                # Create CombinedConnection. connection['combine'] consists of
                # (output_arg, input_arg) combinations of the SingleConnections
                combined_args = connection.pop('combine')
                # Must add actual Connection objects, so retrieving them from
                #  Layout
                nested_connections = [self.get_connection(output_arg=output_arg,
                                                          input_arg=input_arg)
                                      for output_arg, input_arg in combined_args]
                # Remaining properties in connection dict are kwargs
                self.combine_connections(*nested_connections, **connection)
            else:
                # Properties in connection dict are kwargs
                self.add_connection(**connection)

        return self.connections

    def get_connections(self,
                        connection: Connection = None,
                        output_arg: str = None,
                        output_interface: InstrumentInterface = None,
                        output_instrument: Instrument = None,
                        output_channel: Union[Channel, str] = None,
                        input_arg: str = None,
                        input_interface: InstrumentInterface = None,
                        input_instrument: Instrument = None,
                        input_channel: Union[Channel, str] = None,
                        trigger: bool = None,
                        trigger_start: bool = None,
                        acquire: bool = None,
                        software: bool = None) -> List[Connection]:
        """Get all connections that satisfy connection conditions

        Only conditions that have been set to anything other than None are used.

        Args:
            connection: Specific connection to be checked. If the connection
                is in layout.connections, it returns a list with the connection.
                Can be useful when `Pulse`.connection_requirements needs a
                specific connection. If provided, all other conditions are
                ignored.
            output_arg: string representation of output.

                - SingleConnection has form '{instrument}.{channel}'
                - CombinedConnection is list of SingleConnection output_args,
                  which must have equal number of elements as underlying
                  connections. Can be combined with input_arg, in which
                  case the first element of output_arg and input_arg are
                  assumed to belong to the same underlying connection.

            output_interface: Connections must have output_interface object
            output_instrument: name of output instrument
            output_channel: either Channel or channel name
            input_arg: string representation of input.

                - SingleConnection has form ``{instrument}.{channel}``
                - CombinedConnection is list of SingleConnection output_args,
                  which must have equal number of elements as underlying
                  connections. Can be combined with output_arg, in which
                  case the first element of output_arg and input_arg are
                  assumed to belong to the same underlying connection.

            input_interface: Connections must have input_interface object
            input_instrument: Connections must have input_instrument name
            input_channel: either Channel or channel name
            trigger: Connection is used for triggering
            trigger_start: Connection is used for triggering, but can only
                trigger at the start of the sequence
            acquire: Connection is used for acquisition
            software: Connection is not an actual hardware connection. Used
                when a software trigger needs to be sent

        Returns:
            Connections that satisfy kwarg constraints

        Note:
            If connection condition provided, all other conditions are ignored.

        Raises:
            RuntimeError
                Connection is provided but not found
        """
        if connection is not None:
            # Check if connection is in connections.
            if connection in self.connections:
                return [connection]
            else:
                raise RuntimeError(f"{connection} not found in connections")
        else:
            satisfied_connections = []
            for connection in self.connections:
                if connection.satisfies_conditions(
                        connection=connection,
                        output_arg=output_arg,
                        output_interface=output_interface,
                        output_instrument=output_instrument,
                        output_channel=output_channel,
                        input_arg=input_arg,
                        input_interface=input_interface,
                        input_instrument=input_instrument,
                        input_channel=input_channel,
                        trigger=trigger,
                        trigger_start=trigger_start,
                        acquire=acquire,
                        software=software):
                    satisfied_connections.append(connection)

            return satisfied_connections

    def get_connection(self,
                       connection_label: str = None,
                       output_arg: str = None,
                       output_interface: InstrumentInterface = None,
                       output_instrument: Instrument = None,
                       output_channel: Union[Channel, str] = None,
                       input_arg: str = None,
                       input_interface: InstrumentInterface = None,
                       input_instrument: Instrument = None,
                       input_channel: Union[Channel, str] = None,
                       trigger: bool = None,
                       trigger_start: bool = None,
                       acquire: bool = None,
                       software: bool = None):
        """Get unique connection that satisfies conditions.

        Args:
            connection_label: label specifying specific connection.
                If provided, all other conditions are ignored.
            output_arg: string representation of output.

                * SingleConnection has form ``{instrument}.{channel}``
                * CombinedConnection is list of SingleConnection output_args,
                  which must have equal number of elements as underlying
                  connections. Can be combined with input_arg, in which
                  case the first element of output_arg and input_arg are
                  assumed to belong to the same underlying connection.

            output_interface: Connections must have output_interface object
            output_instrument: name of output instrument
            output_channel: either Channel or channel name
            input_arg: string representation of input.

                * SingleConnection has form ``{instrument}.{channel}``.
                * CombinedConnection is list of SingleConnection output_args,
                  which must have equal number of elements as underlying
                  connections. Can be combined with output_arg, in which
                  case the first element of output_arg and input_arg are
                  assumed to belong to the same underlying connection.

            input_interface: Connections must have input_interface object
            input_instrument: Connections must have input_instrument name
            input_channel: either Channel or channel name
            trigger: Connection is used for triggering
            trigger_start: Connection is used for triggering, but can only
                trigger at the start of the sequence
            acquire: Connection is used for acquisition
            software: Connection is not an actual hardware connection. Used
                when a software trigger needs to be sent

        Returns:
            Connection that satisfies kwarg constraints

        Note:
            If connection_label is provided, all other conditions are ignored.

        Raises:
            AssertionError
                ``connection_label`` is specified but not found
            AssertionError
                No unique connection is found
        """
        if connection_label is not None:
            try:
                return next(connection for connection in self.connections
                            if connection.label == connection_label)
            except StopIteration:
                allowed_labels = [
                    connection.label for connection in self.connections
                    if connection_label is not None
                ]
                raise StopIteration(f'Cannot find connection with label {connection_label}. '
                                    f'Allowed labels: {allowed_labels}')
        else:
            # Extract from conditions other than connection_label
            conditions = dict(output_arg=output_arg,
                              output_interface=output_interface,
                              output_instrument=output_instrument,
                              output_channel=output_channel,
                              input_arg=input_arg,
                              input_interface=input_interface,
                              input_instrument=input_instrument,
                              input_channel=input_channel,
                              trigger=trigger,
                              trigger_start=trigger_start,
                              acquire=acquire,
                              software=software)
            connections = self.get_connections(**conditions)
            filtered_conditions = {key: val for key, val in conditions.items()
                                   if val is not None}
            assert len(connections) == 1, \
                f"Found {len(connections)} connections instead of one satisfying " \
                f"{filtered_conditions}"
            return connections[0]

    def _set_primary_instrument(self, primary_instrument: str):
        """Sets primary instrument, updating ``is_primary`` in all interfaces.
        """
        for instrument_name, interface in self._interfaces.items():
            interface.is_primary(instrument_name == primary_instrument)

    def _get_interfaces_hierarchical(
            self, sorted_interfaces: List[InstrumentInterface] = []
    ) -> List[InstrumentInterface]:
        """Sort interfaces by triggering order, from bottom to top.

        This sorting ensures that earlier instruments never trigger later ones.
        The primary instrument will therefore be one of the last elements.

        Args:
            sorted_interfaces: Sorted list of interfaces. Should start empty,
                and is filled recursively.

        Returns:
            Hierarchically sorted list of interfaces.

        Note:
            This function is recursive. It may fail after reaching the recursion
                limit

        Raises:
            RecursionError
                No hierarchy can be found. Occurs if for instance instruments
                are triggering each other.
        """

        # Set acquisition interface as first interface
        if not sorted_interfaces:
            if self.acquisition_interface is not None:
                sorted_interfaces = [self.acquisition_interface]
            else:
                sorted_interfaces = []

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

    def get_pulse_connection(self,
                             pulse: Pulse,
                             interface: InstrumentInterface = None,
                             instrument: str = None,
                             **kwargs):
        """Get connection a given pulse should be targeted to

        If ``Pulse.connection_label`` is specified, it is used to determine
        the connection. Otherwise, conditions in
        ``Pulse.connection_requirements`` are used to determine connection.

        Args:
            pulse: Pulse for which to find connection
            interface (optional): Output instrument interface of pulse
            instrument (optional): Output instrument name of pulse
            **kwargs: Additional kwargs to specify connection

        Returns:
            Connection object for pulse

        Raises:
            AssertionError
                No unique connection can be found
        """
        if pulse.connection is not None:
           return pulse.connection

        connection_requirements = pulse.connection_requirements.copy()

        if interface is not None:
            connection_requirements['output_interface'] = interface
        elif instrument is not None:
            connection_requirements['output_instrument'] = instrument

        if pulse.connection_label is not None:
            connection = self.get_connection(
                connection_label=pulse.connection_label,
                **connection_requirements,
                **kwargs)
        else:
            connection = self.get_connection(**connection_requirements,
                                             **kwargs)
        return connection

    def _target_pulse(
            self,
            pulse: Pulse,
            copy_pulse: bool = True
    ):
        """Target pulse to corresponding connection and instrument interface.

        The connection is determined from either `Pulse.connection_label`,
        or if undefined, from `Pulse.connection_requirements`. At the
        connection targeting stage, and effects such as an amplitude scale are
        applied.

        Afterwards, the pulse is then directed to the connection's output and
        input interfaces.

        Args:
            pulse: pulse to be targeted
            copy_pulse: whether to copy the pulse when targeting via the connection.
                the main pulses should have this set to True, whereas additional
                pulses should have this set to False.
                Note that a MultiConnection will have this set to True regardless

        Notes:
            * At each targeting stage, the pulse is copied such that any
              modifications don't affect the original pulse.
            * Often the instrument requires additional pulses, such as a
              triggering pulse. These pulses are also targeted by recursively
              calling this function.
            * During targeting, all the properties of pulses are fixed, meaning
              that they do not depend on other pulses anymore. By contrast,
              untargeted pulses can be dependent on other pulses, such as
              starting at the end of the previous pulse, even if the previous
              pulse is modified.

        """
        # Add pulse to acquisition instrument if it must be acquired
        if pulse.acquire:
            self.acquisition_interface.pulse_sequence.quick_add(
                pulse, connect=False, reset_duration=False, nest=True
            )

        if isinstance(pulse, MeasurementPulse):
            # Measurement pulses do not need to be output
            return

        # Get default output instrument
        connection = self.get_pulse_connection(pulse)

        # Return a list of pulses. The reason for not necessarily returing a
        # single pulse is that targeting by a CombinedConnection will target the
        # pulse to each of its connections it's composed of.
        # Copies pulse (multiple times if type is CombinedConnection)
        if isinstance(connection, CombinedConnection):
            # A CombinedConnection targets the pulse to each of its connections
            # that it's composed of. Will create a copy for each connection
            pulses = connection.target_pulse(pulse)
        else:
            pulse = connection.target_pulse(pulse, copy_pulse=copy_pulse)
            pulses = [pulse]

        for pulse in pulses:
            instrument = pulse.connection.output['instrument']
            interface = self._interfaces[instrument]

            # Copies pulse via PulseImplementation.target_pulse
            targeted_pulse = interface.get_pulse_implementation(
                pulse, connections=self.connections)

            assert targeted_pulse is not None, \
                f"Interface {interface} could not target pulse {pulse} using " \
                f"connection {connection}."

            # Set parent of targeted pulse to interface.pulse_sequence
            t_start = targeted_pulse.t_start
            # We find the matching pulse sequence because it could be nested
            targeted_pulse.parent = find_matching_pulse_sequence(
                pulse, interface.pulse_sequence
            )
            # Reset t_start because it will shift if the pulse_sequence does not
            # start at t=0
            targeted_pulse.t_start = t_start

            # Do not copy pulse
            self.targeted_pulse_sequence.quick_add(
                targeted_pulse, connect=False, reset_duration=False, copy=False, nest=True
            )

            # Do not copy pulse
            interface.pulse_sequence.quick_add(
                targeted_pulse, connect=False, reset_duration=False, copy=False,
                nest=True
            )

            # Also add pulse to input interface pulse sequence
            input_interface = self._interfaces[pulse.connection.input['instrument']]
            # Do not copy pulse
            input_interface.input_pulse_sequence.quick_add(
                targeted_pulse, connect=False, reset_duration=False, copy=False,
                nest=True
            )

    def _target_pulse_sequence(
            self,
            pulse_sequence: PulseSequence
    ):
        """Targets a pulse sequence.

        For each of the pulses, it finds the instrument that can output it,
        and adds the pulse to its respective interface. It also takes care of
        any additional necessities, such as additional triggering pulses.

        if `Layout.store_pulse_sequence_folder` is defined, the pulse
        sequence is also stored as a .pickle file.

        Args:
            pulse_sequence: Untargeted pulse sequence that is to be targeted.

        Notes:
            * If a measurement is running, all instruments are stopped
            * The original pulse sequence and pulses remain unmodified
        """
        logger.info(f'Targeting pulse sequence {pulse_sequence}')

        if pulse_sequence.duration > self.maximum_pulse_sequence_duration:
            raise RuntimeError(
                f'Pulse sequence duration {pulse_sequence.duration} exceeds '
                f'maximum duration {self.maximum_pulse_sequence_duration}. '
                f'Please change layout.maximum_pulse_sequence_duration'
            )

        # Create a copy of the pulse sequence
        try:
            # Temporarily set all _connected_to_config to False such that
            # copying the pulse sequence won't attach the copied pulses to the
            # config.
            _connected_to_config_list = []
            for pulse in pulse_sequence.pulses:
                _connected_to_config_list.append(pulse._connected_to_config)
                pulse._connected_to_config = False

            # Copy the pulse sequence
            self._pulse_sequence = pulse_sequence.copy(connect_to_config=False)
        finally:
            # Restore original pulse._connected_to_config values
            for pulse, _connected_to_config in zip(pulse_sequence.pulses,
                                                   _connected_to_config_list):
                pulse._connected_to_config = _connected_to_config

        # Copy untargeted pulse sequence so none of its attributes are modified
        self.targeted_pulse_sequence = PulseSequence()
        # Adopt the same structure as the target pulse sequence, including any
        # nested pulse sequences
        self.targeted_pulse_sequence.clone_skeleton(pulse_sequence)

        # Clear pulses sequences of all instruments
        for interface in self._interfaces.values():
            logger.debug(f'Initializing interface {interface.name}')
            interface.initialize()

            # Clone the structure of the target pulse sequence. This includes
            # fixing the duration of pulse sequence and any nested pulse sequences
            interface.pulse_sequence.clone_skeleton(pulse_sequence)
            interface.input_pulse_sequence.clone_skeleton(pulse_sequence)

        # Add pulses in pulse_sequence to pulse_sequences of instruments
        for pulse in self.pulse_sequence:
            self._target_pulse(pulse)

        # Setup each of the instruments hierarchically using its pulse_sequence
        # The ordering is because instruments might need final pulses from
        # triggering instruments (e.g. triggering pulses that can only be
        # defined once all other pulses have been given)
        for interface in self._get_interfaces_hierarchical():
            if interface.pulse_sequence or interface == self.acquisition_interface:
                additional_pulses = interface.get_additional_pulses(
                    connections=self.connections)
                for pulse in additional_pulses:
                    # These pulses do not need to be copied since they were
                    # generated during targeting
                    self._target_pulse(pulse, copy_pulse=False)


        # Finish setting up the pulse sequences
        # kwarg connect=False is added because else the pulse sequence duration
        # might be changed to that of the last pulse, while it should be fixed
        self.targeted_pulse_sequence.finish_quick_add(connect=False)
        for interface in self._interfaces.values():
            # Finish adding pulses, which performs final steps such as sorting
            # and checking for overlaps
            interface.pulse_sequence.finish_quick_add(connect=False)
            interface.input_pulse_sequence.finish_quick_add(connect=False)

        # Store pulse sequence
        if self.store_pulse_sequences_folder:
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
                    dill.dump(self._pulse_sequence, f)
            except:
                logger.exception(f'Could not save pulse sequence.\n {traceback.format_exc}')

    def update_flags(self,
                     new_flags: Dict[str, Dict[str, Any]]):
        """Updates existing interface ``Layout.flags`` with new flags.

        Flags are instructions sent to interfaces, usually from other
        interfaces, to modify the usual operations.
        Examples are ``skip_start``, ``setup`` kwargs, etc.

        Args:
            new_flags: {instrument: {flag: val}} dict

        See Also:
            `Layout.setup` for information on allowed flags.
        """
        if 'skip_start' in new_flags and new_flags['skip_start'] not in self.flags['skip_start']:
            self.flags['skip_start'].append(new_flags['skip_start'])

        if 'post_start_actions' in new_flags:
            self.flags['post_start_actions'] += new_flags['post_start_actions']

        if 'start_last' in new_flags and new_flags['start_last'] not in self.flags['start_last']:
            self.flags['start_last'].append(new_flags['start_last'])

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

    def setup(self,
              samples: int = None,
              repeat: bool = True,
              ignore: List[str] = [],
              **kwargs):
        """Sets up all the instruments after having targeted a pulse sequence.

        Instruments are setup through their respective interfaces, and only
        the instruments that have a pulse sequence are setup.
        The interface setup order is by hierarchy, i.e. instruments that trigger
        other instruments are never setup before the triggered instruments.
        Any flags, such as to skip starting an instrument, are also collected
        and applied at this stage.

        Interfaces can return a dict of flags. The following flags are accepted:

        * setup (dict): key (instrument_name) value setup_flags.
          When the interface with instrument_name (key) is setup, the
          setup flags (value) will be passed along.
        * skip_start (str): instrument name that should be skipped when
          calling `Layout.start`.
        * post_start_actions (list(callable)): callables to perform after all
          interfaces are started.
        * start_last (interface) interface that should be started after others.

        Args:
            samples: Number of samples (by default uses previous value)
            repeat: Whether to repeat pulse sequence indefinitely or stop at end
            ignore: Interfaces to skip during setup
            **kwargs: additional kwargs sent to all interfaces being setup.
        """

        logger.info(f'Layout setup with {samples} samples '
                    f'{f"and kwargs: {kwargs}" if kwargs else ""}')

        if not self.pulse_sequence:
            raise RuntimeError("Cannot setup with an empty PulseSequence.")

        if self.active():
            self.stop(ignore=ignore)

        # Initialize with empty flags, used for instructions between interfaces
        self.flags = {'setup': {},
                      'skip_start': [],
                      'post_start_actions': [],
                      'start_last': [],
                      'auto_stop': None}

        if samples is not None:
            self.samples(samples)

        if self.acquisition_interface is not None:
            self.acquisition_interface.acquisition_channels(
                [ch_name for ch_name, _ in self.acquisition_channels()])


        for interface in self._get_interfaces_hierarchical():
            if interface.pulse_sequence and interface.instrument_name() not in ignore:

                # Get existing setup flags (if any)
                setup_flags = self.flags['setup'].get(interface.instrument_name(), {})
                if setup_flags:
                    logger.debug(f'{interface.name} setup flags: {setup_flags}')

                input_connections = self.get_connections(input_interface=interface)
                output_connections = self.get_connections(output_interface=interface)

                if not self.force_setup() and not interface.requires_setup(
                    samples=self.samples(),
                    input_connections=input_connections,
                    output_connections=output_connections,
                    repeat=repeat,
                    **setup_flags,
                    **kwargs
                ):
                    logger.debug(f'Skipping setup interface {interface.name}')
                    continue

                with self.timings.record(f'setup.{interface.name}'):
                    logger.debug(f'Setup interface {interface.name}')
                    flags = interface.setup(samples=self.samples(),
                                            input_connections=input_connections,
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
            sample_rate=self.sample_rate(), samples=self.samples())
        self.acquisition_shapes = {}
        output_labels = [output[1] for output in self.acquisition_channels()]
        for pulse_name, shape in trace_shapes.items():
            self.acquisition_shapes[pulse_name] = {
                label: shape for label in output_labels}

    def start(self, auto_stop: Union[bool, float] = False,
              ignore: List[str] = ()):
        """Starts all the instruments except the acquisition instrument.

        The interface start order is by hierarchy, i.e. instruments that trigger
        other instruments are never setup before the triggered instruments.

        Args:
            auto_stop: Call stop after specified duration.
                If not specified, uses value from flags (default is False).
                If set to True, waits 3 times the pulse sequence duration
                If set to a value, waits for that amount of seconds.
            ignore: List of instrument names not to start

        Note:
            Does not start instruments that have the flag ``skip_start``
        """
        self.active(True)

        t0 = time.perf_counter()

        for interface in self._get_interfaces_hierarchical():
            if interface == self.acquisition_interface:
                continue
            elif interface.instrument_name() in self.flags['skip_start']:
                logger.info(f'Skipping starting {interface.name} (flag skip_start)')
                continue
            elif interface.instrument_name() in ignore:
                logger.info(f'Skipping starting {interface.name} (name in ignore list)')
            elif interface.instrument_name() in self.flags['start_last']:
                logger.info(f'Delaying starting {interface.name} (flag start_last)')
                continue
            elif interface.pulse_sequence:
                with self.timings.record(f'start.{interface.name}'):
                    interface.start()
                    logger.debug(f'{interface} started')
            else:
                logger.debug(f'Skipping starting {interface} (no pulse sequence)')

        for interface in self.flags['start_last']:
            with self.timings.record(f'start.{interface.name}'):
                interface.start()
                logger.debug(f'Started {interface} after others (flag start_last)')

        self.timings.record('start.combined', time.perf_counter() - t0)

        for action in self.flags['post_start_actions']:
            action()
            logger.debug(f'Called post_start_action {action}')

        logger.debug('Layout started')

        if auto_stop is None:
            auto_stop = self.flags['auto_stop']

        if auto_stop is not False:
            if auto_stop is True:
                # Wait for three times the duration of the pulse sequence, then stop instruments
                time.sleep(self.pulse_sequence.duration * 3)
            elif isinstance(auto_stop, (int, float)):
                # Wait for duration specified in auto_stop
                time.sleep(auto_stop)
            else:
                raise SyntaxError('auto_stop must be either True or a number')

            self.stop()

    def stop(self, ignore: List[str] = []):
        """Stops all instruments."""

        t0 = time.perf_counter()

        for interface in self._get_interfaces_hierarchical():
            if interface.instrument_name() not in ignore:
                with self.timings.record(f'stop.{interface.name}'):
                    interface.stop()
                    logger.debug(f'{interface} stopped')
        self.active(False)
        logger.debug('Layout stopped')

        self.timings.record('stop.combined', time.perf_counter() - t0)

    def acquisition(self,
                    stop: bool = True,
                    save_traces: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """Performs an acquisition.

        By default this includes starting and stopping of all the instruments.

        Args:
            stop: Whether to stop instruments after finishing
                measurements (True by default)
            save_traces: Whether to save unsegmented acquisition traces

        Returns:
            Dictionary of the form
                {pulse.full_name: {acquisition_label: acquisition_signal}.

        Note:
            Stops all instruments if an error occurs during acquisition.

        See Also:
            `Layout.save_traces`
        """
        if self.is_acquiring():
            raise RuntimeError('Layout is already acquiring data. Exiting...')
        self.is_acquiring = True

        try:
            logger.info(f'Performing acquisition, '
                        f'{"stop" if stop else "continue"} when finished')
            if not self.active():
                self.start()

            assert self.acquisition_interface.pulse_sequence, \
                "None of the pulses have acquire=True, nothing to acquire"

            # Obtain traces from acquisition interface as dict
            with self.timings.record('acquisition'):
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
                    output_label = next(item[1] for item in self.acquisition_channels()
                                        if item[0] == channel)
                    data[pulse][output_label] = trace

            if save_traces:
                self.save_traces()
        except:
            # If any error occurs, stop all instruments
            self.stop()
            raise
        finally:
            self.is_acquiring = False

        return data

    def initialize_trace_file(self,
                              name: str,
                              folder: str = None,
                              channels: List[str] = None,
                              precision: Union[int, None] = 3,
                              compression: int = 4):
        """Initialize an HDF5 file for saving traces

        Args:
            name: Name of trace file.
            folder: Folder path for trace file. If not set, the folder of the
                active loop dataset is used, with subfolder 'traces'.
            channels: List of channel labels to acquire. The channel labels must
                be defined as the second arg of an element in
                ``Layout.acquisition_channels``
            precision: Number of digits after the decimal points to retain.
                Set to 0 for lossless compression
            compression_level: gzip compression level, min=0, max=9

        Raises:
            AssertionError if folder is not provided and no active dataset
        """
        if channels is None:
            channels = self.save_trace_channels()

        active_measurement = qc.active_measurement()
        assert active_measurement is not None, "No active loop found for saving traces"

        if folder is None:
            dataset = qc.active_dataset()
            assert dataset is not None, "No dataset found to save traces to. " \
                                        "Set add_to_dataset=False to save to " \
                                        "separate folder."
            dataset_path = Path(dataset.io.to_path(dataset.location))
            folder = dataset_path / 'traces'
            folder.mkdir(parents=True, exist_ok=True)
        # Create new hdf5 file
        filepath = os.path.join(folder, f'{name}.hdf5')
        assert not os.path.exists(filepath), f"Trace file already exists: {filepath}"
        file = h5py.File(filepath, 'w')

        # Save metadata to traces file
        file.attrs['sample_rate'] = self.sample_rate()
        file.attrs['samples'] = self.samples()
        file.attrs['capture_full_trace'] = self.acquisition_interface.capture_full_trace()
        HDF5Format.write_dict_to_hdf5(
            {'pulse_sequence': self.pulse_sequence.snapshot()}, file)
        HDF5Format.write_dict_to_hdf5(
            {'pulse_shapes': self.pulse_sequence.get_trace_shapes(
                sample_rate=self.sample_rate(), samples=self.samples())}, file)
        HDF5Format.write_dict_to_hdf5(
            {'pulse_slices': self.pulse_sequence.get_trace_slices(
                sample_rate=self.sample_rate(),
                capture_full_traces=self.acquisition_interface.capture_full_trace(),
                return_slice=False
            )}, file)
        HDF5Format.write_dict_to_hdf5(
            {'pulse_slices_full': self.pulse_sequence.get_trace_slices(
                sample_rate=self.sample_rate(),
                capture_full_traces=self.acquisition_interface.capture_full_trace(),
                filter_acquire=False,
                return_slice=False
            )}, file)

        # Create traces group and initialize arrays
        file.create_group('traces')
        data_shape = active_measurement.loop_shape
        if isinstance(data_shape, dict):
            # Measurement is a loop consisting whose loop_shape is a dict.
            # Extract shape using actions indices
            data_shape = data_shape[active_measurement.action_indices]
        # Data is saved in chunks, which is one acquisition
        data_shape += (self.samples(), self.acquisition_interface.points_per_trace())
        for channel in channels:
            file['traces'].create_dataset(name=channel, shape=data_shape,
                                          dtype=float, scaleoffset=precision,
                                          chunks=True, compression='gzip',
                                          compression_opts=compression)
        file.flush()
        return file

    def save_traces(self,
                    name: str = None,
                    folder: str = None,
                    channels: str = None,
                    precision: Union[int, None] = 3,
                    compression: int = 4):
        """Save traces to an HDF5 file.

        The HDF5 file contains a group 'traces', which contains a dataset for
        each channel. These datasets can be massive depending on the size of the
        loop, but shouldn't be an issue since HDF5 can save/load portions of the
        dataset.

        Args:
            name: Name of trace file.
                If not set, the name of the current loop parameter is used
            folder: Folder path for trace file. If not set, the folder of the
                active loop dataset is used, with subfolder 'traces'.
            channels: List of channel labels to acquire. The channel labels must
                be defined as the second arg of an element in
                ``Layout.acquisition_channels``"""
        if channels is None:
            channels = self.save_trace_channels()

        active_measurement = qc.active_measurement()
        assert active_measurement is not None, "No active loop found for saving traces"

        with self.timings.record('save_traces'):
            # Create unique action traces name
            if name is None:  # Set name to current loop action
                active_action = active_measurement.active_action
                action_indices = active_measurement.action_indices
                action_indices_str = '_'.join(map(str, action_indices))
                name = f"{active_action.name}_{action_indices_str}"

            if name in self.trace_files:  # Use existing trace file
                trace_file = self.trace_files[name]
            else:  # Create new trace file
                trace_file = self.initialize_trace_file(name=name, folder=folder)
                self.trace_files[name] = trace_file

            traces = self.acquisition_interface.traces
            for channel in channels:
                # Get corresponding acquisition output channel name (chA etc.)
                ch = next(ch_pair[0] for ch_pair in self.acquisition_channels()
                          if ch_pair[1] == channel)
                trace_file['traces'][channel][active_measurement.loop_indices] = traces[ch]
            trace_file.attrs['final_loop_indices'] = active_measurement.loop_indices

        return trace_file

    def plot_traces(self, channel_filter=None, **plot_kwargs):
        traces_channels = self.acquisition_interface.traces
        # TODO add check if no traces have been acquired

        # Map traces to the channel labels
        channel_mappings = dict(self.acquisition_channels())
        traces = {channel_mappings[ch]: trace_arr
                  for ch, trace_arr in traces_channels.items()
                 }
        trace_shape = next(iter(traces.values())).shape

        # Optionally filter outputs
        if channel_filter:
            if isinstance(channel_filter, str):
                channel_filter = [channel_filter]
            traces = {key: val for key, val in traces.items()
                     if any(name in key for name in channel_filter)}

        t_list = np.arange(trace_shape[1]) / self.sample_rate()
        if not self.acquisition_interface.capture_full_trace():
            t_list += min(self.acquisition_interface.pulse_sequence.t_start_list)
        t_list *= 1e3  # Convert to ms
        sample_list = np.arange(trace_shape[0], dtype=float)

        plot = MatPlot(subplots=len(traces), **plot_kwargs)
        for k, (ax, (channel, traces_arr)) in enumerate(zip(plot, traces.items())):
                if traces_arr.shape[0] == 1:
                    ax.add(traces_arr[0], x=t_list)
                    ax.set_ylabel('Amplitude (V)')
                else:
                    ax.add(traces_arr, x=t_list, y=sample_list)
                ax.set_ylabel('Sample number')
                ax.set_xlabel('Time (ms)')
                ax.set_title(channel)

        plot.tight_layout()

        return plot

    def close_trace_files(self) -> None:
        """Close all opened HDF5 trace files.

         See also:
             Layout.save_traces
        """
        for trace_file in self.trace_files.values():
            trace_file.close()
        self.trace_files.clear()
