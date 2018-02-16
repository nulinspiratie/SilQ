import numpy as np
from collections import OrderedDict, Iterable
from copy import copy
from blinker import signal
from functools import partial
import logging
import re

from qcodes import DataSet, DataArray, MultiParameter, active_data_set
from qcodes.data import hdf5_format
from qcodes import Instrument, MatPlot

from silq import config
from silq.pulses import *
from silq.pulses.pulse_sequences import ESRPulseSequence, NMRPulseSequence, \
    T2ElectronPulseSequence
from silq.analysis import analysis
from silq.tools import data_tools
from silq.tools.general_tools import SettingsClass, clear_single_settings, \
    attribute_from_config, UpdateDotDict, convert_setpoints, \
    property_ignore_setter

__all__ = ['AcquisitionParameter', 'DCParameter', 'TraceParameter',
           'DCSweepParameter', 'EPRParameter', 'ESRParameter',
           'NMRParameter', 'VariableReadParameter',
           'BlipsParameter',
           'NeuralNetworkParameter', 'NeuralRetuneParameter']

logger = logging.getLogger(__name__)
h5fmt = hdf5_format.HDF5Format()


class AcquisitionParameter(SettingsClass, MultiParameter):
    layout = None
    formatter = h5fmt
    store_trace_channels = ['output']

    def __init__(self, continuous=False, environment='default',
                 properties_attrs=None, wrap_set=False, save_traces=False,
                 **kwargs):
        SettingsClass.__init__(self)

        if not hasattr(self, 'pulse_sequence'):
            self.pulse_sequence = PulseSequence()
        """Pulse sequence of acquisition parameter"""

        shapes = kwargs.pop('shapes', ((), ) * len(kwargs['names']))
        MultiParameter.__init__(self, shapes=shapes, wrap_set=wrap_set, **kwargs)

        if self.layout is None:
            try:
                AcquisitionParameter.layout = Instrument.find_instrument('layout')
            except KeyError:
                logger.warning(f'No layout found for {self}')

        self.silent = True
        """Do not print results after acquisition"""

        self.save_traces = save_traces
        """ Save traces in separate files"""

        if environment == 'default':
            environment = config.properties.get('default_environment',
                                                'default')
        self.environment = environment
        self.continuous = continuous

        self.samples = None
        self.traces = None
        self.dataset = None
        self.results = None
        self.base_folder = None
        self.subfolder = None

        # Change attribute data_manager from class attribute to instance
        # attribute. This is necessary to ensure that the data_manager is
        # passed along when the parameter is spawned from a new process
        self.layout = self.layout

        # Attach to properties and parameter configs
        self.properties_attrs = properties_attrs
        self.properties_config = self._attach_to_config(
            path=f'{self.environment}.properties',
            select_attrs=self.properties_attrs)
        self.parameter_config = self._attach_to_config(
            path=f'parameters.{self.name}')

        self._meta_attrs.extend(['label', 'name', 'pulse_sequence'])

    def __repr__(self):
        return '{} acquisition parameter'.format(self.name)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return attribute_from_config(item)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

        # Update value in pulse settings if it exists
        try:
            if key in self.pulse_sequence.pulse_settings:
                self.pulse_sequence.pulse_settings[key] = value
        except AttributeError:
            pass

    @property
    def sample_rate(self):
        """ Acquisition sample rate """
        return self.layout.sample_rate

    def _attach_to_config(self, path, select_attrs=None):
        """
        Attach parameter to a subconfig (within silq config).
        This mean
        s that whenever an item in the subconfig is updated,
        the parameter attribute will also be updated to this value.

        Notification of config updates is handled through blinker signals.

        After successfully attaching to a config, the parameter attributes
        that are present in the config are also updated.

        Args:
            path: subconfig path
            select_attrs: if specified, only update attributes in this list

        Returns:
            subconfig that the parameter is attached to
        """
        # TODO special handling of pulse_sequence attr, etc.
        try:
            # Get subconfig from silq config
            subconfig = config[path]
        except (KeyError, AttributeError):
            # No subconfig exists, not attaching
            return None

        # Get signal handling function
        if select_attrs is not None:
            # Only update attributes in select_attrs
            signal_handler = partial(self._handle_config_signal,
                                     select=select_attrs)
        else:
            signal_handler = self._handle_config_signal

        # Connect changes in subconfig to handling function
        signal(f'config:{path}').connect(signal_handler, weak=False)

        # Set attributes that are present in subconfig
        for attr, val in subconfig.items():
            if select_attrs is None or attr in select_attrs:
                setattr(self, attr, val)

        return subconfig

    def _handle_config_signal(self, _, select=None, **kwargs):
        """
        Update attr when attr in pulse config is modified
        Args:
            _: sender config (unused)
            select (Optional(List(str): list of attrs that can be set.
                Will update any attribute if not specified.
            **kwargs: {attr: new_val}

        Returns:

        """
        key, val = kwargs.popitem()
        if select is None or key in select:
            setattr(self, key, val)

    def store_traces(self, pulse_traces, base_folder=None, subfolder=None,
                     channels=None, setpoints=False):

        if channels is None:
            channels = self.store_trace_channels

        # Store raw traces
        if base_folder is None:
            # Extract base_folder from dataset of currently active loop
            active_dataset = active_data_set()
            if self.base_folder is not None:
                base_folder = self.base_folder
            elif getattr(active_dataset, 'location', None):
                base_folder = active_dataset.location
            elif hasattr(active_dataset, '_location'):
                base_folder = active_dataset._location
            else:
                base_folder = DataSet.location_provider(DataSet.default_io)
                subfolder = None

        if subfolder is None and base_folder is not None:
                subfolder = f'traces_{self.name}'

        self.dataset = data_tools.create_data_set(name='traces',
                                                  base_folder=base_folder,
                                                  subfolder=subfolder,
                                                  formatter=self.formatter)

        traces_dict = {}
        for pulse_name, channel_traces in pulse_traces.items():
            for channel in channels:
                traces_name = f'{pulse_name}_{channel}'
                traces = channel_traces[channel]
                traces_dict[traces_name] = traces

        if setpoints:
            # Create dictionary of set arrays
            set_arrs = {}
            for traces_name, traces in traces_dict.items():
                number_of_traces, points_per_trace = traces.shape

                if traces.shape not in set_arrs:
                    time_step = 1 / self.sample_rate
                    t_list = np.arange(0, points_per_trace * time_step,
                                       time_step)
                    t_list_arr = DataArray(
                        name='time',
                        array_id='time',
                        label=' Time',
                        unit='s',
                        shape=traces.shape,
                        preset_data=np.full(traces.shape, t_list),
                        is_setpoint=True)

                    trace_num_arr = DataArray(
                        name='trace_num',
                        array_id='trace_num',
                        label='Trace',
                        unit='num',
                        shape=(number_of_traces, ),
                        preset_data=np.arange(number_of_traces,
                                              dtype=np.float64),
                        is_setpoint=True)
                    set_arrs[traces.shape] = (trace_num_arr, t_list_arr)

            # Add set arrays to dataset
            for k, (t_list_arr, trace_num_arr) in enumerate(set_arrs.values()):
                for arr in (t_list_arr, trace_num_arr):
                    if len(set_arrs) > 1:
                        # Need to give individual array_ids to each of the set arrays
                        arr.array_id += '_{}'.format(k)
                    self.dataset.add_array(arr)
            set_arrays = (t_list_arr, trace_num_arr)
        else:
            set_arrays = ()

        # Add trace arrs to dataset
        for traces_name, traces in traces_dict.items():
            # Must transpose traces array
            trace_arr = DataArray(name=traces_name,
                                  array_id=traces_name,
                                  label=traces_name + ' signal',
                                  unit='V',
                                  shape=traces.shape,
                                  preset_data=traces,
                                  set_arrays=set_arrays)
            self.dataset.add_array(trace_arr)

        self.dataset.finalize(write_metadata=False)

    def print_results(self):
        names = self.names if self.names is not None else [self.name]
        for name in names:
            value = self.results[name]
            if isinstance(value, (int, float)):
                print(f'{name}: {value:.3f}')
            else:
                print(f'{name}: {value}')

    def setup(self, start=None, **kwargs):
        if not self.pulse_sequence.up_to_date():
            self.pulse_sequence.generate()

        self.layout.pulse_sequence = self.pulse_sequence

        samples = kwargs.pop('samples', self.samples)
        self.layout.setup(samples=samples, **kwargs)

        if start is None:
            start = self.continuous

        if start:
            self.layout.start()

    def acquire(self, stop=None, setup=None, **kwargs):
        """
        Performs a layout.acquisition. The result is stored in self.traces
        Args:
            stop (Bool): Whether to stop instruments after acquisition.
                If not specified, it will stop if self.continuous is False
            setup (Bool): Whether to setup layout before acquisition.
                If not specified, it will setup if pulse_sequences are different
            **kwargs: Additional kwargs to be given to layout.acquisition

        Returns:
            acquisition output
        """
        if stop is None:
            stop = not self.continuous

        if not self.pulse_sequence.up_to_date():
            self.pulse_sequence.generate()

        if setup or (setup is None and
                     self.layout.pulse_sequence != self.pulse_sequence) or \
                self.layout.samples() != self.samples:
            self.setup()


        # Perform acquisition
        self.traces = self.layout.acquisition(stop=stop, **kwargs)
        return self.traces

    def analyse(self, traces):
        raise NotImplementedError('`analyse` must be implemented in subclass')

    def plot_traces(self, filter=None, channels=['output']):
        plot_traces = OrderedDict()
        for pulse_name, trace in self.traces.items():
            if filter is not None:
                if isinstance(filter, str):
                    filter = [filter]
                if not any(elem in pulse_name for elem in filter):
                    continue

            plot_traces[pulse_name] = trace

        if len(channels) > 1:
            subplots = (len(plot_traces), len(channels))
        else:
            subplots = len(plot_traces)
        plot = MatPlot(subplots=subplots)

        k = 0
        for pulse_name, traces in plot_traces.items():
            for channel in channels:
                trace_arr = traces[channel]
                pts = trace_arr.shape[-1]
                t_list = np.linspace(0, pts / self.sample_rate, pts,
                                     endpoint=False)
                if trace_arr.ndim == 2:
                    plot[k].add(traces[channel], x=t_list,
                                y=np.arange(trace_arr.shape[0], dtype=float))
                else:
                    plot[k].add(traces[channel], x=t_list)
                plot[k].set_xlabel('Time (s)')
                plot[k].set_title(pulse_name)
                k += 1
        plot.tight_layout()
        return plot


    @clear_single_settings
    def get_raw(self):
        self.traces = self.acquire()

        self.results = self.analyse(self.traces)

        if self.save_traces:
            self.store_traces(self.traces)

        if not self.silent:
            self.print_results()

        return tuple(self.results[name] for name in self.names)

    def set(self, **kwargs):
        return self.single_settings(**kwargs)()


class PulseSequenceAcquisitionParameter(AcquisitionParameter):
    def __init__(self, name, tile_pulses=[], samples=1, **kwargs):
        self.tile_pulses = tile_pulses
        super().__init__(name=name, names=[], shapes=[], **kwargs)
        self.samples = samples
        self.tile_pulses = tile_pulses

    @property_ignore_setter
    def names(self):
        acquire_pulses = self.pulse_sequence.get_pulses(acquire=True)
        acquire_pulse_names = [pulse.name for pulse in acquire_pulses
                               if not pulse.name in self.tile_pulses]
        acquire_pulse_names.extend(self.tile_pulses)
        return acquire_pulse_names

    @property_ignore_setter
    def labels(self):
        return self.names

    @property_ignore_setter
    def shapes(self):
        trace_shapes = self.pulse_sequence.get_trace_shapes(self.sample_rate,
                                                            self.samples)

        # Tile pulses
        for acquire_pulse_name in self.names:
            if acquire_pulse_name not in self.tile_pulses:
                continue

            pulse_trace_shapes = []
            for pulse_name, pulse_shape in list(trace_shapes.items()):
                if (pulse_name == acquire_pulse_name
                    or pulse_name.startswith(f'{acquire_pulse_name}[')):
                    pulse_trace_shapes.append(pulse_shape)
                    trace_shapes.pop(pulse_name)

            pulse_trace_shape = list(pulse_trace_shapes[0])
            pulse_trace_shape[-1] = sum(shape[-1] for shape in pulse_trace_shapes)

            trace_shapes[acquire_pulse_name] = pulse_trace_shape

        return [tuple(trace_shapes[name]) for name in self.names]

    def analyse(self, traces):
        tile_indices = {name: 0 for name in self.names}

        tiled_traces = {acquire_pulse_name: np.zeros(shape)
                        for acquire_pulse_name, shape in zip(self.names,
                                                             self.shapes)}
        for acquire_pulse_name in self.names:
            if acquire_pulse_name not in self.tile_pulses:
                tiled_traces[acquire_pulse_name] = traces[acquire_pulse_name]['output']
            else:
                tile_traces = {pulse_name: trace for pulse_name, trace in traces.items()
                               if (pulse_name == acquire_pulse_name
                                   or pulse_name.startswith(f'{acquire_pulse_name}['))}
                if len(tile_traces) == 1:
                    tiled_traces[acquire_pulse_name] = next(iter(tiled_traces.values()))
                else:
                    pulse_ids = sorted(int(pulse_name.split('[')[1].rstrip(']'))
                                       for pulse_name in traces)
                    for pulse_id in pulse_ids:
                        trace = traces[f'{acquire_pulse_name}[{pulse_id}]']['output']

                        if isinstance(trace, np.ndarray):
                            idx_increment = trace.shape[-1]
                        else:
                            idx_increment = 1

                        tile_slice = slice(tile_indices[acquire_pulse_name],
                                           tile_indices[acquire_pulse_name]+idx_increment)
                        tiled_traces[acquire_pulse_name][..., tile_slice] = trace
                        tile_indices[acquire_pulse_name] += idx_increment

        return tiled_traces


class DCParameter(AcquisitionParameter):
    # TODO implement continuous acquisition
    def __init__(self, name='DC', unit='V', **kwargs):
        self.pulse_sequence = PulseSequence([
            DCPulse(name='DC', acquire=True, average='point'),
            DCPulse(name='DC_final')])

        super().__init__(name=name,
                         names=['DC_voltage'],
                         labels=['DC voltage'],
                         units=[unit],
                         snapshot_value=False,
                         continuous = True,
                         **kwargs)
        self.samples = 1

    def analyse(self, traces):
        return {'DC_voltage': traces['DC']['output']}


class TraceParameter(AcquisitionParameter):
    """An acquisition parameter for obtaining a trace or multiple traces
    of a given PulseSequence.

    A generic initial PulseSequence is defined, but can be redefined at
    run-time.
    e.g.
        parameter.average_mode = 'none'
        parameter.pulse_sequence = my_pulse_sequence

    Note that for the above example, all pulses in my_pulse_sequence will be
    copied.

    """
    def __init__(self, name='trace_pulse', average_mode='none', **kwargs):
        self._average_mode = average_mode
        self._pulse_sequence = PulseSequence()
        self.trace_pulse = MeasurementPulse(name=name, duration=1e-3,
                                            average=self.average_mode)

        super().__init__(name='Trace_acquisition',
                         names=self.names,
                         labels=self.labels,
                         units=self.units,
                         shapes=self.shapes,
                         snapshot_value=False,
                         **kwargs)
        self.samples = 1

    @property
    def average_mode(self):
        return self._average_mode

    @average_mode.setter
    def average_mode(self, mode):
        if (self._average_mode != mode):
            self._average_mode = mode
            if self.trace_pulse is not None:
                self.trace_pulse.average = mode

    @property
    def pulse_sequence(self):
        return self._pulse_sequence

    @pulse_sequence.setter
    def pulse_sequence(self, pulse_sequence):
        self._pulse_sequence = copy(pulse_sequence)

    @property_ignore_setter
    def names(self):
        return tuple(self.trace_pulse.full_name + f'_{output[1]}'
                     for output in self.layout.acquisition_channels())

    @property_ignore_setter
    def labels(self):
        return tuple(f'{output[1]} Trace'
                for output in self.layout.acquisition_channels())

    @property_ignore_setter
    def units(self):
        return ('V', ) * len(self.layout.acquisition_channels())

    @property_ignore_setter
    def shapes(self):
        if self.trace_pulse in self.pulse_sequence:
            trace_shapes = self.pulse_sequence.get_trace_shapes(
                self.layout.sample_rate, self.samples)
            trace_pulse_shape = tuple(trace_shapes[self.trace_pulse.full_name])
            if self.samples > 1 and self.average_mode == 'none':
                return ((trace_pulse_shape,),) * len(self.layout.acquisition_channels())
            else:
                return ((trace_pulse_shape[1], ), ) * \
                        len(self.layout.acquisition_channels())
        else:
            return ((1,),) * len(self.layout.acquisition_channels())


    @property_ignore_setter
    def setpoints(self):
        if self.trace_pulse in self.pulse_sequence:
            duration = self.trace_pulse.duration
        else:
            return ((1,),) * len(self.layout.acquisition_channels())

        num_traces = len(self.layout.acquisition_channels())

        pts = int(round(duration * self.sample_rate))
        t_list = tuple(np.linspace(0, duration, pts, endpoint=True))

        if self.samples > 1 and self.average_mode == 'none':
            setpoints = ((tuple(np.arange(self.samples, dtype=float)),
                          t_list), ) * num_traces
        else:
            setpoints = ((t_list, ), ) * num_traces
        return setpoints

    @property_ignore_setter
    def setpoint_names(self):
        if self.samples > 1 and self.average_mode == 'none':
            return (('sample', 'time', ), ) * \
                   len(self.layout.acquisition_channels())
        else:
            return (('time', ), ) * len(self.layout.acquisition_channels())


    @property_ignore_setter
    def setpoint_units(self):
        if self.samples > 1 and self.average_mode == 'none':
            return ((None, 's', ), ) * len(self.layout.acquisition_channels())
        else:
            return (('s', ), ) * len(self.layout.acquisition_channels())


    def setup(self, start=None, **kwargs):
        """
        Modifies provided pulse sequence by creating a single
        pulse which overlaps all other pulses with acquire=True and
        then acquires only this pulse.
        """

        acquired_pulses = self.pulse_sequence.get_pulses(acquire=True)

        if not acquired_pulses:
            raise RuntimeError('PulseSequence has no pulses to acquire.')

        # Find the start and stop times for all acquired pulses
        t_start = min(pulse.t_start for pulse in acquired_pulses)
        t_stop = max(pulse.t_stop for pulse in acquired_pulses)

        self.trace_pulse.t_start = t_start
        self.trace_pulse.t_stop = t_stop

        # Ensure that each pulse is not acquired as this could cause
        # overlapping issues
        for pulse in self.pulse_sequence.get_pulses():
            if pulse is self.trace_pulse:
                continue
            pulse.acquire = False

        if self.trace_pulse.full_name in self.pulse_sequence:
            self.pulse_sequence.remove(self.trace_pulse.full_name)
        self.pulse_sequence.add(self.trace_pulse)

        super().setup(start=start, **kwargs)

    def acquire(self, **kwargs):
        """
        Acquires the number of traces defined in self.samples

        return:  A tuple of data points. e.g.
                 ((data_for_1st_output), (data_for_2nd_output), ...)
        """
        super().acquire(**kwargs)

        traces = {self.trace_pulse.full_name + '_' + output:
                      self.traces[self.trace_pulse.full_name][output]
                  for _, output in self.layout.acquisition_channels()}

        return traces

    def analyse(self, traces):
        return {self.names[k] : traces[name].tolist()[0]
                for k, name in enumerate(traces)}


class DCSweepParameter(AcquisitionParameter):
    def __init__(self, name='DC_sweep', **kwargs):

        self.sweep_parameters = OrderedDict()
        # Pulse to acquire trace at the end, disabled by default
        self.trace_pulse = DCPulse(name='trace', duration=100e-3, enabled=False,
                                   acquire=True, average='trace', amplitude=0)

        self.pulse_duration = 1e-3
        self.final_delay = 120e-3
        self.inter_delay = 200e-6
        self.use_ramp = False

        self.additional_pulses = []

        super().__init__(name=name, names=['DC_voltage'],
                         labels=['DC voltage'], units=['V'],
                         snapshot_value=False, setpoint_names=(('None',),),
                         shapes=((1,),), **kwargs)
        self.samples = 1

    def __getitem__(self, item):
        return self.sweep_parameters[item]

    @property_ignore_setter
    def setpoints(self):
        iter_sweep_parameters = iter(self.sweep_parameters.values())
        if len(self.sweep_parameters) == 1:
            sweep_dict = next(iter_sweep_parameters)
            sweep_voltages = sweep_dict.sweep_voltages
            if sweep_dict.offset_parameter is not None:
                sweep_voltages = sweep_voltages + sweep_dict.offset_parameter.get_latest()
            setpoints = (convert_setpoints(sweep_voltages),),

        elif len(self.sweep_parameters) == 2:
            inner_sweep_dict = next(iter_sweep_parameters)
            inner_sweep_voltages = inner_sweep_dict.sweep_voltages
            if inner_sweep_dict.offset_parameter is not None:
                inner_sweep_voltages = inner_sweep_voltages + inner_sweep_dict.offset_parameter.get_latest()
            outer_sweep_dict = next(iter_sweep_parameters)
            outer_sweep_voltages = outer_sweep_dict.sweep_voltages
            if outer_sweep_dict.offset_parameter is not None:
                outer_sweep_voltages = outer_sweep_voltages + outer_sweep_dict.offset_parameter.get_latest()

            setpoints = (convert_setpoints(outer_sweep_voltages,
                                           inner_sweep_voltages)),

        if self.trace_pulse.enabled:
            # Also obtain a time trace at the end
            points = round(self.trace_pulse.duration * self.sample_rate)
            trace_setpoints = tuple(
                np.linspace(0, self.trace_pulse.duration, points))
            setpoints += (convert_setpoints(trace_setpoints),)
        return setpoints

    @property_ignore_setter
    def names(self):
        if self.trace_pulse.enabled:
            return ('DC_voltage', 'trace_voltage')
        else:
            return ('DC_voltage',)

    @property_ignore_setter
    def labels(self):
        if self.trace_pulse.enabled:
            return ('DC voltage', 'Trace voltage')
        else:
            return ('DC voltage',)

    @property_ignore_setter
    def units(self):
        return ('V', 'V') if self.trace_pulse.enabled else ('V',)

    @property_ignore_setter
    def shapes(self):
        iter_sweep_parameters = iter(self.sweep_parameters.values())
        if len(self.sweep_parameters) == 0:
            shapes = (),
        elif len(self.sweep_parameters) == 1:
            sweep_voltages = next(iter_sweep_parameters).sweep_voltages
            shapes = (len(sweep_voltages),),
        elif len(self.sweep_parameters) == 2:
            inner_sweep_voltages = next(iter_sweep_parameters).sweep_voltages
            outer_sweep_voltages = next(iter_sweep_parameters).sweep_voltages
            shapes = (len(outer_sweep_voltages), len(inner_sweep_voltages)),

        if self.trace_pulse.enabled:
            shapes += (round(
                self.trace_pulse.duration * self.sample_rate),),
        return shapes

    @property_ignore_setter
    def setpoint_names(self):
        iter_sweep_parameters = reversed(self.sweep_parameters.keys())
        names = tuple(iter_sweep_parameters),
        if self.trace_pulse.enabled:
            names += (('time',), )
        return names

    @property_ignore_setter
    def setpoint_units(self):
        setpoint_units = (('V',) * len(self.sweep_parameters),)
        if self.trace_pulse.enabled:
            setpoint_units += (('s',), )
        return setpoint_units

    def add_sweep(self, parameter_name, sweep_voltages=None,
                  connection_label=None, offset_parameter=None):
        if connection_label is None:
            connection_label = parameter_name

        self.sweep_parameters[parameter_name] = UpdateDotDict(
            update_function=self.generate, name=parameter_name,
            sweep_voltages=sweep_voltages, connection_label=connection_label,
            offset_parameter=offset_parameter)

        self.generate()

    def generate(self):
        self.pulse_sequence.clear()

        iter_sweep_parameters = iter(self.sweep_parameters.items())
        if len(self.sweep_parameters) == 1:
            sweep_name, sweep_dict = next(iter_sweep_parameters)
            sweep_voltages = sweep_dict.sweep_voltages
            connection_label = sweep_dict.connection_label
            if self.use_ramp:
                sweep_points = len(sweep_voltages)
                pulses = [DCRampPulse('DC_inner',
                                      duration=self.pulse_duration*sweep_points,
                                      amplitude_start=sweep_voltages[0],
                                      amplitude_stop=sweep_voltages[-1],
                                      acquire=True,
                                      average=f'point_segment:{sweep_points}',
                                      connection_label=connection_label)]
            else:
                pulses = [
                    DCPulse('DC_inner', duration=self.pulse_duration,
                            acquire=True, average='point',
                            amplitude=sweep_voltage,
                            connection_label=connection_label)
                for sweep_voltage in sweep_voltages]

            self.pulse_sequence = PulseSequence(pulses=pulses)
            #             self.pulse_sequence.add(*self.additional_pulses)

        elif len(self.sweep_parameters) == 2:
            inner_sweep_name, inner_sweep_dict = next(iter_sweep_parameters)
            inner_sweep_voltages = inner_sweep_dict.sweep_voltages
            inner_connection_label = inner_sweep_dict.connection_label
            outer_sweep_name, outer_sweep_dict = next(iter_sweep_parameters)
            outer_sweep_voltages = outer_sweep_dict.sweep_voltages
            outer_connection_label = outer_sweep_dict.connection_label

            pulses = []
            if outer_connection_label == inner_connection_label:
                if self.use_ramp:
                    raise NotImplementedError('Ramp Pulse not implemented for '
                                              'CombinedConnection')
                for outer_sweep_voltage in outer_sweep_voltages:
                    for inner_sweep_voltage in inner_sweep_voltages:
                        sweep_voltage = (
                            inner_sweep_voltage, outer_sweep_voltage)
                        pulses.append(
                            DCPulse('DC_read', duration=self.pulse_duration,
                                    acquire=True, amplitude=sweep_voltage,
                                    average='point',
                                    connection_label=outer_connection_label))
            else:
                t = 0
                sweep_duration = self.pulse_duration * len(inner_sweep_voltages)
                for outer_sweep_voltage in outer_sweep_voltages:
                    pulses.append(
                        DCPulse('DC_outer', t_start=t,
                                duration=sweep_duration + self.inter_delay,
                                amplitude=outer_sweep_voltage,
                                connection_label=outer_connection_label))
                    if self.inter_delay > 0:
                        pulses.append(
                            DCPulse('DC_inter_delay', t_start=t,
                                    duration=self.inter_delay,
                                    amplitude=inner_sweep_voltages[0],
                                    connection_label=inner_connection_label))
                        t += self.inter_delay

                    if self.use_ramp:
                        sweep_points = len(inner_sweep_voltages)
                        pulses.append(
                            DCRampPulse('DC_inner', t_start=t,
                                        duration=sweep_duration,
                                        amplitude_start=inner_sweep_voltages[0],
                                        amplitude_stop=inner_sweep_voltages[-1],
                                        acquire=True,
                                        average=f'point_segment:{sweep_points}',
                                        connection_label=inner_connection_label)
                        )
                        t += sweep_duration
                    else:
                        for inner_sweep_voltage in inner_sweep_voltages:
                            pulses.append(
                                DCPulse('DC_inner', t_start=t,
                                        duration=self.pulse_duration,
                                        acquire=True, average='point',
                                        amplitude=inner_sweep_voltage,
                                        connection_label=inner_connection_label)
                            )
                            t += self.pulse_duration

        else:
            raise NotImplementedError(
                f"Cannot handle {len(self.sweep_parameters)} parameters")

        if self.trace_pulse.enabled:
            # Also obtain a time trace at the end
            pulses.append(self.trace_pulse)

        self.pulse_sequence = PulseSequence(pulses=pulses)
        self.pulse_sequence.final_delay = self.final_delay

    def analyse(self, traces):
        # Process results
        DC_voltages = np.array(
            [traces[pulse.full_name]['output'] for pulse in
             self.pulse_sequence.get_pulses(name='DC_inner')])

        if self.use_ramp:
            if len(self.sweep_parameters) == 1:
                results = {'DC_voltage': DC_voltages[0]}
            elif len(self.sweep_parameters) == 2:
                results = {'DC_voltage': DC_voltages}
        else:
            if len(self.sweep_parameters) == 1:
                results = {'DC_voltage': DC_voltages}
            elif len(self.sweep_parameters) == 2:
                results = {'DC_voltage':
                    DC_voltages.reshape(self.shapes[0])}

        if self.trace_pulse.enabled:
            results['trace_voltage'] = traces['trace']['output']

        return results


class EPRParameter(AcquisitionParameter):
    def __init__(self, name='EPR', **kwargs):
        self.pulse_sequence = PulseSequence([
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read_long', acquire=True),
            DCPulse('final')])

        super().__init__(name=name,
                         names=['contrast', 'dark_counts',
                                'voltage_difference_read',
                                'fidelity_empty', 'fidelity_load'],
                         snapshot_value=False,
                         properties_attrs=['t_skip', 't_read',
                                           'min_filter_proportion',
                                           'filter_traces'],
                         **kwargs)

    @property_ignore_setter
    def labels(self):
        return [name.replace('_', ' ').capitalize() for name in self.names]

    def analyse(self, traces):
        return analysis.analyse_EPR(
            empty_traces=self.traces['empty']['output'],
            plunge_traces=self.traces['plunge']['output'],
            read_traces=self.traces['read_long']['output'],
            sample_rate=self.sample_rate,
            t_skip=self.t_skip,
            t_read=self.t_read,
            min_filter_proportion=self.min_filter_proportion,
            filter_traces=self.filter_traces)


class ESRParameter(AcquisitionParameter):
    def __init__(self, name='ESR', **kwargs):
        """
        Parameter used to perform electron spin resonance (ESR)
        """
        self._names = []

        self.pulse_sequence = ESRPulseSequence()
        self.ESR = self.pulse_sequence.ESR
        self.EPR = self.pulse_sequence.EPR
        self.pre_pulses = self.pulse_sequence.pre_pulses
        self.post_pulses = self.pulse_sequence.post_pulses

        super().__init__(name=name,
                         names=['contrast', 'dark_counts',
                                'voltage_difference_read'],
                         snapshot_value=False,
                         properties_attrs=['t_skip', 't_read',
                                           'min_filter_proportion',
                                           'filter_traces'],
                         **kwargs)

    @property
    def names(self):
        if self.EPR['enabled']:
            names = copy(self._names)
        else:
            # Ignore all names, only add the ESR up proportions
            names = []

        ESR_pulse_names = [pulse if isinstance(pulse, str) else pulse.name
                           for pulse in self.ESR['pulses']]

        for pulse in self.ESR['pulses']:
            pulse_name = pulse if isinstance(pulse, str) else pulse.name

            if ESR_pulse_names.count(pulse_name) == 1:
                # Ignore suffix
                name = pulse_name
            else:
                suffix= len([name for name in names
                             if f'up_proportion_{pulse_name}' in name])
                name = f'{pulse_name}_{suffix}'
            names.append(f'up_proportion_{name}')
            if self.EPR['enabled']:
                names.append(f'contrast_{name}')
        return names

    @names.setter
    def names(self, names):
        self._names = [name for name in names
                       if not 'contrast_' in name
                       and not 'up_proportion_' in name]

    @property_ignore_setter
    def shapes(self):
        return ((), ) * len(self.names)

    @property_ignore_setter
    def units(self):
        return ('', ) * len(self.names)

    @property_ignore_setter
    def labels(self):
        return [name[0].capitalize() + name[1:].replace('_', ' ')
                for name in self.names]

    @property
    def ESR_frequencies(self):
        return [pulse.frequency if isinstance(pulse, Pulse)
                else self.ESR[pulse].frequency
                for pulse in self.ESR['pulses']]

    @ESR_frequencies.setter
    def ESR_frequencies(self, ESR_frequencies):
        if len(ESR_frequencies) != len(self.ESR['pulses']):
            logger.warning('Different number of frequencies. '
                           'Reprogramming ESR pulses to default ESR_pulse')

        self.pulse_sequence.generate(ESR_frequencies=ESR_frequencies)

    def analyse(self, traces):
        if self.EPR['enabled']:
            # Analyse EPR sequence, which also gets the dark counts
            results = analysis.analyse_EPR(
                empty_traces=traces['empty']['output'],
                plunge_traces=traces['plunge']['output'],
                read_traces=traces['read_long']['output'],
                sample_rate=self.sample_rate,
                min_filter_proportion=self.min_filter_proportion,
                filter_traces=self.filter_traces,
                t_skip=self.t_skip, # Use t_skip to keep length consistent
                t_read=self.t_read)
        else:
            results = {}

        ESR_pulse_names = [pulse if isinstance(pulse, str) else pulse.name
                           for pulse in self.ESR['pulses']]
        read_pulses = self.pulse_sequence.get_pulses(name=self.ESR["read_pulse"].name)
        results['ESR_results'] = []
        for read_pulse, ESR_pulse in zip(read_pulses, self.ESR['pulses']):
            read_traces = traces[read_pulse.full_name]['output']
            ESR_results = analysis.analyse_traces(
                traces=read_traces,
                sample_rate=self.sample_rate,
                filter='low' if self.filter_traces else None,
                min_filter_proportion=self.min_filter_proportion,
                t_skip=self.t_skip,
                t_read=self.t_read)
            results['ESR_results'].append(ESR_results)

            # Extract ESR pulse labels
            if ESR_pulse_names.count(ESR_pulse.name) == 1:
                # Ignore suffix
                pulse_label = ESR_pulse.name
            else:
                suffix = len([name for name in results
                              if f'up_proportion_{ESR_pulse.name}' in name])
                pulse_label = f'{ESR_pulse.name}_{suffix}'

            # Add up proportion and dark counts
            results[f'up_proportion_{pulse_label}'] = ESR_results['up_proportion']
            if self.EPR['enabled']:
                # Add contrast obtained by subtracting EPR dark counts
                contrast = ESR_results['up_proportion'] - results['dark_counts']
                results[f'contrast_{pulse_label}'] = contrast

        self.results = results
        return results


class NMRParameter(AcquisitionParameter):

    def __init__(self, name='NMR',
                 names=['flips', 'flip_probability', 'up_proportions'],
                 **kwargs):
        """
        Parameter used to determine the Rabi frequency
        """
        self.pulse_sequence = NMRPulseSequence()
        self.NMR = self.pulse_sequence.NMR
        self.ESR = self.pulse_sequence.ESR
        self.pre_pulses = self.pulse_sequence.pulse_settings['pre_pulses']
        self.post_pulses = self.pulse_sequence.pulse_settings['post_pulses']

        super().__init__(name=name,
                         names=names,
                         snapshot_value=False,
                         properties_attrs=['t_read', 't_skip', 'threshold_up_proportion'],
                         **kwargs)

    @property
    def names(self):
        names = []

        for name in self._names:
            if name in ['flips', 'flip_probability', 'up_proportions']:
                if len(self.ESR_frequencies) == 1:
                    names.append(name)
                else:
                    names += [f'{name}_{k}'
                              for k in range(len(self.ESR_frequencies))]
            elif name in ['combined_flips', 'combined_flip_probability',
                          'filtered_combined_flips',
                          'filtered_combined_flip_probability'] and \
                            len(self.ESR_frequencies) > 1:
                names += [f'{name}_{k}{k+1}'
                          for k in range(len(self.ESR_frequencies) - 1)]
            elif name in ['filtered_flips', 'filtered_flip_probability'] and \
                            len(self.ESR_frequencies) > 1:
                for k in range(0, len(self.ESR_frequencies)):
                    if k > 0:
                        names.append(f'{name}_{k}_{k-1}{k}')
                    if k < len(self.ESR_frequencies) - 1:
                        names.append(f'{name}_{k}_{k}{k+1}')
        return names

    @names.setter
    def names(self, names):
        self._names = names

    @property_ignore_setter
    def shapes(self):
        return tuple((self.samples,) if 'up_proportions' in name else ()
                     for name in self.names)

    @property_ignore_setter
    def units(self):
        return ('', ) * len(self.names)

    @property_ignore_setter
    def labels(self):
        return [name.replace('_', ' ').capitalize() for name in self.names]

    @property
    def ESR_frequencies(self):
        return [pulse.frequency if isinstance(pulse, Pulse)
                else self.ESR[pulse].frequency
                for pulse in self.ESR['pulses']]

    @ESR_frequencies.setter
    def ESR_frequencies(self, ESR_frequencies):
        if len(ESR_frequencies) != len(self.ESR['pulses']):
            logger.warning('Different number of frequencies. '
                           'Reprogramming ESR pulses to default ESR_pulse')
            self.ESR['pulses']= [copy(self.ESR['pulse'])
                                 for _ in range(len(ESR_frequencies))]
        self.ESR['pulses'] = [copy(self.ESR[p]) if isinstance(p, str) else p
                              for p in self.ESR['pulses']]
        for pulse, ESR_frequency in zip(self.ESR['pulses'], ESR_frequencies):
            pulse.frequency = ESR_frequency

    def analyse(self, traces):
        results = {'results_read': []}

        # Calculate threshold voltages from combined read traces
        high_low = analysis.find_high_low(
            np.ravel([trace['output'] for trace in traces.values()]))
        threshold_voltage = high_low['threshold_voltage']

        # Extract points per shot from a single read trace
        single_read_traces_name = f"{self.ESR['read_pulse'].name}[0]"
        single_read_traces = traces[single_read_traces_name]['output']
        points_per_shot = single_read_traces.shape[1]

        self.read_traces = np.zeros((len(self.ESR_frequencies), self.samples,
                                     self.ESR['shots_per_frequency'],
                                     points_per_shot))
        up_proportions = np.zeros((len(self.ESR_frequencies), self.samples))
        for f_idx, ESR_frequency in enumerate(self.ESR_frequencies):
            for sample in range(self.samples):
                # Create array containing all read traces
                read_traces = np.zeros(
                    (self.ESR['shots_per_frequency'], points_per_shot))
                for shot_idx in range(self.ESR['shots_per_frequency']):
                    # Read traces of different frequencies are interleaved
                    traces_idx = f_idx + shot_idx * len(self.ESR_frequencies)
                    traces_name = f"{self.ESR['read_pulse'].name}[{traces_idx}]"
                    read_traces[shot_idx] = traces[traces_name]['output'][sample]
                self.read_traces[f_idx, sample] = read_traces
                read_result = analysis.analyse_traces(
                    traces=read_traces,
                    sample_rate=self.sample_rate,
                    t_read=self.t_read,
                    t_skip=self.t_skip,
                    threshold_voltage=threshold_voltage)
                up_proportions[f_idx, sample] = read_result['up_proportion']
                results['results_read'].append(read_result)

            if len(self.ESR_frequencies) > 1:
                results[f'up_proportions_{f_idx}'] = up_proportions[f_idx]
            else:
                results['up_proportions'] = up_proportions[f_idx]

        # Add singleton dimension because analyse_flips handles 3D up_proportions
        up_proportions = np.expand_dims(up_proportions, 1)
        results_flips = analysis.analyse_flips(
            up_proportions_arrs=up_proportions,
            threshold_up_proportion=self.threshold_up_proportion)
        # Add results, only choosing first element so its no longer an array
        results.update({k: v[0] for k, v in results_flips.items()})
        return results


class FlipNucleusParameter(AcquisitionParameter):
    def __init__(self, name='flip_nucleus', **kwargs):
        self.pulse_sequence = NMRPulseSequence()
        self.NMR = self.pulse_sequence.NMR
        self.ESR = self.pulse_sequence.ESR
        self.pre_pulses = self.pulse_sequence.pulse_settings['pre_pulses']
        self.post_pulses = self.pulse_sequence.pulse_settings['post_pulses']

        super().__init__(name=name,
                         names=[],
                         snapshot_value=False,
                         wrap_set=False,
                         **kwargs)

    def get_NMR_pulses(self, initial_state, final_state, mode='full'):
        if mode not in ['neighbour', 'full']:
            raise SyntaxError('Mode must be either `neighbour` or `full`')

        elif initial_state == final_state:
            return []

        pulses_config = config[self.environment].pulses

        if mode == 'neighbour':
            if initial_state < final_state:
                states = range(initial_state, final_state)
            else:
                states = range(initial_state - 1, final_state - 1, -1)

            pulse_names = [f'NMR{state}{state+1}_pi' for state in states]

            if not all(pulse_name in pulses_config for pulse_name in pulse_names):
                raise RuntimeError('Not all pulses are defined in config')

        elif mode == 'full':
            state_sequences = {}
            new_state_sequences = {str(initial_state): 0}
            k = 0
            while new_state_sequences:
                if k > 10:
                    break
                k += 1
                state_sequences.update(**new_state_sequences)
                accessed_states = set(map(int, ''.join(state_sequences)))
                new_state_sequences = {}

                for state_sequence, duration in state_sequences.items():
                    current_state = int(state_sequence[-1])
                    pattern = f'^NMR({current_state}([0-9])|([0-9]){current_state})_pi$'

                    for pulse_name, pulse_settings in pulses_config.items():
                        match = re.match(pattern, pulse_name)
                        if not match:
                            continue

                        next_state = int(match.group(2) or match.group(3))
                        if next_state not in accessed_states:
                            new_duration = duration + pulse_settings['duration']
                            new_state_sequence = state_sequence + str(next_state)
                            new_state_sequences[new_state_sequence] = new_duration


            # Update accessed states to include new states
            state_sequences.update(**new_state_sequences)
            accessed_states = set(map(int, ''.join(state_sequences)))

            if final_state not in accessed_states:
                raise RuntimeError(f'Cannot find pulse sequence to {final_state}')
            else:
                valid_state_sequences = {k: v for k, v in state_sequences.items()
                                         if int(k[-1]) == final_state}
                if len(valid_state_sequences) > 1:
                    state_sequence = min(valid_state_sequences,
                                         key=valid_state_sequences.get)
                else:
                    state_sequence = next(iter(valid_state_sequences))

                pulse_names = []
                for state1, state2 in zip(state_sequence[:-1], state_sequence[1:]):
                    pulse_name = f'NMR{min(state1, state2)}{max(state1, state2)}_pi'
                    pulse_names.append(pulse_name)

        return [SinePulse(pulse_name) for pulse_name in pulse_names]


    def set(self, initial_state, final_state, run=True):
        if initial_state == final_state:
            # No need to perform any pulse sequence
            return

        self.ESR['pulses'] = []
        self.NMR['pulses'] = self.get_NMR_pulses(initial_state, final_state)
        self.pulse_sequence.generate()

        if run:
            self.setup(repeat=False)
            self.layout.start()
            self.layout.stop()


class T2ElectronParameter(AcquisitionParameter):
    def __init__(self, name='Electron_T2', **kwargs):
        self.pulse_sequence = T2ElectronPulseSequence()

        super().__init__(name=name,
                         names=['up_proportion', 'num_traces'],
                         labels=['Up proportion', 'Number of traces'],
                         snapshot_value=False,
                         properties_attrs=['t_skip', 'filter_traces'],
                         **kwargs)

        self.pre_pulses = self.pulse_sequence.pre_pulses
        self.post_pulses = self.pulse_sequence.post_pulses
        self.ESR = self.pulse_sequence.ESR
        self.EPR = self.pulse_sequence.EPR

    @property
    def inter_delay(self):
        return self.ESR['inter_delay']

    @inter_delay.setter
    def inter_delay(self, inter_delay):
        self.ESR['inter_delay'] = inter_delay

    def analyse(self, traces):
        if self.EPR['enabled']:
            # Analyse EPR sequence, which also gets the dark counts
            results = analysis.analyse_EPR(
                empty_traces=traces['empty']['output'],
                plunge_traces=traces['plunge']['output'],
                read_traces=traces['read_long']['output'],
                sample_rate=self.sample_rate,
                min_filter_proportion=self.min_filter_proportion,
                t_skip=self.t_skip, # Use t_skip to keep length consistent
                t_read=self.t_read)
        else:
            results = {}

        read_pulse = self.pulse_sequence.get_pulse(name=self.ESR["read_pulse"].name)
        read_traces = traces[read_pulse.full_name]['output']
        ESR_results = analysis.analyse_traces(
            traces=read_traces,
            sample_rate=self.sample_rate,
            filter='low' if self.filter_traces else None,
            t_skip=self.t_skip,
            t_read=self.t_read)

        results['ESR_results'] = ESR_results
        results[f'up_proportion_{read_pulse.name}'] = ESR_results['up_proportion']
        if self.EPR['enabled']:
            # Add contrast obtained by subtracting EPR dark counts
            contrast = ESR_results['up_proportion'] - results['dark_counts']
            results[f'contrast_{read_pulse.name}'] = contrast

        return results


class VariableReadParameter(AcquisitionParameter):
    def __init__(self, name='variable_read', **kwargs):
        self.pulse_sequence = PulseSequence([
            DCPulse(name='plunge', acquire=True, average='trace'),
            DCPulse(name='read', acquire=True, average='trace'),
            DCPulse(name='empty', acquire=True, average='trace'),
            DCPulse(name='final')])

        super().__init__(name=name,
                         names=('read_voltage',),
                         labels=('Read voltage',),
                         units=('V',),
                         shapes=((1,),),
                         setpoint_names=(('time',),),
                         setpoint_labels=(('Time',),),
                         setpoint_units=(('s',),),
                         snapshot_value=False,
                         **kwargs)

    @property_ignore_setter
    def setpoints(self):
        duration = sum(pulse.duration for pulse in
                       self.pulse_sequence.get_pulses(acquire=True))
        return (tuple(np.linspace(0, duration, self.shapes[0][0])), ),

    @property_ignore_setter
    def shapes(self):
        shapes = self.layout.acquisition_shapes
        pts = sum(shapes[pulse_name]['output'][0]
                  for pulse_name in ['plunge', 'read', 'empty'])
        return (pts,),

    def analyse(self, traces):
        return {'read_voltage':
                    np.concatenate([self.traces['plunge']['output'],
                                    self.traces['read']['output'],
                                    self.traces['empty']['output']])}


class BlipsParameter(AcquisitionParameter):
    """
    Parameter that measures properties of blips in a trace
    """
    def __init__(self, name='count_blips', duration=None,
                 pulse_name='DC_trace', **kwargs):
        """

        Args:
            name: parameter name (default `count_blips`)
            duration: duration of tracepulse
            pulse_name: name of trace pulse (default `read`)
            **kwargs: kwargs passed to AcquisitionParameter
        """
        self.pulse_name = pulse_name

        self.pulse_sequence = PulseSequence([
            DCPulse(name=pulse_name, acquire=True, average='none')])

        super().__init__(name=name,
                         names=['blips',
                                'blips_per_second',
                                'mean_low_blip_duration',
                                'mean_high_blip_duration'],
                         units=['', '1/s', 's', 's'],
                         shapes=((), (), (),()),
                         snapshot_value=False,
                         continuous = True,
                         **kwargs)
        self.samples = 1
        self.duration = duration
        self.threshold_voltage = 0.3

    @property
    def duration(self):
        return self.pulse_sequence[self.pulse_name].duration

    @duration.setter
    def duration(self, duration):
        self.pulse_sequence[self.pulse_name].duration = duration

    def analyse(self, traces):
        return analysis.count_blips(
            traces=self.traces[self.pulse_name]['output'],
            t_skip=0,
            sample_rate=self.sample_rate,
            threshold_voltage=self.threshold_voltage)


class NeuralNetworkParameter(AcquisitionParameter):
    # TODO: Make a measurement Parameter
    def __init__(self, target_parameter, input_names, output_names=None,
                 model_filepath=None, include_target_output=None, **kwargs):
        # Load model here because it takes quite a while to load
        from keras.models import load_model

        self.target_parameter = target_parameter
        self.input_names = input_names
        self.include_target_output = include_target_output
        self.output_names = output_names

        if model_filepath is None:
            model_filepath = self.properties_config.get(
                f'{self.name}_model_filepath', None)
        self.model_filepath = model_filepath
        if self.model_filepath is not None:
            self.model = load_model(self.model_filepath)
        else:
            logger.warning(f'No neural network model loaded for {self}')

        super().__init__(names=self.names, **kwargs)

    @property_ignore_setter
    def pulse_sequence(self):
        return self.target_parameter.pulse_sequence

    @property_ignore_setter
    def names(self):
        names = self.output_names
        if self.include_target_output is True:
            names = names + self.target_parameter.names
        elif self.include_target_output is False:
            pass
        elif isinstance(self.include_target_output, Iterable):
            names = names + self.include_target_output
        return names

    def acquire(self):
        if self.samples is not None:
            self.target_parameter.samples = self.samples
        self.target_parameter()
        # Extract target results using input names, because target_parameter.get
        # may provide results in a different order
        target_results = [self.target_parameter.results[name]
                          for name in self.input_names]

        # Convert target results to array
        target_results_arr = np.array([target_results])
        neural_network_results = self.model.predict(target_results_arr)[0]

        # Convert neural network results to dict
        self.neural_network_results = dict(zip(self.output_names,
                                               neural_network_results))
        return self.neural_network_results

    def analyse(self, traces):
        results = dict(**self.neural_network_results)

        if self.include_target_output is True:
            results.update(**self.target_parameter.results)
        elif isinstance(self.include_target_output, Iterable):
            for name in self.include_target_output:
                results[name] = self.target_parameter.results[name]
        return results


class NeuralRetuneParameter(NeuralNetworkParameter):
    # TODO: Make a measurement Parameter
    def __init__(self, target_parameter, output_parameters, update=False,
                 **kwargs):
        self.output_parameters = output_parameters
        output_names = [f'{output_parameter.name}_delta' for
                        output_parameter in output_parameters]

        input_names = ['contrast', 'dark_counts', 'high_blip_duration',
                       'fidelity_empty', 'voltage_difference_empty',
                       'low_blip_duration', 'fidelity_load',
                       'voltage_difference_load', 'voltage_difference_read']

        self.update = update

        super().__init__(target_parameter=target_parameter,
                         input_names=input_names,
                         output_names=output_names, **kwargs)

    @property_ignore_setter
    def names(self):
        names = [f'{output_parameter.name}_optimal' for
                   output_parameter in self.output_parameters]
        if self.include_target_output is True:
            names = names + self.target_parameter.names
        elif self.include_target_output is False:
            pass
        elif isinstance(self.include_target_output, Iterable):
            names = names + self.include_target_output
        return names

    @property
    def base_folder(self):
        return self.target_parameter.base_folder

    @base_folder.setter
    def base_folder(self, base_folder):
        self.target_parameter.base_folder = base_folder

    def analyse(self, traces):
        results = {}
        for output_parameter in self.output_parameters:
            # Get neural network otput (change in output parameter value)
            result_name = f'{output_parameter.name}_delta'
            delta_value = self.neural_network_results[result_name]

            optimal_value = output_parameter() + delta_value

            if self.update:
                # Update parameter to optimal value
                output_parameter(optimal_value)

            results[f'{output_parameter.name}_optimal'] = optimal_value

        if self.include_target_output is True:
            results.update(**self.target_parameter.results)
        elif isinstance(self.include_target_output, Iterable):
            for name in self.include_target_output:
                results[name] = self.target_parameter.results[name]
        return results