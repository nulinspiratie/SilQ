from typing import List, Dict, Any, Union, Sequence
import numpy as np
from collections import OrderedDict, Iterable
from copy import copy
from blinker import signal
from functools import partial
import logging
import re

from qcodes import DataSet, MultiParameter, active_dataset, active_measurement, \
    Parameter
from qcodes.data import hdf5_format
from qcodes import Instrument, MatPlot

from silq import config
from silq.pulses import *
from silq.pulses.pulse_sequences import ESRPulseSequence, NMRPulseSequence, \
    T2ElectronPulseSequence, FlipFlopPulseSequence, ESRRamseyDetuningPulseSequence
from silq.analysis import analysis
from silq.tools.general_tools import SettingsClass, clear_single_settings, \
    attribute_from_config, UpdateDotDict, convert_setpoints, \
    property_ignore_setter

__all__ = ['AcquisitionParameter', 'DCParameter', 'TraceParameter',
           'DCSweepParameter', 'EPRParameter', 'ESRParameter',
           'NMRParameter', 'EDSRParameter', 'VariableReadParameter',
           'BlipsParameter', 'FlipNucleusParameter', 'FlipFlopParameter',
           'NeuralNetworkParameter', 'NeuralRetuneParameter','ESRRamseyDetuningParameter']

logger = logging.getLogger(__name__)
h5fmt = hdf5_format.HDF5Format()


class AcquisitionParameter(SettingsClass, MultiParameter):
    """Parameter used for acquisitions involving a `PulseSequence`.

    Each AcquisitionParameter has an associated pulse sequence, which it directs
    to the Layout, after which it acquires traces and performs post-processing.

    Generally, the flow of an AcquisitionParameter is as follows:

    1. `AcquisitionParameter.acquire`, which acquires traces.
       This stage can be subdivided into several steps:

       1. Generate PulseSequence if pulse sequence properties have changed
          Note that this is only necessary for a `PulseSequenceGenerator`, which
          is a more complicated pulse sequences that can be generated from
          properties.
       2. Target pulse sequence in `Layout`.
          This only happens if `Layout.pulse_sequence` differs from the pulse
          sequence used.
       3. Call `Layout.setup` which sets up instruments with new pulse sequence.
          Again, this is only done if `Layout.pulse_sequence` differs.
       4. Acquire traces using `Layout.acquisition`
          This in turn gets the traces fromm the acquisition instrument.
          The returned traces are segmented by pulse and acquisition channel.

    2. `AcquisitionParameter.analyse`, which analyses the traces
       This method differs per AcquisitionParameter.
    3. Perform ancillary actions such as saving traces, printing results
    4. Return list of results for any measurement.
       The subset of results that are in ``AcquisitionParameter.names`` is
       returned.

    Args:
        continuous: If True, instruments keep running after acquisition
        properties_attrs: attributes to match with
            ``silq.config.properties`` (see notes below).
        save_traces: Save acquired traces to disk
        **kwargs: Additional kwargs passed to ``MultiParameter``

    Parameters:
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Notes:
        - AcquisitionParameters are subclasses of the ``MultiParameter``, and
          therefore always returns multiple values
        - AcquisitionParameters are also subclasses of `SettingsClass`, which
          gives it the ability to temporarily override its attributes using
          methods `single_settings` and `temporary_settings`. These overridden
          settings can be clear later on. This is useful if you temporarily want
          to change settings. See `SettingsClass` for more info.
        - When certain elements in ``silq.config`` are updated, this will also
          update the corresponding attributes in the AcquisitionParameter.
          Two config paths are monitored:

          - ``silq.config.properties``, though only the attributes
            specified in ``properties_attrs``.
          - ``silq.parameters.{self.name}``.

    """

    layout = None
    formatter = h5fmt

    def __init__(self,
                 continuous: bool = False,
                 properties_attrs: List[str] = None,
                 wrap_set: bool = False,
                 save_traces: bool = False,
                 **kwargs):
        SettingsClass.__init__(self)

        if not hasattr(self, 'pulse_sequence'):
            self.pulse_sequence = PulseSequence()

        shapes = kwargs.pop('shapes', ((), ) * len(kwargs['names']))
        MultiParameter.__init__(self, shapes=shapes, wrap_set=wrap_set, **kwargs)

        if self.layout is None:
            try:
                AcquisitionParameter.layout = Instrument.find_instrument('layout')
            except KeyError:
                logger.warning(f'No layout found for {self}')

        self.silent = True

        self.save_traces = save_traces

        self.continuous = continuous

        self.samples = None
        self.traces = None
        self.dataset = None
        self.results = None
        self.base_folder = None
        self.subfolder = None

        # Attach to properties and parameter configs
        self.properties_attrs = properties_attrs
        self.properties_config = self._attach_to_config(path=f'properties',
            select_attrs=self.properties_attrs)
        self.parameter_config = self._attach_to_config(
            path=f'parameters.{self.name}')

        self._meta_attrs.extend(['label', 'name', 'pulse_sequence'])

    def __repr__(self):
        return f'{self.name} acquisition parameter'

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return attribute_from_config(
                item, config=config.properties)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

        # Update value in pulse settings if it exists
        try:
            if key in self.pulse_sequence.pulse_settings:
                self.pulse_sequence.pulse_settings[key] = value
        except AttributeError:
            pass

    @property_ignore_setter
    def labels(self):
        return [name[0].capitalize() + name[1:].replace('_', ' ')
                for name in self.names]

    @property
    def sample_rate(self):
        """ Acquisition sample rate """
        return self.layout.sample_rate

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=None,
                      simplify: bool = False):
        snapshot = super().snapshot_base(update=update,
                                         params_to_skip_update=params_to_skip_update,
                                         simplify=simplify)
        snapshot['pulse_sequence'] = snapshot['pulse_sequence'].snapshot()
        return snapshot

    def _attach_to_config(self,
                          path: str,
                          select_attrs: List[str] = None):
        """Attach parameter to a subconfig (within silq config).

        This means that whenever an item in the subconfig is updated,
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
        signal(f'environment:{path}').connect(signal_handler, weak=False)

        # Set attributes that are present in subconfig
        for attr, val in subconfig.items():
            if select_attrs is None or attr in select_attrs:
                setattr(self, attr, val)

        return subconfig

    def _handle_config_signal(self, _,
                              select: List[str] = None,
                              **kwargs):
        """Update attr when attr in pulse config is modified

        Args:
            _: sender config (unused)
            select: list of attrs that can be set.
                Will update any attribute if not specified.
            **kwargs: {attr: new_val}
        """
        key, val = kwargs.popitem()
        if select is None or key in select:
            setattr(self, key, val)

    def setup(self,
              start: bool = None,
              **kwargs):
        """Setup instruments with current pulse sequence.

        Args:
            start: Start instruments after setup. If not specified, will use
                value in ``AcquisitionParameter.continuous``.
            **kwargs: Additional kwargs passed to `Layout.setup`.
        """
        if not self.pulse_sequence.up_to_date():
            self.pulse_sequence.generate()

        self.layout.pulse_sequence = self.pulse_sequence

        samples = kwargs.pop('samples', self.samples)
        self.layout.setup(samples=samples, **kwargs)

        if start is None:
            start = self.continuous

        if start:
            self.layout.start()

    def acquire(self,
                stop: bool = None,
                setup: bool = None,
                save_traces: bool = None,
                **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
        """Performs a `Layout.acquisition`.

        Args:
            stop: Stop instruments after acquisition. If not specified, it will
                stop if ``AcquisitionParameter.continuous`` is False.
            setup: Whether to setup layout before acquisition.
                If not specified, it will setup if pulse_sequences are different
            save_traces: whether to save traces during
            **kwargs: Additional kwargs to be given to `Layout.acquisition`.

        Returns:
            acquisition traces dictionary, segmented by pulse.
            dictionary has the following format:
            {pulse.full_name: {acquisition_channel_label: traces}}
            where acquisition_channel_label is specified in `Layout`.
        """
        if stop is None:
            stop = not self.continuous

        if not self.pulse_sequence.up_to_date():
            self.pulse_sequence.generate()

        if setup or (setup is None and
                     self.layout.pulse_sequence != self.pulse_sequence) or \
                self.layout.samples() != self.samples:
            self.setup()

        if save_traces is None:
            save_traces = self.save_traces
        if save_traces and active_measurement() is None:
            logger.warning('Cannot save traces since there is no active loop')
            save_traces = False

        # Perform acquisition
        self.traces = self.layout.acquisition(stop=stop,
                                              save_traces=save_traces,**kwargs)
        return self.traces

    def analyse(self,
                traces: Dict[str, Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Analyse traces, should be implemented in subclass"""
        raise NotImplementedError('`analyse` must be implemented in subclass')

    def plot_traces(self, filter=None, channels=['output'],
                    t_skip: Union[bool, float] = True,
                    **kwargs):

        plot_traces = OrderedDict()
        for pulse_name, trace in self.traces.items():
            if filter is not None:
                if isinstance(filter, str):
                    filter = [filter]
                if not any(elem in pulse_name for elem in filter):
                    continue

            plot_traces[pulse_name] = trace

        if t_skip is False:
            start_idx = 0
        elif t_skip is True:
            start_idx = int(self.sample_rate * self.t_skip)
        else:
            start_idx = int(self.sample_rate * t_skip)

        if len(channels) > 1:
            subplots = (len(plot_traces), len(channels))
        else:
            subplots = len(plot_traces)
        plot = MatPlot(subplots=subplots, **kwargs)

        k = 0
        for pulse_name, traces in plot_traces.items():
            for channel in channels:
                trace_arr = traces[channel]
                pts = trace_arr.shape[-1]
                t_list = np.linspace(0, pts / self.sample_rate, pts,
                                     endpoint=False)
                if trace_arr.ndim == 2:
                    plot[k].add(traces[channel][:,start_idx:], x=t_list[start_idx:],
                                y=np.arange(trace_arr.shape[0], dtype=float))
                else:
                    plot[k].add(traces[channel][start_idx:], x=t_list[start_idx:])
                plot[k].set_xlabel('Time (s)')
                plot[k].set_title(pulse_name)
                k += 1
        plot.tight_layout()
        return plot

    def print_results(self):
        """Print results whose keys are in ``AcquisitionParameter.names``"""
        names = self.names if self.names is not None else [self.name]
        for name in names:
            value = self.results[name]
            if isinstance(value, (int, float)):
                print(f'{name}: {value:.3f}')
            else:
                print(f'{name}: {value}')

    @clear_single_settings
    def get_raw(self):
        self.traces = self.acquire()

        self.results = self.analyse(self.traces)

        if not self.silent:
            self.print_results()

        return tuple(self.results[name] for name in self.names)

    def set(self, **kwargs):
        """Perform an acquisition with temporarily modified settings

        Shorthand for:
        ```
        AcquisitionParameter.single_settings(**kwargs)
        AcquisitionParameter()
        ```
        """
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

    def analyse(self, traces = None):
        if traces is None:
            traces = self.traces

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
    """Acquires DC voltage

    The pulse sequence contains a single read pulse, the duration of which
    specifies how long should be averaged.

    Args:
        name: Parameter name.
        unit: Unit of DC voltage (e.g. can be changed to nA)

    Parameters:
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Returns:
        DC_voltage: Average DC voltage measured on the output channel
        DC_noise: noise standard deviation measured on the output channel

    Notes:
        - ``DCParameter.continuous`` is True by default

    Todo:
        implement continuous acquisition in the ATS interface
    """
    def __init__(self,
                 name: str = 'DC',
                 unit: str = 'V',
                 **kwargs):
        self.pulse_sequence = PulseSequence([
            DCPulse(name='DC', acquire=True),
            DCPulse(name='DC_final')])

        super().__init__(name=name,
                         names=['DC_voltage', 'DC_noise'],
                         units=[unit, unit],
                         snapshot_value=False,
                         continuous=True,
                         **kwargs)
        self.samples = 1

    def analyse(self, traces = None):
        if traces is None:
            traces = self.traces
        return {'DC_voltage': np.mean(traces['DC']['output']),
                'DC_noise': np.std(traces['DC']['output'])}


class TraceParameter(AcquisitionParameter):
    """An acquisition parameter for obtaining a trace or multiple traces
    of a given PulseSequence.

    Example:
        >>> parameter.average_mode = 'none'
        >>> parameter.pulse_sequence = my_pulse_sequence

        Note that for the above example, all pulses in my_pulse_sequence will be
        copied.

    Parameters:
        trace_pulse (Pulse): Acquisition measurement pulse.
            Duration is dynamically set to the duration of acquired pulses.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.


    """
    def __init__(self, name='trace_pulse', average_mode='none', **kwargs):
        self._average_mode = average_mode
        self._pulse_sequence = PulseSequence()
        self.trace_pulse = MeasurementPulse(name=name, duration=1e-3,
                                            average=self.average_mode)
        self._pulse_sequence.add(self.trace_pulse)

        super().__init__(name='Trace_acquisition',
                         names=self.names,
                         units=self.units,
                         shapes=self.shapes,
                         snapshot_value=False,
                         **kwargs)
        self.samples = 1

    @property
    def average_mode(self):
        """Acquisition averaging mode.

        The attribute `Pulse.average` is overridden.
        """
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
        """ Modifies provided pulse sequence by creating a single
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
        for pulse in self.pulse_sequence:
            if pulse is self.trace_pulse:
                continue
            pulse.acquire = False

        # Remove any existing trace pulse
        if self.trace_pulse.full_name in self.pulse_sequence:
            self.pulse_sequence.remove(self.trace_pulse.full_name)
        self.pulse_sequence.add(self.trace_pulse)

        super().setup(start=start, **kwargs)

    def acquire(self, **kwargs):
        """Acquires the number of traces defined in self.samples

        Args:
            **kwargs: kwargs passed to `AcquisitionParameter.acquire`

        Returns:
            A tuple of data points. e.g.
            ((data_for_1st_output), (data_for_2nd_output), ...)
        """
        super().acquire(**kwargs)

        traces = {self.trace_pulse.full_name + '_' + output:
                      self.traces[self.trace_pulse.full_name][output]
                  for _, output in self.layout.acquisition_channels()}

        return traces

    def analyse(self, traces = None):
        """Rearrange traces to match ``AcquisitionParameter.names``"""
        if traces is None:
            traces = self.traces

        return {self.names[k] : traces[name] if isinstance(traces[name], float)
                else traces[name].tolist()[0]
                for k, name in enumerate(traces)}


class DCSweepParameter(AcquisitionParameter):
    """Perform 1D and 2D DC sweeps by rapidly varying AWG output voltages

    Using this parameter, a 2D DC sweep of 100x100 points can be obtained in
    ~1 second. This does of course depend on the filtering of the lines, and the
    acquisition sampling rate. This is used in the `DCSweepPlot` to continuously
    update and display the charge stability diagram.

    The pulse sequence is created by first calling `DCSweepParameter.add_sweep`,
    which adds a dimension every time it's called.
    After adding the sweeps, `DCSweepParameter.generate` will create the
    corresponding `PulseSequence`.

    Args:
        name: parameter name
        **kwargs: Additional kwargs passed to AcquisitionParameter

    Parameters:
        trace_pulse (Pulse): Trace pulse at fixed voltage at the end of sweep.
            Can be turned off by ``trace_pulse.enabled = False``.
        pulse_duration (float): Duration of each point in DC sweep
        final_delay (float): Delay at end of pulse sequence.
        inter_delay (float): Delay after each row of DC points
        use_ramp (bool): Combine single row of DC points into a ramp pulse that
            will be segmented later. This saves number of waveforms sent,
            reduces triggers, and creates less `Pulse` objects.
        sweep_parameters (UpdateDotDict): Sweep parameters. Every time an item
            is updated, `DCSweepParameter.generate` is called.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Note:
        Currently only works up to 2D.

    Todo:
        Convert pulse sequence and generator into `PulseSequenceGenerator`
    """
    connect_to_config = True  # Whether pulses should connect to config (speedup)
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
                         units=['V'],
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
                sweep_voltages = sweep_voltages + sweep_dict.offset_parameter()
            setpoints = (convert_setpoints(sweep_voltages),),

        elif len(self.sweep_parameters) == 2:
            inner_sweep_dict = next(iter_sweep_parameters)
            inner_sweep_voltages = inner_sweep_dict.sweep_voltages
            if inner_sweep_dict.offset_parameter is not None:
                inner_sweep_voltages = inner_sweep_voltages + inner_sweep_dict.offset_parameter()
            outer_sweep_dict = next(iter_sweep_parameters)
            outer_sweep_voltages = outer_sweep_dict.sweep_voltages
            if outer_sweep_dict.offset_parameter is not None:
                outer_sweep_voltages = outer_sweep_voltages + outer_sweep_dict.offset_parameter()

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

    def add_sweep(self,
                  parameter_name: str,
                  sweep_voltages: np.ndarray = None,
                  connection_label: str = None,
                  offset_parameter: Parameter = None):
        """Add sweep to ``DCSweepParameter.sweep_parameters``

        Each call will add a sweep as the outer dimension.

        Args:
            parameter_name: Name of parameter (for axis labelling).
            sweep_voltages: List of sweep voltages. If
                ``DCSweepParameter.use_ramp`` is True, these must be
                equidistant.
            connection_label: Connection label to target pulses to.
                For multiple sweeps, each connection label must be distinct.
                Connection labels are defined in ``Layout.acquisition_outputs``.
            offset_parameter: Parameter used for offsetting the sweep voltages.
                Usually this is the corresponding DC voltage parameter.
        """

        if connection_label is None:
            connection_label = parameter_name

        self.sweep_parameters[parameter_name] = UpdateDotDict(
            update_function=self.generate, name=parameter_name,
            sweep_voltages=sweep_voltages, connection_label=connection_label,
            offset_parameter=offset_parameter)

        self.generate()

    def generate(self):
        """Generates pulse sequence using sweeps in `DCSweepParameter.add_sweep`

        Note:
            Currently only works for 1D and 2D
        """
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
                                      connection_label=connection_label,
                                      connect_to_config=self.connect_to_config)]
            else:
                pulses = [
                    DCPulse('DC_inner', duration=self.pulse_duration,
                            acquire=True, average='point',
                            amplitude=sweep_voltage,
                            connection_label=connection_label,
                            connect_to_config=self.connect_to_config)
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
                                    connection_label=outer_connection_label,
                                    connect_to_config=self.connect_to_config))
            else:
                t = 0
                sweep_duration = self.pulse_duration * len(inner_sweep_voltages)
                for outer_sweep_voltage in outer_sweep_voltages:
                    pulses.append(
                        DCPulse('DC_outer', t_start=t,
                                duration=sweep_duration + self.inter_delay,
                                amplitude=outer_sweep_voltage,
                                connection_label=outer_connection_label,
                                connect_to_config=self.connect_to_config))
                    if self.inter_delay > 0:
                        pulses.append(
                            DCPulse('DC_inter_delay', t_start=t,
                                    duration=self.inter_delay,
                                    amplitude=inner_sweep_voltages[0],
                                    connection_label=inner_connection_label,
                                    connect_to_config=self.connect_to_config))
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
                                        connection_label=inner_connection_label,
                                        connect_to_config=self.connect_to_config)
                        )
                        t += sweep_duration
                    else:
                        for inner_sweep_voltage in inner_sweep_voltages:
                            pulses.append(
                                DCPulse('DC_inner', t_start=t,
                                        duration=self.pulse_duration,
                                        acquire=True, average='point',
                                        amplitude=inner_sweep_voltage,
                                        connection_label=inner_connection_label,
                                        connect_to_config=self.connect_to_config)
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

    def analyse(self,
                traces: Dict[str, Dict[str, np.ndarray]] = None):
        """Analyse traces, ensuring resulting dimensionality is correct

        Args:
            traces: Traces returned by `AcquisitionParameter.acquire`.

        Returns:
            (Dict[str, Any]): Dict containing:

            :DC_voltage (np.ndarray): DC voltages with dimensionality
              corresponding to number of sweeps.
            :trace_voltage (np.ndarray): voltage trace of final trace pulse.
              Only used if ``DCSweepParameter.trace_pulse.enabled``.
              Trace in ``output`` connection label is returned.
        """
        if traces is None:
            traces = self.traces

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


class VariableReadParameter(AcquisitionParameter):
    """Parameter for measuring spin tails.

    The pulse sequence is ``plunge`` - ``read`` - ``empty``.
    By varying the read amplitude, the voltage should transition between
    high voltage (``empty``) to low voltage (``plunge``), and somewhere in
    between an increased voltage should be visible at the start, indicating
    spin-dependent tunneling.

    Args:
        name: Parameter name
        **kwargs: Additional kwargs passed to AcquisitionParameter

    Parameters:
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples to average over.
        results (dict): Results obtained after analysis of traces.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
    """
    def __init__(self, name='variable_read', **kwargs):
        self.pulse_sequence = PulseSequence([
            DCPulse(name='plunge', acquire=True, average='trace'),
            DCPulse(name='read', acquire=True, average='trace'),
            DCPulse(name='empty', acquire=True, average='trace')])

        super().__init__(name=name,
                         names=('read_voltage',),
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
        pts = sum([shapes[pulse.full_name]['output'][0]
                  for pulse in self.pulse_sequence.get_pulses(acquire=True)])
        return (pts,),

    def analyse(self, traces = None):
        if traces is None:
            traces = self.traces

        return {'read_voltage':
                    np.concatenate([traces[pulse.full_name]['output']
                                    for pulse in self.pulse_sequence.get_pulses(acquire=True)])}


class EPRParameter(AcquisitionParameter):
    """Parameter for an empty-plunge-read sequence.

    Args:
        name: Name of acquisition parameter
        **kwargs: Additional kwargs passed to `AcquisitionParameter`.

    Parameters:
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        t_skip (float): initial part of read trace to ignore for measuring
            blips. Useful if there is a voltage spike at the start, which could
            otherwise be measured as a ``blip``. Retrieved from
            ``silq.config.properties.t_skip``.
        t_read (float): duration of read trace to include for measuring blips.
            Useful if latter half of read pulse is used for initialization.
            Retrieved from ``silq.config.properties.t_read``.
        min_filter_proportion (float): Minimum number of read traces needed in
            which the voltage starts low (loaded donor). Otherwise, most results
            are set to zero. Retrieved from
            ``silq.config.properties.min_filter_proportion``.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Note:
        A ``read_long`` pulse is used instead of ``read`` because this allows
        comparison of the start and end of the pulse, giving the ``contrast``.
    """
    def __init__(self, name='EPR', **kwargs):
        self.pulse_sequence = PulseSequence([
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read_long', acquire=True)])

        super().__init__(name=name,
                         names=['contrast', 'up_proportion',
                                'dark_counts',
                                'voltage_difference_read',
                                'fidelity_empty', 'fidelity_load', 'voltage_average_read'],
                         snapshot_value=False,
                         properties_attrs=['t_skip', 't_read',
                                           'min_filter_proportion',
                                           'filter_traces'],
                         **kwargs)

    def analyse(self,
                traces: Dict[str, Dict[str, np.ndarray]] = None,
                plot: bool = False) -> Dict[str, Any]:
        """Analyse traces using `analyse_EPR`"""
        if traces is None:
            traces = self.traces

        threshold_voltage = getattr(self, 'threshold_voltage', None)

        return analysis.analyse_EPR(
            empty_traces=traces['empty']['output'],
            plunge_traces=traces['plunge']['output'],
            read_traces=traces['read_long']['output'],
            sample_rate=self.sample_rate,
            t_skip=self.t_skip,
            t_read=self.t_read,
            min_filter_proportion=self.min_filter_proportion,
            threshold_voltage=threshold_voltage,
            filter_traces=self.filter_traces,
            plot=plot)


class ESRParameter(AcquisitionParameter):
    """Parameter for most pulse sequences involving electron spin resonance.

    This parameter can handle many of the simple pulse sequences involving ESR.
    It uses the `ESRPulseSequence`, which will generate a pulse sequence from
    settings (see parameters below).

    In general the pulse sequence is as follows:

    1. Perform any pre_pulses defined in ``ESRParameter.pre_pulses``.
    2. Perform stage pulse ``ESRParameter.ESR['stage_pulse']``.
       By default, this is the ``plunge`` pulse.
    3. Perform ESR pulse within plunge pulse, the delay from start of plunge
       pulse is defined in ``ESRParameter.ESR['pulse_delay']``.
    4. Perform read pulse ``ESRParameter.ESR['read_pulse']``.
    5. Repeat steps 2 and 3 for each ESR pulse in
       ``ESRParameter.ESR['ESR_pulses']``, which by default contains single
       pulse ``ESRParameter.ESR['ESR_pulse']``.
    6. Perform empty-plunge-read sequence (EPR), but only if
       ``ESRParameter.EPR['enabled']`` is True.
       EPR pulses are defined in ``ESRParameter.EPR['pulses']``.
    7. Perform any post_pulses defined in ``ESRParameter.post_pulses``.

    A shorthand for using the default ESR pulse for multiple frequencies is by
    setting `ESRParameter.ESR_frequencies`. Settings this will create a copy
    of ESRParameter.ESR['ESR_pulse'] with the respective frequency.

    Examples:
        The following code measures two ESR frequencies and performs an EPR
        from which the contrast can be determined for each ESR frequency:

        >>> ESR_parameter = ESRParameter()
        >>> ESR_parameter.ESR['pulse_delay'] = 5e-3
        >>> ESR_parameter.ESR['stage_pulse'] = DCPulse['plunge']
        >>> ESR_parameter.ESR['ESR_pulse'] = FrequencyRampPulse('ESR_adiabatic')
        >>> ESR_parameter.ESR_frequencies = [39e9, 39.1e9]
        >>> ESR_parameter.EPR['enabled'] = True
        >>> ESR_parameter.pulse_sequence.generate()

        The total pulse sequence is plunge-read-plunge-read-empty-plunge-read
        with an ESR pulse in the first two plunge pulses, 5 ms after the start
        of the plunge pulse. The ESR pulses have different frequencies.

    Args:
        name: Name of acquisition parameter
        **kwargs: Additional kwargs passed to `AcquisitionParameter`.

    Parameters:
        ESR (dict): `ESRPulseSequence` generator settings for ESR. Settings are:
            ``stage_pulse``, ``ESR_pulse``, ``ESR_pulses``, ``pulse_delay``,
            ``read_pulse``.
        EPR (dict): `ESRPulseSequence` generator settings for EPR.
            This is optional and can be toggled in ``EPR['enabled']``.
            If disabled, contrast is not calculated.
            Settings are: ``enabled``, ``pulses``.
        pre_pulses (List[Pulse]): Pulses to place at the start of the sequence.
        post_pulses (List[Pulse]): Pulses to place at the end of the sequence.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        t_skip (float): initial part of read trace to ignore for measuring
            blips. Useful if there is a voltage spike at the start, which could
            otherwise be measured as a ``blip``. Retrieved from
            ``silq.config.properties.t_skip``.
        t_read (float): duration of read trace to include for measuring blips.
            Useful if latter half of read pulse is used for initialization.
            Retrieved from ``silq.config.properties.t_read``.
        min_filter_proportion (float): Minimum number of read traces needed in
            which the voltage starts low (loaded donor). Otherwise, most results
            are set to zero. Retrieved from
            ``silq.config.properties.min_filter_proportion``.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties``.
            See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Notes:
        - All pulse settings are copies of
          ``ESRParameter.pulse_sequence.pulse_settings``.
        - For given pulse settings, ``ESRParameter.pulse_sequence.generate``
          will recreate the pulse sequence from settings.
    """
    def __init__(self, name='ESR', **kwargs):
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
            if 'voltage_difference' in self._names:
                names.append('voltage_difference')

        ESR_pulse_names = [pulse.name for pulse in self.pulse_sequence.primary_ESR_pulses]

        for pulse in self.pulse_sequence.primary_ESR_pulses:
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
            names.append(f'num_traces_{name}')
        return names

    @names.setter
    def names(self, names):
        """Set all the names to return upon .get() for the EPR sequence"""
        self._names = [name for name in names
                       if not 'contrast_' in name
                       and not 'up_proportion_' in name]

    @property_ignore_setter
    def shapes(self):
        return ((), ) * len(self.names)

    @property_ignore_setter
    def units(self):
        return ('', ) * len(self.names)

    @property
    def ESR_frequencies(self):
        """Apply default ESR pulse for each ESR frequency given."""
        return self.pulse_sequence.ESR_frequencies

    @ESR_frequencies.setter
    def ESR_frequencies(self, ESR_frequencies: List[float]):
        self.pulse_sequence.generate(ESR_frequencies=ESR_frequencies)

    def analyse(self, traces = None, plot=False):
        """Analyse ESR traces.

        If there is only one ESR pulse, returns ``up_proportion_{pulse.name}``.
        If there are several ESR pulses, adds a zero-based suffix at the end for
        each ESR pulse. If ``ESRParameter.EPR['enabled'] == True``, the results
        from `analyse_EPR` are also added, as well as ``contrast_{pulse.name}``
        (plus a suffix if there are several ESR pulses).
        """
        if traces is None:
            traces = self.traces

        threshold_voltage = getattr(self, 'threshold_voltage', None)

        if self.EPR['enabled']:
            # Analyse EPR sequence, which also gets the dark counts
            results = analysis.analyse_EPR(
                empty_traces=traces[self.pulse_sequence._EPR_pulses[0].full_name]['output'],
                plunge_traces=traces[self.pulse_sequence._EPR_pulses[1].full_name]['output'],
                read_traces=traces[self.pulse_sequence._EPR_pulses[2].full_name]['output'],
                sample_rate=self.sample_rate,
                min_filter_proportion=self.min_filter_proportion,
                threshold_voltage=threshold_voltage,
                filter_traces=self.filter_traces,
                t_skip=self.t_skip, # Use t_skip to keep length consistent
                t_read=self.t_read)
        else:
            results = {}

        ESR_pulses = self.pulse_sequence.primary_ESR_pulses
        ESR_pulse_names = [pulse.name for pulse in ESR_pulses]
        read_pulses = self.pulse_sequence.get_pulses(name=self.ESR["read_pulse"].name)
        results['ESR_results'] = []

        for read_pulse, ESR_pulse in zip(read_pulses, ESR_pulses):
            read_traces = traces[read_pulse.full_name]['output']
            ESR_results = analysis.analyse_traces(
                traces=read_traces,
                sample_rate=self.sample_rate,
                filter='low' if self.filter_traces else None,
                min_filter_proportion=self.min_filter_proportion,
                threshold_voltage=threshold_voltage,
                t_skip=self.t_skip,
                t_read=self.t_read,
                plot=plot)
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
            results[f'num_traces_{pulse_label}'] = ESR_results['num_traces']

        voltage_differences = [ESR_result['voltage_difference']
                               for ESR_result in results['ESR_results']
                               if ESR_result['voltage_difference'] is not None]
        if voltage_differences:
            results['voltage_difference'] = np.mean(voltage_differences)
        else:
            results['voltage_difference'] = np.nan

        self.results = results
        return results


class T2ElectronParameter(AcquisitionParameter):
    """Parameter for measuring electron decoherence.

    This parameter can apply any number of refocusing pulses.
    It uses the `T2ElectronPulseSequence`, which will generate a pulse sequence
    from settings (see parameters below).

    In general, the pulse sequence is as follows:

    1. Perform any pre_pulses defined in `T2ElectronParameter.pre_pulses`.
    2. Perform stage pulse ``T2ElectronParameter.ESR['stage_pulse']``.
       By default, this is the ``plunge`` pulse.
    3. Perform ESR pulse within plunge pulse, the delay from start of plunge
       pulse is defined in ``T2ElectronParameter.ESR['pulse_delay']``.
    4. Perform read pulse ``T2ElectronParameter.ESR['read_pulse']``.
    5. Repeat steps 2 and 3 for each ESR pulse in
       ``T2ElectronParameter.ESR['ESR_pulses']``, which by default contains the
       single pulse ``T2ElectronParameter.ESR['ESR_pulse']``.
    6. Perform empty-plunge-read sequence (EPR), but only if
       ``T2ElectronParameter.EPR['enabled']`` is True.
       EPR pulses are defined in ``T2ElectronParameter.EPR['pulses']``.
    7. Perform any post_pulses defined in ``T2ElectronParameter.post_pulses``.

    Args:
        name: Parameter name
        **kwargs: Additional kwargs passed to `AcquisitionParameter`.

    Parameters:
        ESR (dict): `T2ElectronPulseSequence` generator settings for ESR.
            Settings are: ``stage_pulse``, ``ESR_initial_pulse``,
            ``ESR_refocusing_pulse``, ``ESR_final_pulse``, ``read_pulse``,
            ``num_refocusing_pulses``, ``pre_delay``, ``inter_delay``,
            ``post_delay``.
        EPR (dict): `T2ElectronPulseSequence` generator settings for EPR.
            This is optional and can be toggled in ``EPR['enabled']``.
            If disabled, contrast is not calculated.
            Settings are: ``enabled``, ``pulses``.
        pre_pulses (List[Pulse]): Pulses to place at the start of the sequence.
        post_pulses (List[Pulse]): Pulses to place at the end of the sequence.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        t_skip (float): initial part of read trace to ignore for measuring
            blips. Useful if there is a voltage spike at the start, which could
            otherwise be measured as a ``blip``. Retrieved from
            ``silq.config.properties.t_skip``.
        t_read (float): duration of read trace to include for measuring blips.
            Useful if latter half of read pulse is used for initialization.
            Retrieved from ``silq.config.properties.t_read``.
        min_filter_proportion (float): Minimum number of read traces needed in
            which the voltage starts low (loaded donor). Otherwise, most results
            are set to zero. Retrieved from
            ``silq.config.properties.min_filter_proportion``.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.
    """
    def __init__(self, name='Electron_T2', **kwargs):
        self.pulse_sequence = T2ElectronPulseSequence()

        super().__init__(name=name,
                         names=['up_proportion', 'num_traces'],
                         labels=['Up proportion', 'Number of traces'],
                         snapshot_value=False,
                         properties_attrs=['t_skip'],
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

    def analyse(self, traces = None):
        """Analyse ESR traces.

        If there is only one ESR pulse, returns ``up_proportion_{pulse.name}``.
        If there are several ESR pulses, adds a zero-based suffix at the end for
        each ESR pulse. If ``ESRParameter.EPR['enabled'] == True``, the results
        from `analyse_EPR` are also added, as well as `contrast_{pulse.name}`
        (plus a suffix if there are several ESR pulses).
        """
        if traces is None:
            traces = self.traces

        threshold_voltage = getattr(self, 'threshold_voltage', None)

        if self.EPR['enabled']:
            # Analyse EPR sequence, which also gets the dark counts
            results = analysis.analyse_EPR(
                empty_traces=traces['empty']['output'],
                plunge_traces=traces['plunge']['output'],
                read_traces=traces['read_long']['output'],
                sample_rate=self.sample_rate,
                min_filter_proportion=self.min_filter_proportion,
                threshold_voltage=threshold_voltage,
                t_skip=self.t_skip, # Use t_skip to keep length consistent
                t_read=self.t_read)
        else:
            results = {}

        read_pulse = self.pulse_sequence.get_pulse(name=self.ESR["read_pulse"].name)
        read_traces = traces[read_pulse.full_name]['output']
        ESR_results = analysis.analyse_traces(
            traces=read_traces,
            sample_rate=self.sample_rate,
            filter='low',
            threshold_voltage=threshold_voltage,
            t_skip=self.t_skip,
            t_read=self.t_read)

        results['ESR_results'] = ESR_results
        results[f'up_proportion_{read_pulse.name}'] = ESR_results['up_proportion']
        if self.EPR['enabled']:
            # Add contrast obtained by subtracting EPR dark counts
            contrast = ESR_results['up_proportion'] - results['dark_counts']
            results[f'contrast_{read_pulse.name}'] = contrast

        return results


class NMRParameter(AcquisitionParameter):
    """ Parameter for most measurements involving an NMR pulse.

    This parameter can apply several NMR pulses, and also measure several ESR
    frequencies. It uses the `NMRPulseSequence`, which will generate a pulse
    sequence from settings (see parameters below).

    In general, the pulse sequence is as follows:

    1. Perform any pre_pulses defined in ``NMRParameter.pre_pulses``.
    2. Perform NMR sequence

       1. Perform stage pulse ``NMRParameter.NMR['stage_pulse']``.
          Default is 'empty' `DCPulse`.
       2. Perform NMR pulses within the stage pulse. The NMR pulses defined
          in ``NMRParameter.NMR['NMR_pulses']`` are applied successively.
          The delay after start of the stage pulse is
          ``NMRParameter.NMR['pre_delay']``, delays between NMR pulses is
          ``NMRParameter.NMR['inter_delay']``, and the delay after the final
          NMR pulse is ``NMRParameter.NMR['post_delay']``.

    3. Perform ESR sequence

       1. Perform stage pulse ``NMRParameter.ESR['stage_pulse']``.
          Default is 'plunge' `DCPulse`.
       2. Perform ESR pulse within stage pulse for first pulse in
          ``NMRParameter.ESR['ESR_pulses']``.
       3. Perform ``NMRParameter.ESR['read_pulse']``, and acquire trace.
       4. Repeat steps 1 - 3 for each ESR pulse. The different ESR pulses
          usually correspond to different ESR frequencies (see
          `NMRParameter`.ESR_frequencies).
       5. Repeat steps 1 - 4 for ``NMRParameter.ESR['shots_per_frequency']``
          This effectively interleaves the ESR pulses, which counters effects of
          the nucleus flipping within an acquisition.

    This acquisition is repeated ``NMRParameter.samples`` times. If the nucleus
    is in one of the states for which an ESR frequency is on resonance, a high
    ``up_proportion`` is measured, while for the other frequencies a low
    ``up_proportion`` is measured. By looking over successive samples and
    measuring how often the ``up_proportions`` switch between above/below
    ``NMRParameter.threshold_up_proportion``, nuclear flips can be measured
    (see `NMRParameter.analyse` and `analyse_flips`).

    Args:
        name: Parameter name
        **kwargs: Additional kwargs passed to `AcquisitionParameter`

    Parameters:
        NMR (dict): `NMRPulseSequence` pulse settings for NMR. Settings are:
            ``stage_pulse``, ``NMR_pulse``, ``NMR_pulses``, ``pre_delay``,
            ``inter_delay``, ``post_delay``.
        ESR (dict): `NMRPulseSequence` pulse settings for ESR. Settings are:
            ``ESR_pulse``, ``stage_pulse``, ``ESR_pulses``, ``read_pulse``,
            ``pulse_delay``.
        EPR (dict): `PulseSequenceGenerator` settings for EPR. This is optional
            and can be toggled in ``EPR['enabled']``. If disabled, contrast is
            not calculated.
        pre_pulses (List[Pulse]): Pulses to place at the start of the sequence.
        post_pulses (List[Pulse]): Pulses to place at the end of the sequence.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        ESR_frequencies (List[float]): List of ESR frequencies to use. When set,
            a copy of ``NMRParameter.ESR['ESR_pulse']`` is created for each
            frequency, and added to ``NMRParameter.ESR['ESR_pulses']``.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        t_skip (float): initial part of read trace to ignore for measuring
            blips. Useful if there is a voltage spike at the start, which could
            otherwise be measured as a ``blip``. Retrieved from
            ``silq.config.properties.t_skip``.
        t_read (float): duration of read trace to include for measuring blips.
            Useful if latter half of read pulse is used for initialization.
            Retrieved from ``silq.config.properties.t_read``.
        threshold_up_proportion (Union[float, Tuple[float, float]): threshold
            for up proportions needed to determine ESR pulse to be on-resonance.
            If tuple, first element is threshold below which ESR pulse is
            off-resonant, and second element is threshold above which ESR pulse
            is on-resonant. Useful for filtering of up proportions at boundary.
            Retrieved from
            ``silq.config.properties.threshold_up_proportion``.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.

    Note:
        - The `NMRPulseSequence` does not have an empty-plunge-read (EPR)
          sequence, and therefore does not add a contrast or dark counts.
          Verifying that the system is in tune is therefore a little bit tricky.

    """
    def __init__(self, name: str = 'NMR',
                 names: List[str] = ['flips', 'flip_probability', 'up_proportions',
                                     'state_probability_0', 'state_probability_1',
                                     'threshold_up_proportion'],
                 **kwargs):
        """
        Parameter used to determine the Rabi frequency
        """
        self.pulse_sequence = NMRPulseSequence()
        self.NMR = self.pulse_sequence.NMR
        self.ESR = self.pulse_sequence.ESR
        self.pre_pulses = self.pulse_sequence.pulse_settings['pre_pulses']
        self.pre_ESR_pulses = self.pulse_sequence.pulse_settings['pre_ESR_pulses']
        self.post_pulses = self.pulse_sequence.pulse_settings['post_pulses']

        super().__init__(name=name,
                         names=names,
                         snapshot_value=False,
                         properties_attrs=['t_read', 't_skip',
                                           'threshold_up_proportion'],
                         **kwargs)

    @property
    def names(self):
        names = []

        for name in self._names:
            if name in ['flips', 'flip_probability', 'up_proportions',
                        'state_probability_0', 'state_probability_1',
                        'threshold_up_proportion']:
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

    @property
    def ESR_frequencies(self):
        """ESR frequencies to measure.

        For each ESR frequency, ``NMRParameter.ESR['shots_per_read']`` reads
        are performed.
        """
        ESR_frequencies = []
        for pulse in self.ESR['ESR_pulses']:
            if isinstance(pulse, Pulse):
                ESR_frequencies.append(pulse.frequency)
            elif isinstance(pulse, str):
                ESR_frequencies.append(self.ESR[pulse].frequency)
            elif isinstance(pulse, Iterable):
                ESR_subfrequencies = []
                for subpulse in pulse:
                    if isinstance(subpulse, Pulse):
                        ESR_subfrequencies.append(subpulse.frequency)
                    elif isinstance(subpulse, str):
                        ESR_subfrequencies.append(self.ESR[subpulse].frequency)
                    else:
                        raise SyntaxError(f'Subpulse type not allowed: {subpulse}')
                ESR_frequencies.append(ESR_subfrequencies)
            else:
                raise SyntaxError(f'pulse type not allowed: {pulse}')
        return ESR_frequencies

    @ESR_frequencies.setter
    def ESR_frequencies(self, ESR_frequencies: List):
        assert len(ESR_frequencies) == len(self.ESR['ESR_pulses']), \
        'Different number of frequencies to ESR pulses.'

        updated_ESR_pulses = []
        for ESR_subpulses, ESR_subfrequencies in zip(self.ESR['ESR_pulses'], ESR_frequencies):
            if isinstance(ESR_subpulses, str):
                ESR_subpulses = copy(self.ESR[ESR_subpulses])
            elif isinstance(ESR_subpulses, Iterable):
                ESR_subpulses = [
                    copy(self.ESR[p]) if isinstance(p, str) else p
                    for p in ESR_subpulses]

            # Either both the subpulses and subfrequencies must be iterable, or neither are (XNOR)
            assert \
                (
                    isinstance(ESR_subpulses, Iterable) and
                    isinstance(ESR_subfrequencies, Iterable)
                ) or (
                    not (isinstance(ESR_subpulses, Iterable) or isinstance(
                        ESR_subfrequencies, Iterable))
                ), \
            'Data structures for frequencies and pulses do not have the same shape.'

            if not isinstance(ESR_subpulses, Iterable):
                ESR_subpulses = [ESR_subpulses]
            if not isinstance(ESR_subfrequencies, Iterable):
                ESR_subfrequencies = [ESR_subfrequencies]

            for pulse, frequency in zip(ESR_subpulses,
                                        ESR_subfrequencies):
                    pulse.frequency = frequency

            updated_ESR_pulses.append(ESR_subpulses)
        self.ESR['ESR_pulses'] = updated_ESR_pulses

    def analyse(self, traces: Dict[str, Dict[str, np.ndarray]] = None):
        """Analyse flipping events between nuclear states and determine nuclear state

        Returns:
            (Dict[str, Any]): Dict containing:

            * **results_read** (dict): `analyse_traces` results for each read
              trace
            * **up_proportions_{idx}** (np.ndarray): Up proportions, the
              dimensionality being equal to ``NMRParameter.samples``.
              ``{idx}`` is replaced with the zero-based ESR frequency index.
            * **state_probability_0_{idx}**, **state_probability_1_{idx}** (np.ndarray):
              respectively up, down nuclear spin state proportions when reading-out ESR1 transition
            * Results from `analyse_flips`. These are:

              - flips_{idx},
              - flip_probability_{idx}
              - combined_flips_{idx1}{idx2}
              - combined_flip_probability_{idx1}{idx2}

              Additionally, each of the above results will have another result
              with the same name, but prepended with ``filtered_``, and appended
              with ``_{idx1}{idx2}`` if not already present. Here, all the
              values are filtered out where the corresponding pair of
              up_proportion samples do not have exactly one high and one low for
              each sample. The values that do not satisfy the filter are set to
              ``np.nan``.

              * **filtered_scans_{idx1}{idx2}**:
        """
        if traces is None:
            traces = self.traces

        results = {'results_read': []}

        if hasattr(self, 'threshold_voltage'):
            threshold_voltage = getattr(self, 'threshold_voltage')
        else:
            # Calculate threshold voltages from combined read traces
            high_low = analysis.find_high_low(
                np.ravel([trace['output'] for pulse_name, trace in traces.items()
                          if pulse_name.startswith('read_initialize')]))
            threshold_voltage = high_low['threshold_voltage']

        # Extract points per shot from a single read trace
        single_read_traces_name = f"{self.ESR['read_pulse'].name}[0]"
        single_read_traces = traces[single_read_traces_name]['output']
        points_per_shot = single_read_traces.shape[1]

        self.read_traces = np.zeros((len(self.ESR_frequencies), self.samples,
                                     self.ESR['shots_per_frequency'],
                                     points_per_shot))
        up_proportions = np.zeros((len(self.ESR_frequencies), self.samples))
        state_probability_0 = np.zeros(len(self.ESR_frequencies))
        state_probability_1 = np.zeros(len(self.ESR_frequencies))
        threshold_up_proportion = np.zeros(len(self.ESR_frequencies))
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

            if self.threshold_up_proportion is None:
                threshold_up_proportion[f_idx] = analysis.analyse_threshold_up_proportion(
                    up_proportions_arrs=up_proportions[f_idx],
                    shots_per_frequency=self.ESR['shots_per_frequency'])
            else:
                threshold_up_proportion[f_idx] = self.threshold_up_proportion

            state_probability_0[f_idx] = np.sum(up_proportions[f_idx] <
                                                threshold_up_proportion[f_idx])/self.samples
            state_probability_1[f_idx] = np.sum(up_proportions[f_idx] >=
                                                threshold_up_proportion[f_idx])/self.samples

            if len(self.ESR_frequencies) > 1:
                results[f'up_proportions_{f_idx}'] = up_proportions[f_idx]
                results[f'state_probability_0_{f_idx}'] = state_probability_0[f_idx]
                results[f'state_probability_1_{f_idx}'] = state_probability_1[f_idx]
                results[f'threshold_up_proportion_{f_idx}'] = threshold_up_proportion[f_idx]
            else:
                results['up_proportions'] = up_proportions[f_idx]
                results['state_probability_0'] = state_probability_0[f_idx]
                results['state_probability_1'] = state_probability_1[f_idx]
                results['threshold_up_proportion'] = threshold_up_proportion[f_idx]

        # Add singleton dimension because analyse_flips handles 3D up_proportions
        up_proportions = np.expand_dims(up_proportions, 1)
        results_flips = analysis.analyse_flips(
            up_proportions_arrs=up_proportions,
            threshold_up_proportion=self.threshold_up_proportion,
            shots_per_frequency=self.ESR['shots_per_frequency'])
        # Add results, only choosing first element so its no longer an array
        results.update({k: v[0] for k, v in results_flips.items()})
        return results


class EDSRParameter(NMRParameter):
    """
    Parameter for EDSR measurements based on NMR parameter and pulse sequence

    Refer to NMRParameter for details. In addition to all properties copied from NMRParameter,
    EDSRParameter has additional analysis of electron readout right after EDSR(NMR) pulse during
    NMR['post_pulse'] = DCPulse('read') that needs to be present in NMRPulseSequence.

    Args:
        Refer to NMRParameter
    Parameters:
        Refer to NMRParameter
    """

    def __init__(self, name: str = 'EDSR',
                 names: List[str] = ['flips', 'flip_probability', 'up_proportions',
                                     'state_probability_0', 'state_probability_1',
                                     'EDSR_up_proportion'],
                 **kwargs):
        super().__init__(name=name,
                         names=names,
                         **kwargs)

    @property
    def names(self):
        names = super().names
        names.append('EDSR_up_proportion')
        return names

    @names.setter
    def names(self, names):
        self._names = names

    def analyse(self, traces: Dict[str, Dict[str, np.ndarray]] = None):
        """
        Reading out electron spin-up proportion after EDSR (NMR) pulse during 'read' DCPulse.

        Returns:
            (Dict[str, Any]): Dict containing:
            * all results from NMRParameter
            * **EDSR_up_proportion**: electron spin-up proportion right after EDSR (NMR) pulse.
        """
        results = super().analyse(traces)

        # Extract points for read DC pulse after EDSR pulse
        EDSR_read_traces_name = f"{self.NMR['post_pulse'].name}"
        EDSR_read_traces = traces[EDSR_read_traces_name]['output']
        EDSR_trace_points = EDSR_read_traces.shape[1]

        self.EDSR_traces = np.zeros((self.samples, EDSR_trace_points))
        self.EDSR_traces = EDSR_read_traces
        EDSR_read_result = analysis.analyse_traces(
            traces=EDSR_read_traces,
            sample_rate=self.sample_rate,
            t_read=self.t_read,
            t_skip=self.t_skip,
            threshold_voltage=self.threshold_up_proportion)
        EDSR_up_proportion = EDSR_read_result['up_proportion']
        results['EDSR_up_proportion'] = EDSR_up_proportion

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

        pulses_config = config['environment:pulses']

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

        self.ESR['ESR_pulses'] = []
        self.NMR['NMR_pulses'] = self.get_NMR_pulses(initial_state, final_state)
        self.pulse_sequence.generate()

        if run:
            self.setup(repeat=False)
            self.layout.start()
            self.layout.stop()


class FlipFlopParameter(AcquisitionParameter):
    """Parameter for performing flip-flopping, not meant for acquiring data"""
    def __init__(self, name='flip_flop', **kwargs):
        super().__init__(name=name, wrap_set=False, names=[], shapes=(),
                         **kwargs)
        self.pulse_sequence = FlipFlopPulseSequence()

        self.ESR = self.pulse_sequence.ESR
        self.pre_pulses = self.pulse_sequence.pre_pulses
        self.post_pulses = self.pulse_sequence.post_pulses

    def analyse(self, traces=None):
        return

    def set(self, frequencies=None, pre_flip=None, evaluate=True, **kwargs):
        if frequencies is not None:
            self.ESR['frequencies'] = frequencies
        if pre_flip is not None:
            self.ESR['pre_flip'] = pre_flip

        if frequencies is not None or pre_flip is not None:
            self.pulse_sequence.generate()

        if evaluate:
            self.setup(**kwargs)
            return self.get(**kwargs)


class BlipsParameter(AcquisitionParameter):
    """Parameter that measures properties of blips in a trace

    The `PulseSequence` consists of a single read pulse.
    From this trace, the number of blips per second is counted, as well as the
    mean time in ``low`` and ``high`` voltage state.
    This parameter can be used in retuning sequence.

    Args:
        name: Parameter name.
        duration: Duration of read trace
        pulse_name: Name of read pulse
        **kwargs: Additional kwargs passed to `AcquisitionParameter`.

    Parameters:
        threshold_voltage (float): Threshold voltage for a blip in voltage.
        pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
        samples (int): Number of acquisition samples
        results (dict): Results obtained after analysis of traces.
        t_skip (float): initial part of read trace to ignore for measuring
            blips. Useful if there is a voltage spike at the start, which could
            otherwise be measured as a ``blip``. Retrieved from
            ``silq.config.properties.t_skip``.
        t_read (float): duration of read trace to include for measuring blips.
            Useful if latter half of read pulse is used for initialization.
            Retrieved from ``silq.config.properties.t_read``.
        min_filter_proportion (float): Minimum number of read traces needed in
            which the voltage starts low (loaded donor). Otherwise, most results
            are set to zero. Retrieved from
            ``silq.config.properties.min_filter_proportion``.
        traces (dict): Acquisition traces segmented by pulse and acquisition
            label
        silent (bool): Print results after acquisition
        continuous (bool): If True, instruments keep running after acquisition.
            Useful if stopping/starting instruments takes a considerable amount
            of time.
        properties_attrs (List[str]): Attributes to match with
            ``silq.config.properties`` See notes below for more info.
        save_traces (bool): Save acquired traces to disk.
            If the acquisition has been part of a measurement, the traces are
            stored in a subfolder of the corresponding data set.
            Otherwise, a new dataset is created.
        dataset (DataSet): Traces DataSet
        base_folder (str): Base folder in which to save traces. If not specified,
            and acquisition is part of a measurement, the base folder is the
            folder of the measurement data set. Otherwise, the base folder is
            the default data folder
        subfolder (str): Subfolder within the base folder to save traces.


    See Also:
        - `RetuneBlipsParameter`.

    """
    def __init__(self,
                 name: str = 'count_blips',
                 duration: float = None,
                 pulse_name: str = 'DC_trace',
                 **kwargs):
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
        """Shorthand for read pulse duration."""
        return self.pulse_sequence[self.pulse_name].duration

    @duration.setter
    def duration(self, duration):
        self.pulse_sequence[self.pulse_name].duration = duration

    def analyse(self, traces = None):
        """`count_blips` analysis."""
        if traces is None:
            traces = self.traces

        return analysis.count_blips(
            traces=traces[self.pulse_name]['output'],
            t_skip=0,
            sample_rate=self.sample_rate,
            threshold_voltage=self.threshold_voltage)


class NeuralNetworkParameter(AcquisitionParameter):
    """Base parameter for neural networks

    Todo:
        - Needs to be updated
        - Transform into a `MeasurementParameter`.
    """
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
    """Parameter that uses neural network for retuning.

    Todo:
        - Needs to be updated
        - Transform into a `MeasurementParameter`.
    """
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


class ESRRamseyDetuningParameter(AcquisitionParameter):
    """Parameter for most pulse sequences involving electron spin resonance.

        This parameter can handle many of the simple pulse sequences involving ESR.
        It uses the `ESRPulseSequence`, which will generate a pulse sequence from
        settings (see parameters below).

        In general the pulse sequence is as follows:

        1. Perform any pre_pulses defined in ``ESRParameter.pre_pulses``.
        2. Perform stage pulse ``ESRParameter.ESR['stage_pulse']``.
           By default, this is the ``plunge`` pulse.
        3. Perform ESR pulse within plunge pulse, the delay from start of plunge
           pulse is defined in ``ESRParameter.ESR['pulse_delay']``.
        4. Perform read pulse ``ESRParameter.ESR['read_pulse']``.
        5. Repeat steps 2 and 3 for each ESR pulse in
           ``ESRParameter.ESR['ESR_pulses']``, which by default contains single
           pulse ``ESRParameter.ESR['ESR_pulse']``.
        6. Perform empty-plunge-read sequence (EPR), but only if
           ``ESRParameter.EPR['enabled']`` is True.
           EPR pulses are defined in ``ESRParameter.EPR['pulses']``.
        7. Perform any post_pulses defined in ``ESRParameter.post_pulses``.

        A shorthand for using the default ESR pulse for multiple frequencies is by
        setting `ESRParameter.ESR_frequencies`. Settings this will create a copy
        of ESRParameter.ESR['ESR_pulse'] with the respective frequency.

        Examples:
            The following code measures two ESR frequencies and performs an EPR
            from which the contrast can be determined for each ESR frequency:

            >>> ESR_parameter = ESRParameter()
            >>> ESR_parameter.ESR['pulse_delay'] = 5e-3
            >>> ESR_parameter.ESR['stage_pulse'] = DCPulse['plunge']
            >>> ESR_parameter.ESR['ESR_pulse'] = FrequencyRampPulse('ESR_adiabatic')
            >>> ESR_parameter.ESR_frequencies = [39e9, 39.1e9]
            >>> ESR_parameter.EPR['enabled'] = True
            >>> ESR_parameter.pulse_sequence.generate()

            The total pulse sequence is plunge-read-plunge-read-empty-plunge-read
            with an ESR pulse in the first two plunge pulses, 5 ms after the start
            of the plunge pulse. The ESR pulses have different frequencies.

        Args:
            name: Name of acquisition parameter
            **kwargs: Additional kwargs passed to `AcquisitionParameter`.

        Parameters:
            ESR (dict): `ESRPulseSequence` generator settings for ESR. Settings are:
                ``stage_pulse``, ``ESR_pulse``, ``ESR_pulses``, ``pulse_delay``,
                ``read_pulse``.
            EPR (dict): `ESRPulseSequence` generator settings for EPR.
                This is optional and can be toggled in ``EPR['enabled']``.
                If disabled, contrast is not calculated.
                Settings are: ``enabled``, ``pulses``.
            pre_pulses (List[Pulse]): Pulses to place at the start of the sequence.
            post_pulses (List[Pulse]): Pulses to place at the end of the sequence.
            pulse_sequence (PulseSequence): Pulse sequence used for acquisition.
            samples (int): Number of acquisition samples
            results (dict): Results obtained after analysis of traces.
            t_skip (float): initial part of read trace to ignore for measuring
                blips. Useful if there is a voltage spike at the start, which could
                otherwise be measured as a ``blip``. Retrieved from
                ``silq.config.properties.t_skip``.
            t_read (float): duration of read trace to include for measuring blips.
                Useful if latter half of read pulse is used for initialization.
                Retrieved from ``silq.config.properties.t_read``.
            min_filter_proportion (float): Minimum number of read traces needed in
                which the voltage starts low (loaded donor). Otherwise, most results
                are set to zero. Retrieved from
                ``silq.config.properties.min_filter_proportion``.
            traces (dict): Acquisition traces segmented by pulse and acquisition
                label
            silent (bool): Print results after acquisition
            continuous (bool): If True, instruments keep running after acquisition.
                Useful if stopping/starting instruments takes a considerable amount
                of time.
            properties_attrs (List[str]): Attributes to match with
                ``silq.config.properties``.
                See notes below for more info.
            save_traces (bool): Save acquired traces to disk.
                If the acquisition has been part of a measurement, the traces are
                stored in a subfolder of the corresponding data set.
                Otherwise, a new dataset is created.
            dataset (DataSet): Traces DataSet
            base_folder (str): Base folder in which to save traces. If not specified,
                and acquisition is part of a measurement, the base folder is the
                folder of the measurement data set. Otherwise, the base folder is
                the default data folder
            subfolder (str): Subfolder within the base folder to save traces.

        Notes:
            - All pulse settings are copies of
              ``ESRParameter.pulse_sequence.pulse_settings``.
            - For given pulse settings, ``ESRParameter.pulse_sequence.generate``
              will recreate the pulse sequence from settings.
        """

    def __init__(self, name='ESRRamsey', **kwargs):
        self._names = []

        self.pulse_sequence = ESRRamseyDetuningPulseSequence()
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
            if 'voltage_difference' in self._names:
                names.append('voltage_difference')

        ESR_pulse_names = [pulse.name for pulse in self.pulse_sequence.primary_ESR_pulses]

        for pulse in self.pulse_sequence.primary_ESR_pulses:
            pulse_name = pulse if isinstance(pulse, str) else pulse.name

            if ESR_pulse_names.count(pulse_name) == 1:
                # Ignore suffix
                name = pulse_name
            else:
                suffix = len([name for name in names
                              if f'up_proportion_{pulse_name}' in name])
                name = f'{pulse_name}_{suffix}'
            names.append(f'up_proportion_{name}')
            if self.EPR['enabled']:
                names.append(f'contrast_{name}')
            names.append(f'num_traces_{name}')
        return names

    @names.setter
    def names(self, names):
        """Set all the names to return upon .get() for the EPR sequence"""
        self._names = [name for name in names
                       if not 'contrast_' in name
                       and not 'up_proportion_' in name]

    @property_ignore_setter
    def shapes(self):
        return ((),) * len(self.names)

    @property_ignore_setter
    def units(self):
        return ('',) * len(self.names)

    @property
    def ESR_frequencies(self):
        """Apply default ESR pulse for each ESR frequency given."""
        return self.pulse_sequence.ESR_frequencies

    @ESR_frequencies.setter
    def ESR_frequencies(self, ESR_frequencies: List[float]):
        self.pulse_sequence.generate(ESR_frequencies=ESR_frequencies)

    def analyse(self, traces=None, plot=False):
        """Analyse ESR traces.

        If there is only one ESR pulse, returns ``up_proportion_{pulse.name}``.
        If there are several ESR pulses, adds a zero-based suffix at the end for
        each ESR pulse. If ``ESRParameter.EPR['enabled'] == True``, the results
        from `analyse_EPR` are also added, as well as ``contrast_{pulse.name}``
        (plus a suffix if there are several ESR pulses).
        """
        if traces is None:
            traces = self.traces

        threshold_voltage = getattr(self, 'threshold_voltage', None)

        if self.EPR['enabled']:
            # Analyse EPR sequence, which also gets the dark counts
            results = analysis.analyse_EPR(
                empty_traces=traces[self.pulse_sequence._EPR_pulses[0].full_name]['output'],
                plunge_traces=traces[self.pulse_sequence._EPR_pulses[1].full_name]['output'],
                read_traces=traces[self.pulse_sequence._EPR_pulses[2].full_name]['output'],
                sample_rate=self.sample_rate,
                min_filter_proportion=self.min_filter_proportion,
                threshold_voltage=threshold_voltage,
                filter_traces=self.filter_traces,
                t_skip=self.t_skip,  # Use t_skip to keep length consistent
                t_read=self.t_read)
        else:
            results = {}

        ESR_pulses = self.pulse_sequence.primary_ESR_pulses
        ESR_pulse_names = [pulse.name for pulse in ESR_pulses]
        read_pulses = self.pulse_sequence.get_pulses(name=self.ESR["read_pulse"].name)
        results['ESR_results'] = []

        for read_pulse, ESR_pulse in zip(read_pulses, ESR_pulses):
            read_traces = traces[read_pulse.full_name]['output']
            ESR_results = analysis.analyse_traces(
                traces=read_traces,
                sample_rate=self.sample_rate,
                filter='low' if self.filter_traces else None,
                min_filter_proportion=self.min_filter_proportion,
                threshold_voltage=threshold_voltage,
                t_skip=self.t_skip,
                t_read=self.t_read,
                plot=plot)
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
            results[f'num_traces_{pulse_label}'] = ESR_results['num_traces']

        voltage_differences = [ESR_result['voltage_difference']
                               for ESR_result in results['ESR_results']
                               if ESR_result['voltage_difference'] is not None]
        if voltage_differences:
            results['voltage_difference'] = np.mean(voltage_differences)
        else:
            results['voltage_difference'] = np.nan

        self.results = results
        return results
