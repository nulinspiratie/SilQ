from time import sleep
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from blinker import signal
from functools import partial

from qcodes.instrument.parameter import MultiParameter
from qcodes.data import hdf5_format, io
from qcodes.data.data_array import DataArray
from qcodes.loops import active_loop

from silq import config
from silq.pulses import *
from silq.analysis import analysis
from silq.tools import data_tools
from silq.tools.general_tools import SettingsClass, clear_single_settings, \
    attribute_from_config, UpdateDotDict, convert_setpoints, \
    property_ignore_setter


h5fmt = hdf5_format.HDF5Format()


class AcquisitionParameter(SettingsClass, MultiParameter):
    layout = None
    formatter = h5fmt

    def __init__(self, continuous=False, environment='default',
                 properties_attrs=[], **kwargs):
        SettingsClass.__init__(self)

        shapes = kwargs.pop('shapes', ((), ) * len(kwargs['names']))
        MultiParameter.__init__(self, shapes=shapes, **kwargs)

        self.pulse_sequence = PulseSequence()
        """Pulse sequence of acquisition parameter"""

        self.silent = True
        """Do not print results after acquisition"""

        self.save_traces = False
        """ Save traces in separate files"""

        if environment == 'default':
            environment = config.properties.get('default_environment',
                                                'default')
        self.environment = environment

        # Setup properties config. If pulse requires additional
        # properties_attrs, place them before calling Pulse.__init__,
        # else they are not added to attrs.
        # Make sure that self.properties_attrs is never replaced, only appended.
        # Else it is no longer used for self._handle_properties_config_signal.
        try:
            # Set properties_config from SilQ environment config
            self.properties_config = config[self.environment].properties
        except (KeyError, AttributeError):
            self.properties_config = None

        self.properties_attrs = properties_attrs

        # Set handler that only uses attributes in properties_attrs
        self._handle_properties_config_signal = partial(
            self._handle_config_signal,
            select=self.properties_attrs)
        # Connect changes in properties config to handling method
        # If environment has no properties key, this will never be called.
        signal(f'config:{self.environment}.properties').connect(
            self._handle_properties_config_signal)

        # Set attributes that can also be retrieved from properties_config
        if self.properties_config is not None:
            for attr in self.properties_attrs:
                setattr(self, attr, self.properties_config.get(attr, None))

        self.samples = None
        self.data = None
        self.dataset = None
        self.results = None

        self.subfolder = None

        self.continuous = continuous

        # Change attribute data_manager from class attribute to instance
        # attribute. This is necessary to ensure that the data_manager is
        # passed along when the parameter is spawned from a new process
        self.layout = self.layout

        self._meta_attrs.extend(['label', 'name', 'pulse_sequence'])
    def __repr__(self):
        return '{} acquisition parameter'.format(self.name)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return attribute_from_config(item)

    @property
    def sample_rate(self):
        """ Acquisition sample rate """
        return self.layout.sample_rate

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
                     channels=['output']):
        # Store raw traces
        if base_folder is None:
            # Extract base_folder from dataset of currently active loop
            active_dataset = active_loop().get_data_set()
            if active_dataset.location:
                base_folder = active_dataset.location
            elif hasattr(active_dataset, '_location'):
                base_folder = active_dataset._location
        self.dataset = data_tools.create_data_set(name='traces',
                                                  base_folder=base_folder,
                                                  subfolder=subfolder,
                                                  formatter=self.formatter)

        # Create dictionary of set arrays
        set_arrs = {}
        traces_dict = {}
        for pulse_name, channel_traces in pulse_traces.items():
            for channel in channels:
                traces_name = f'{pulse_name}_{channel}'
                traces = channel_traces[channel]
                traces_dict[traces_name] = traces

                number_of_traces, points_per_trace = traces.shape

                if traces.shape not in set_arrs:
                    time_step = 1 / self.sample_rate * 1e3
                    t_list = np.arange(0, points_per_trace * time_step, time_step)
                    t_list_arr = DataArray(name='time',
                                           array_id='time',
                                           label=' Time',
                                           unit='ms',
                                           shape=traces.shape,
                                           preset_data=np.full(traces.shape,
                                                               t_list),
                                           is_setpoint=True)

                    trace_num_arr = DataArray(name='trace_num',
                                              array_id='trace_num',
                                              label='Trace',
                                              unit='num',
                                              shape=(number_of_traces, ),
                                              preset_data=np.arange(
                                                  number_of_traces, dtype=np.float64),
                                              is_setpoint=True)
                    set_arrs[traces.shape] = (trace_num_arr, t_list_arr)

        # Add set arrays to dataset
        for k, (t_list_arr, trace_num_arr) in enumerate(set_arrs.values()):
            for arr in (t_list_arr, trace_num_arr):
                if len(set_arrs) > 1:
                    # Need to give individual array_ids to each of the set arrays
                    arr.array_id += '_{}'.format(k)
                self.dataset.add_array(arr)

        # Add trace arrs to dataset
        for traces_name, traces in traces_dict.items():
            t_list_arr, trace_num_arr = set_arrs[traces.shape]

            # Must transpose traces array
            trace_arr = DataArray(name=traces_name,
                                  array_id=traces_name,
                                  label=traces_name + ' signal',
                                  unit='V',
                                  shape=traces.shape,
                                  preset_data=traces,
                                  set_arrays=(t_list_arr, trace_num_arr))
            self.dataset.add_array(trace_arr)

        self.dataset.finalize()

    def print_results(self):
        if self.names is not None:
            for name, result in zip(self.names, self.results):
                print('{}: {:.3f}'.format(name, result))
        else:
            print('{}: {:.3f}'.format(self.name, self.results))

    def setup(self, start=None, **kwargs):
        # Create a hard copy of pulse sequence. This ensures that pulse
        # attributes no longer depend on pulse_config, and can therefore be
        # safely transferred to layout.
        pulse_sequence = self.pulse_sequence.copy()
        self.layout.target_pulse_sequence(pulse_sequence)

        samples = kwargs.pop('samples', self.samples)
        self.layout.setup(samples=samples, **kwargs)

        if start is None:
            start = self.continuous

        if start:
            self.layout.start()

    def acquire(self, start=None, stop=None, setup=None, **kwargs):
        if start is None and stop is None:
            start = not self.continuous
            stop = not self.continuous

        if setup is None and not self.continuous:
            self.setup()

        # Perform acquisition
        self.data = self.layout.acquisition(start=start, stop=stop, **kwargs)
    #
    # def plot_traces(self, channel='output'):
    #     fig, ax = plt.subplots(1,1)
    #
    #     acquire_pulses = self.pulse_sequence.get_pulses(acquire=True)
    #     if len((pulse.average for pulse in acquire_pulses)) > 1:
    #         raise RuntimeError('All pulses must have same average mode')
    #
    #     acquire_traces = {pulse.name: self.data[pulse.name][channel]
    #                       for pulse in acquire_pulses}
    #
    #     if acquire_pulses[0].average == 'trace':
    #
    #     elif acquire_pulses[0].average == 'none':
    #         cax = ax.pcolormesh(range(traces.shape[1]),
    #                             range(traces.shape[0] + 1), traces)
    #         ax.set_xlim([0, traces.shape[1]])
    #         ax.set_ylim([0, traces.shape[0] + 1])
    #         ax.invert_yaxis()
    #
    #     plt.colorbar(cax)
    #
    #     if plot1D:
    #         fig, axes = plt.subplots(len(traces), sharex=True)
    #         for k, trace in enumerate(traces):
    #             axes[k].plot(trace)
    #             #         axes[k].plot(trace > 0.5)
    #             if traces_AWG is not None:
    #                 trace_AWG = traces_AWG[k]
    #                 trace_AWG /= (np.max(trace_AWG) - np.min(trace_AWG))
    #                 trace_AWG -= np.min(trace_AWG)
    #                 axes[k].plot(trace_AWG)
    #             if threshold_voltage is not None:
    #                 axes[k].plot([threshold_voltage] * len(trace), 'r')
    #             axes[k].locator_params(nbins=2)


class DCParameter(AcquisitionParameter):
    # TODO implement continuous acquisition
    def __init__(self, **kwargs):
        super().__init__(name='DC_acquisition',
                         names=['DC_voltage'],
                         labels=['DC voltage'],
                         units=['V'],
                         snapshot_value=False,
                         continuous = True,
                         **kwargs)

        self.samples = 1

        self.pulse_sequence.add(
            DCPulse(name='read', acquire=True, average='point',
                    connection_label='stage'),
            DCPulse(name='final',
                    connection_label='stage'))

    @clear_single_settings
    def get(self):
        # Note that this function does not have a setup, and so the setup
        # must be done once beforehand.
        self.acquire()
        self.results = [self.data['read']['output']]
        return self.results


class TraceParameter(AcquisitionParameter):
    """An acquisition parameter for obtaining a trace or multiple traces
    of a given PulseSequence.

    A generic initial PulseSequence is defined, but can be redefined at
    run-time.
    e.g.
        parameter.average_mode = 'none'
        parameter.pulse_sequence = my_pulse_sequence

    Note that for the above example, all pulses in my_pulse_sequence will be
    copied and then their 'average' attribute will be set to the parameter's
    'average_mode' attribute.

    """
    def __init__(self, average_mode='none', **kwargs):
        self._average_mode = average_mode
        self._pulse_sequence = PulseSequence()
        self.samples = 1
        self.trace_pulse = 'trace_pulse'

        super().__init__(name='Trace_acquisition',
                         names=self.names,
                         labels=self.names,
                         units=self.units,
                         shapes=self.shapes,
                         snapshot_value=False,
                         **kwargs)

    @property
    def average_mode(self):
        return self._average_mode

    @average_mode.setter
    def average_mode(self, mode):
        if (self._average_mode != mode):
            self._average_mode = mode
            if self.trace_pulse in self.pulse_sequence:
                self.pulse_sequence[self.trace_pulse].average = mode

    @property
    def pulse_sequence(self):
        return self._pulse_sequence

    @pulse_sequence.setter
    def pulse_sequence(self, pulse_sequence):
        self._pulse_sequence = pulse_sequence.copy()

    @property_ignore_setter
    def names(self):
        return tuple(self.trace_pulse + f'_{output[1]}'
                for output in self.layout.acquisition_outputs())

    @property_ignore_setter
    def units(self):
        return ('V', ) * len(self.layout.acquisition_outputs())

    @property_ignore_setter
    def shapes(self):
        if self.trace_pulse in self.pulse_sequence:
            return self.pulse_sequence.get_trace_shapes(self.layout.sample_rate, self.samples)[self.trace_pulse]
        else:
            return ((1,),) * len(self.layout.acquisition_outputs())


    @property_ignore_setter
    def setpoints(self):
        if self.trace_pulse in self.pulse_sequence:
            duration = self.pulse_sequence[self.trace_pulse].duration
        else:
            return ((1,),) * len(self.layout.acquisition_outputs())

        num_traces = len(self.layout.acquisition_outputs())

        pts = duration * self.sample_rate
        t_list = np.linspace(0, duration, pts, endpoint=True)

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
                   len(self.layout.acquisition_outputs())
        else:
            return (('time', ), ) * len(self.layout.acquisition_outputs())


    @property_ignore_setter
    def setpoint_units(self):
        if self.samples > 1 and self.average_mode == 'none':
            return ((None, 'ms', ), ) * len(self.layout.acquisition_outputs())
        else:
            return (('ms', ), ) * len(self.layout.acquisition_outputs())


    def setup(self, start=None, **kwargs):
        """
        Modifies provided pulse sequence by creating a single
        pulse which overlaps all other pulses with acquire=True and
        then acquires only this pulse.
        """

        if self.trace_pulse not in self.pulse_sequence:
            # Find the start and stop times for all acquired pulses
            t_start = min(pulse.t_start for pulse in self.pulse_sequence.get_pulses())
            t_stop = max(pulse.t_stop for pulse in self.pulse_sequence.get_pulses())

            # Ensure that each pulse is not acquired as this could cause
            # overlapping issues
            for pulse in self.pulse_sequence.get_pulses():
                pulse.acquire = False

            # Create a new single pulse to acquire with t_start and t_stop
            self.pulse_sequence.add(MeasurementPulse(self.trace_pulse, t_start=t_start,
                                                     t_stop=t_stop, acquire=True,
                                                     average=self.average_mode))

        super().setup(start=start, **kwargs)

    def acquire(self, **kwargs):
        """
        Acquires the number of traces defined in self.samples

        return:  A tuple of data points. e.g.
                 ((data_for_1st_output), (data_for_2nd_output), ...)
        """
        super().acquire(**kwargs)
        traces = []

        # Merge all pulses together for a single acquisition channel
        for k, (_, output) in enumerate(self.layout.acquisition_outputs()):
            trace = self.data[self.trace_pulse][output]
            traces.append(trace)

        return traces

    @clear_single_settings
    def get(self):
        # Note that this function does not have a setup, and so the setup
        # must be done once beforehand.
        traces = self.acquire()

        self.results = {self.names[k] : trace
                        for k, trace in enumerate(traces)}

        return tuple(self.results[name] for name in self.names)


class DCSweepParameter(AcquisitionParameter):
    def __init__(self, **kwargs):

        self.sweep_parameters = OrderedDict()
        # Pulse to acquire trace at the end, disabled by default
        self.trace_pulse = DCPulse(name='trace', duration=100, enabled=False,
                                   acquire=True, average='trace', amplitude=0)

        super().__init__(name='DC_acquisition', names=['DC_voltage'],
                         labels=['DC voltage'], units=['V'],
                         snapshot_value=False, setpoint_names=(('None',),),
                         shapes=((1,),), **kwargs)

        self.pulse_duration = 1
        self.final_delay = 120
        self.inter_delay = 0.2
        self.use_ramp = False

        self.additional_pulses = []
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
            points = round(self.trace_pulse.duration * 1e-3 * self.sample_rate)
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
                self.trace_pulse.duration * 1e-3 * self.sample_rate),),
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
            setpoint_units += (('ms',), )
        return setpoint_units

    def add_sweep(self, parameter_name, sweep_voltages=None,
                  connection_label=None, offset_parameter=None):
        if connection_label is None:
            connection_label = parameter_name

        self.sweep_parameters[parameter_name] = UpdateDotDict(
            update_function=self.generate_pulse_sequence, name=parameter_name,
            sweep_voltages=sweep_voltages, connection_label=connection_label,
            offset_parameter=offset_parameter)

        self.generate_pulse_sequence()

    def generate_pulse_sequence(self):
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

    def acquire(self, **kwargs):
        super().acquire(**kwargs)

        # Process results
        DC_voltages = np.array(
            [self.data[pulse.full_name]['output'] for pulse in
             self.pulse_sequence.get_pulses(name='DC_inner')])

        if self.use_ramp:
            if len(self.sweep_parameters) == 1:
                self.results = [DC_voltages[0]]
            elif len(self.sweep_parameters) == 2:
                self.results = [DC_voltages]
        else:
            if len(self.sweep_parameters) == 1:
                self.results = [DC_voltages]
            elif len(self.sweep_parameters) == 2:
                self.results = [DC_voltages.reshape(self.shapes[0])]

        if self.trace_pulse.enabled:
            self.results.append(self.data['trace']['output'])

        return self.results

    @clear_single_settings
    def get(self):
        self.acquire()
        return self.results


class EPRParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='EPR_acquisition',
                         names=['contrast', 'dark_counts',
                                'voltage_difference_read',
                                'fidelity_empty', 'fidelity_load'],
                         snapshot_value=False,
                         properties_attrs=['t_skip', 't_read'],
                         **kwargs)

        self.pulse_sequence.add(
            DCPulse('empty', acquire=True, connection_label='stage'),
            DCPulse('plunge', acquire=True, connection_label='stage'),
            DCPulse('read_long', acquire=True, connection_label='stage'),
            DCPulse('final', connection_label='stage'))

    @property
    def labels(self):
        return [name.replace('_', ' ').capitalize() for name in self.names]

    @labels.setter
    def labels(self, labels):
        pass

    @clear_single_settings
    def get(self):
        self.acquire()

        fidelities = analysis.analyse_EPR(pulse_traces=self.data,
                                          sample_rate=self.sample_rate,
                                          t_skip=self.t_skip,
                                          t_read=self.t_read)
        self.results = [fidelities[name] for name in self.names]

        if self.save_traces:
            self.store_traces(self.data)

        if not self.silent:
            self.print_results()

        return self.results


class AdiabaticParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to perform an adiabatic sweep
        """
        super().__init__(name='adiabatic_acquisition',
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         labels=['Contrast', 'Dark counts',
                                 'Voltage difference'],
                         snapshot_value=False,
                         properties_attrs=['t_skip', 't_read'],
                         **kwargs)

        self.pulse_sequence.add(
            # SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True, connection_label='stage'),
            DCPulse('read_long', acquire=True, connection_label='stage'),
            DCPulse('final', connection_label='stage'),
            FrequencyRampPulse('adiabatic_ESR', connection_label='ESR'))

    @property
    def frequency(self):
        return self.pulse_sequence['adiabatic'].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['adiabatic'].frequency = frequency

    def acquire(self, **kwargs):
        super().acquire(**kwargs)

    @clear_single_settings
    def get(self):
        self.acquire()

        fidelities = analysis.analyse_PR(pulse_traces=self.data,
                                         sample_rate=self.sample_rate,
                                         t_skip=self.t_skip,
                                         t_read=self.t_read)
        self.results = [fidelities[name] for name in self.names]

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            self.store_traces(self.data, subfolder=self.subfolder)

        if not self.silent:
            self.print_results()

        return self.results


class RabiParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to determine the Rabi frequency
        """
        super().__init__(name='rabi_acquisition',
                         names=['contrast_ESR', 'contrast', 'dark_counts',
                                'voltage_difference_read'],
                         labels=['ESR contrast', 'Contrast', 'Dark counts',
                                 'Voltage difference read'],
                         snapshot_value=False,
                         properties_attrs=['t_skip', 't_read'],
                         **kwargs)

        self.pulse_sequence.add(
            # SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', connection_label='stage'),
            DCPulse('read', acquire=True, connection_label='stage'),
            DCPulse('empty', acquire=True, connection_label='stage'),
            DCPulse('plunge', acquire=True, connection_label='stage'),
            DCPulse('read_long', acquire=True, connection_label='stage'),
            DCPulse('final', connection_label='stage'),
            SinePulse('ESR', connection_label='ESR'))

    @property
    def frequency(self):
        return self.pulse_sequence['ESR'].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['ESR'].frequency = frequency

    @clear_single_settings
    def get(self):
        self.acquire()

        self.results = analysis.analyse_PREPR(pulse_traces=self.data,
                                              sample_rate=self.sample_rate,
                                              t_skip=self.t_skip,
                                              t_read=self.t_read)

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            self.store_traces(self.data, subfolder=self.subfolder)

        if not self.silent:
            self.print_results()

        return self.results


class T1Parameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='T1_acquisition',
                         names=['up_proportion', 'num_traces'],
                         labels=['Up proportion', 'Number of traces'],
                         snapshot_value=False,
                         properties_attrs=['t_skip'],
                         **kwargs)

        self.pulse_sequence.add(
            # SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('empty', connection_label='stage'),
            DCPulse('plunge', connection_label='stage'),
            DCPulse('read', acquire=True, connection_label='stage'),
            DCPulse('final', connection_label='stage'))
            # FrequencyRampPulse('adiabatic_ESR'))

        self.readout_threshold_voltage = None

        self._meta_attrs.append('readout_threshold_voltage')

    @property
    def wait_time(self):
        return self.pulse_sequence['plunge'].duration

    @clear_single_settings
    def get(self):
        self.acquire()

        # Analysis
        fidelities = analysis.analyse_read(
            traces=self.data['read']['output'],
            threshold_voltage=self.readout_threshold_voltage,
            start_idx=round(self.t_skip * 1e-3 * self.sample_rate))
        self.results = [fidelities[name] for name in self.names]

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            if self.subfolder is not None:
                subfolder = '{}/tau_{:.0f}'.format(self.subfolder,
                                               self.wait_time)
            else:
                subfolder = 'tau_{:.0f}'.format(self.wait_time)

            self.store_traces(self.data, subfolder=subfolder)

        if not self.silent:
            self.print_results()

        return self.results


class DarkCountsParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to perform an adiabatic sweep
        """
        super().__init__(name='dark_counts_acquisition',
                         names=['dark_counts'],
                         labels=['Dark counts'],
                         snapshot_value=False,
                         properties_attrs=['t_skip'],
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=True),
            DCPulse('read', acquire=True))

        self.readout_threshold_voltage = None

        self._meta_attrs.append('readout_threshold_voltage')

    def acquire(self, **kwargs):
        super().acquire(**kwargs)

    @clear_single_settings
    def get(self):
        self.acquire()

        fidelities = analysis.analyse_read(
            traces=self.data['read']['output'],
            threshold_voltage=self.readout_threshold_voltage,
            start_idx=round(self.t_skip * 1e-3 * self.sample_rate))
        self.results = [fidelities['up_proportion']]

        # Store raw traces if self.save_traces is True
        if self.save_traces:
            self.store_traces(self.data, subfolder=self.subfolder)

        if not self.silent:
            self.print_results()

        return self.results


class VariableReadParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='variable_read_acquisition',
                         names=('read_voltage',),
                         labels=('Read voltage',),
                         units=('V',),
                         shapes=((1,),),
                         setpoint_names=(('time',),),
                         setpoint_labels=(('Time',),),
                         setpoint_units=(('ms',),),
                         snapshot_value=False,
                         **kwargs)
        self.pulse_sequence.add(
            DCPulse(name='plunge', acquire=True, average='trace',
                    connection_label='stage'),
            DCPulse(name='read', acquire=True, average='trace',
                    connection_label='stage'),
            DCPulse(name='empty', acquire=True, average='trace',
                    connection_label='stage'),
            DCPulse(name='final',
                    connection_label='stage'))

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

    def get(self):
        self.acquire()

        self.results = np.concatenate([self.data['plunge']['output'],
                                       self.data['read']['output'],
                                       self.data['empty']['output']])
        return self.results,