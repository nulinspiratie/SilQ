from time import sleep
import numpy as np
from collections import OrderedDict

from qcodes.instrument.parameter import MultiParameter
from qcodes.data import hdf5_format, io
from qcodes.data.data_array import DataArray
from qcodes.loops import active_loop

from silq.pulses import *
from silq.analysis import analysis
from silq.tools import data_tools
from silq.tools.general_tools import SettingsClass, clear_single_settings, \
    attribute_from_config, UpdateDotDict


h5fmt = hdf5_format.HDF5Format()


class AcquisitionParameter(SettingsClass, MultiParameter):
    layout = None
    formatter = h5fmt

    def __init__(self, **kwargs):
        SettingsClass.__init__(self)

        shapes = kwargs.pop('shapes', ((), ) * len(kwargs['names']))
        MultiParameter.__init__(self, shapes=shapes, **kwargs)

        self.pulse_sequence = PulseSequence()
        """Pulse sequence of acquisition parameter"""

        self.silent = True
        """Do not print results after acquisition"""

        self.save_traces = False
        """ Save traces in separate files (does not work)"""

        self.samples = None
        self.t_read = None
        self.t_skip = None
        self.data = None
        self.dataset = None
        self.results = None

        self.subfolder = None

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

    @property
    def start_idx(self):
        return round(self.t_skip * 1e-3 * self.sample_rate)

    def store_traces(self, traces_dict, base_folder=None, subfolder=None):
        # Store raw traces
        if base_folder is None:
            # Extract base_folder from dataset of currently active loop
            active_dataset = active_loop().get_data_set()
            base_folder = active_dataset.location
        self.dataset = data_tools.create_data_set(name='traces',
                                                  base_folder=base_folder,
                                                  subfolder=subfolder)

        # Create dictionary of set arrays
        set_arrs = {}
        for traces_name, traces in traces_dict.items():
            number_of_traces, points_per_trace = traces.shape

            if traces.shape not in set_arrs:
                time_step = 1 / self.sample_rate * 1e3
                t_list = np.arange(0, points_per_trace * time_step, time_step)
                t_list_arr = DataArray(name='time',
                                       array_id='time',
                                       label=' Time (ms)',
                                       # shape=(points_per_trace, ),
                                       # preset_data=t_list,
                                       shape=traces.shape,
                                       preset_data=np.full(traces.shape,
                                                           t_list),
                                       is_setpoint=True)

                trace_num_arr = DataArray(name='trace_num',
                                          array_id='trace_num',
                                          label='Trace number',
                                          # shape=traces.shape,
                                          # preset_data=np.full(traces.shape[
                                          #                     ::-1],
                                          #                     np.arange(number_of_traces),
                                          #                     dtype=np.float64).transpose(),
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
                                  label=traces_name + ' signal (V)',
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

    def setup(self, start=False, **kwargs):
        # Create a hard copy of pulse sequence. This ensures that pulse
        # attributes no longer depend on pulse_config, and can therefore be
        # safely transferred to layout.
        pulse_sequence = self.pulse_sequence.copy()
        self.layout.target_pulse_sequence(pulse_sequence)

        samples = kwargs.pop('samples', self.samples)
        self.layout.setup(samples=samples, **kwargs)

        if start:
            self.layout.start()

    def acquire(self, **kwargs):
        # Perform acquisition
        self.data = self.layout.do_acquisition(**kwargs)


class DCParameter(AcquisitionParameter):
    # TODO implement continuous acquisition
    def __init__(self, **kwargs):
        super().__init__(name='DC_acquisition',
                         names=['DC_voltage'],
                         labels=['DC voltage'],
                         units=['V'],
                         snapshot_value=False,
                         **kwargs)

        self.samples = 1

        self.pulse_sequence.add(
            DCPulse(name='read', acquire=True, average='point',
                    connection_label='stage'),
            DCPulse(name='final',
                    connection_label='stage'))

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.layout.start()

    def acquire(self, **kwargs):
        super().acquire(start=False, stop=False)

    @clear_single_settings
    def get(self):
        # Note that this function does not have a setup, and so the setup
        # must be done once beforehand.
        self.acquire()
        self.results = [self.data['read']['output']]
        return self.results


class DCSweepParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='DC_acquisition',
                         names=['DC_voltage'],
                         labels=['DC voltage'],
                         snapshot_value=False,
                         setpoint_names=(('None',),),
                         shapes=((1,),),
                         **kwargs)

        self.pulse_duration = 1
        self.final_delay = 120

        self.additional_pulses = []
        self.samples = 1

        # Pulse to acquire trace at the end, disabled by default
        self.trace_pulse = DCPulse(name='trace',
                                   duration=100,
                                   enabled=False,
                                   acquire=True,
                                   average='trace',
                                   amplitude=0)

        self.sweep_parameters = OrderedDict()

    def __getitem__(self, item):
        return self.sweep_parameters[item]

    def add_sweep(self, parameter_name,
                  sweep_voltages=None, connection_label=None):
        if connection_label is None:
            connection_label = parameter_name

        self.sweep_parameters[parameter_name] = UpdateDotDict(
            update_function=self.generate_pulse_sequence,
            name=parameter_name,
            sweep_voltages=sweep_voltages,
            connection_label=connection_label)

        self.generate_pulse_sequence()

    def generate_pulse_sequence(self):
        self.pulse_sequence.clear()

        iter_sweep_parameters = iter(self.sweep_parameters.items())
        if len(self.sweep_parameters) == 1:
            sweep_name, sweep_dict = next(iter_sweep_parameters)
            sweep_voltages = sweep_dict.sweep_voltages

            pulses = [DCPulse('DC_read',
                              duration=self.pulse_duration,
                              acquire=True,
                              amplitude=sweep_voltage,
                              connection_label=sweep_dict.connection_label)
                      for sweep_voltage in sweep_voltages]
            self.pulse_sequence = PulseSequence(pulses=pulses)
            #             self.pulse_sequence.add(*self.additional_pulses)

            self.setpoint_names = ((sweep_name,),)
            self.shapes = ((len(sweep_voltages),),)
            self.setpoints = ((sweep_voltages,),)
        elif len(self.sweep_parameters) == 2:
            inner_sweep_name, inner_sweep_dict = next(iter_sweep_parameters)
            inner_sweep_voltages = inner_sweep_dict.sweep_voltages
            inner_connection_label = inner_sweep_dict.connection_label
            outer_sweep_name, outer_sweep_dict = next(iter_sweep_parameters)
            outer_sweep_voltages = outer_sweep_dict.sweep_voltages
            outer_connection_label = outer_sweep_dict.connection_label

            pulses = []
            if outer_connection_label == inner_connection_label:
                for outer_sweep_voltage in outer_sweep_voltages:
                    for inner_sweep_voltage in inner_sweep_voltages:
                        sweep_voltage = (inner_sweep_voltage,
                                         outer_sweep_voltage)
                        pulses.append(
                            DCPulse('DC_read',
                                    duration=self.pulse_duration,
                                    acquire=True,
                                    amplitude=sweep_voltage,
                                    average='point',
                                    connection_label=outer_connection_label))
            else:
                t = 0
                for outer_sweep_voltage in outer_sweep_voltages:
                    pulses.append(
                        DCPulse('DC_outer',
                                t_start=t,
                                duration=self.pulse_duration * len(
                                    inner_sweep_voltages),
                                amplitude=outer_sweep_voltage,
                                connection_label=outer_connection_label))
                    for inner_sweep_voltage in inner_sweep_voltages:
                        pulses.append(
                            DCPulse('DC_read',
                                    t_start=t,
                                    duration=self.pulse_duration,
                                    acquire=True,
                                    average='point',
                                    amplitude=inner_sweep_voltage,
                                    connection_label=inner_connection_label))
                        t += self.pulse_duration

            self.setpoint_names = (inner_sweep_name, outer_sweep_name),
            self.shapes = (len(inner_sweep_voltages),
                            len(outer_sweep_voltages),),
            self.setpoints = (inner_sweep_voltages, outer_sweep_voltages),
        else:
            raise NotImplementedError(
                f"Cannot handle {len(self.sweep_parameters)} parameters")

        if self.trace_pulse.enabled:
            # Also obtain a time trace at the end
            pulses.append(self.trace_pulse)
            self.names += ('DC_voltage',),
            self.setpoint_names += ('time',),
            points = round(self.trace_pulse.duration * 1e-3 * self.sample_rate)
            setpoints = np.linspace(0, self.trace_pulse.duration, points)
            self.setpoints += (setpoints,),
            self.shapes += (len(setpoints),),

        self.pulse_sequence = PulseSequence(pulses=pulses)
        self.pulse_sequence.duration += self.final_delay

    def acquire(self, stop=False, **kwargs):
        super().acquire(stop=stop, **kwargs)

        # Process results
        DC_voltages = np.array([self.data[pulse.full_name]['output']
                                for pulse in
                                self.pulse_sequence.get_pulses(name='DC_read')])
        if len(self.sweep_parameters) == 1:
            self.results = [DC_voltages]
        elif len(self.sweep_parameters) == 2:
            self.results = [DC_voltages.reshape(self.shapes[0])]
        else:
            raise NotImplementedError(
                f"Cannot handle {len(self.sweep_parameters)} parameters")

        if self.trace_pulse.enabled:
            self.results.append(self.data['trace']['output'])

        return self.results

    @clear_single_settings
    def get(self):
        self.setup()
        self.acquire(stop=True)
        return self.results


class EPRParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='EPR_acquisition',
                         names=['contrast', 'dark_counts', 'voltage_difference',
                                'fidelity_empty', 'fidelity_load'],
                         labels=['Contrast', 'Dark counts',
                                 'Voltage difference',
                                 'Fidelity empty', 'Fidelity load'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            DCPulse('empty', acquire=True),
            DCPulse('plunge', acquire=True),
            DCPulse('read_long', acquire=True),
            DCPulse('final'))

    @clear_single_settings
    def get(self):
        self.setup()

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
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read_long', acquire=True),
            DCPulse('final'),
            FrequencyRampPulse('adiabatic_ESR'))

        self.pulse_sequence.sort()

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
        self.setup()

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
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         labels=['Contrast', 'Dark counts',
                                'Voltage difference'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            SinePulse('rabi_ESR', duration=0.1))

        # Disable previous pulse for sine pulse, since it would
        # otherwise be after 'final' pulse
        self.pulse_sequence.sort()

    @property
    def frequency(self):
        return self.pulse_sequence['rabi'].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['rabi'].frequency = frequency

    def acquire(self, **kwargs):
        super().acquire(**kwargs)

    @clear_single_settings
    def get(self):
        self.setup()

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


class RabiDriveParameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        """
        Parameter used to drive Rabi oscillations
        """
        super().__init__(name='rabi_drive',
                         names=['contrast', 'dark_counts',
                                'voltage_difference'],
                         labels=['Contrast', 'Dark counts',
                                'Voltage difference'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge', acquire=True),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            SinePulse('rabi_ESR', duration=0.1))

        self.pulse_sequence.sort()

    @property
    def frequency(self):
        return self.pulse_sequence['rabi'].frequency

    @frequency.setter
    def frequency(self, frequency):
        self.pulse_sequence['rabi'].frequency = frequency

    @property
    def duration(self):
        return self.pulse_sequence['rabi'].duration

    @duration.setter
    def duration(self, duration):
        self.pulse_sequence['rabi'].duration = duration

    def acquire(self, **kwargs):
        super().acquire(**kwargs)

    @clear_single_settings
    def get(self):
        self.setup()

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


class T1Parameter(AcquisitionParameter):
    def __init__(self, **kwargs):
        super().__init__(name='T1_acquisition',
                         names=['up_proportion', 'num_traces'],
                         labels=['Up proportion', 'Number of traces'],
                         snapshot_value=False,
                         **kwargs)

        self.pulse_sequence.add(
            SteeredInitialization('steered_initialization', enabled=False),
            DCPulse('plunge'),
            DCPulse('read', acquire=True),
            DCPulse('final'),
            FrequencyRampPulse('adiabatic_ESR'))
        self.pulse_sequence.sort()

        self.readout_threshold_voltage = None

        self._meta_attrs.append('readout_threshold_voltage')

    @property
    def wait_time(self):
        return self.pulse_sequence['plunge'].duration

    def acquire(self, **kwargs):
        super().acquire(**kwargs)

    @clear_single_settings
    def get(self):
        self.setup()

        self.acquire()

        # Analysis
        fidelities = analysis.analyse_read(
            traces=self.data['read']['output'],
            threshold_voltage=self.readout_threshold_voltage,
            start_idx=self.start_idx)
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
        self.setup()

        self.acquire()

        fidelities = analysis.analyse_read(
            traces=self.data['read']['output'],
            threshold_voltage=self.readout_threshold_voltage,
            start_idx=self.start_idx)
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
                         units=['V'],
                         snapshot_value=False,
                         setpoint_names=(('time',),),
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

    @property
    def setpoints(self):
        duration = sum(pulse.duration for pulse in
                       self.pulse_sequence.get_pulses(acquire=True))
        return (duration * 1e-3 * self.sample_rate, ),

    @setpoints.setter
    def setpoints(self, setpoints):
        pass

    @property
    def shapes(self):
        shapes = self.layout.acquisition_shapes
        pts = sum(shapes[pulse_name]['output']
                  for pulse_name in ['plunge', 'read', 'empty'])
        return (pts,),

    @shapes.setter
    def shapes(self, shapes):
        pass

    def get(self):
        self.setup()

        self.acquire()

        self.results = np.concatenate([self.data['plunge']['output'],
                                       self.data['read']['output'],
                                       self.data['empty']['output']])
        return self.results,