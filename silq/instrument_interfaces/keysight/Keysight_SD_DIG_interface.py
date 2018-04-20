from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses.pulse_types import TriggerPulse
from qcodes.utils import validators as vals
from qcodes import ManualParameter
from qcodes.instrument_drivers.Keysight.SD_common.SD_acquisition_controller import Triggered_Controller

import logging
logger = logging.getLogger(__name__)

import numpy as np

class Keysight_SD_DIG_interface(InstrumentInterface):
    def __init__(self, instrument_name, acquisition_controller_names=[], **kwargs):
        super().__init__(instrument_name, **kwargs)
        self.pulse_sequence.allow_untargeted_pulses = True
        self.pulse_sequence.allow_pulse_overlap = True

        # Initialize channels
        self._acquisition_channels  = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, input=True)
            for k in range(self.instrument.n_channels)}

        self._pxi_channels = {
            f'pxi{k}': Channel(instrument_name=self.instrument_name(),
                               name=f'pxi{k}', id=4000 + k, input_trigger=True,
                               output=True, input=True)
            for k in range(self.instrument.n_triggers)}

        self._channels = {
            **self._acquisition_channels ,
            **self._pxi_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input=True),
        }

        # Organize acquisition controllers
        self.acquisition_controllers = {}
        for acquisition_controller_name in acquisition_controller_names:
            self.add_acquisition_controller(acquisition_controller_name)

        self.add_parameter(name='default_acquisition_controller',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(None, *self.acquisition_controllers.keys()))

        self.add_parameter(name='acquisition_controller',
                           set_cmd=None)

        # Names of acquisition channels [chA, chB, etc.]
        self.add_parameter(name='acquisition_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Anything())

        # Add ManualParameters which will be distributed to the active acquisition
        # controller during the setup routine
        self.add_parameter('sample_rate',
                           vals=vals.Numbers(),
                           set_cmd=None)

        self.add_parameter('samples',
                           vals=vals.Numbers(),
                           set_cmd=None)

        self.add_parameter('channel_selection',
                           set_cmd=None)

        self.add_parameter('minimum_timeout_interval',
                           unit='s',
                           vals=vals.Numbers(),
                           initial_value=5,
                           set_cmd=None)

        # Set up the driver to a known default state
        self.initialize_driver()

    def acquisition(self):
        """
        Perform acquisition
        """
        acquisition_data = self.acquisition_controller().acquisition()

        acquisition_average_mode = self.acquisition_controller().average_mode()

        # The start of acquisition
        t_start = min(self.pulse_sequence.t_start_list)

        # Split data into pulse traces
        data = {}
        for pulse in self.pulse_sequence.get_pulses(acquire=True):
            name = pulse.full_name
            data[name] = {}
            for ch_idx, ch in enumerate(self.channel_selection()):
                ch_data = acquisition_data[ch_idx]
                ch_name = f'ch{ch}'
                ts = (pulse.t_start - t_start, pulse.t_stop - t_start)
                sample_range = [int(round(t * self.sample_rate())) for t in ts]
                pts = len(sample_range)

                # Extract pulse data from the channel data
                data[name][ch_name] = ch_data[:, sample_range[0]:sample_range[1]]
                # Further average the pulse data
                if pulse.average == 'none':
                    pass
                elif pulse.average == 'trace':
                    data[name][ch_name] = np.mean(data[name][ch_name], axis=0)
                elif pulse.average == 'point':
                    data[name][ch_name] = np.mean(data[name][ch_name])
                elif 'point_segment' in pulse.average:
                    segments = int(pulse.average.split(':')[1])
                    split_arrs = np.array_split(data[name][ch_name], segments, axis=1)
                    data[name][ch_name] = np.mean(split_arrs, axis=(1,2))
                else:
                    raise SyntaxError(f'average mode {pulse.average} not configured')

        return data

    def add_acquisition_controller(self, acquisition_controller_name,
                                   cls_name=None):
        """
        Adds an acquisition controller to the available controllers.
        If another acquisition controller exists of the same class, it will
        be overwritten.

        Args:
            acquisition_controller_name: instrument name of controller.
                Must be on same server as interface and Keysight digitizer
            cls_name: Optional name of class, which is used as controller key.
                If no cls_name is provided, it is found from the instrument
                class name
        """
        acquisition_controller = self.find_instrument(acquisition_controller_name)
        if cls_name is None:
            cls_name = acquisition_controller.__class__.__name__
        # Remove _Controller from cls_name

        cls_name = cls_name.replace('_Controller', '')

        self.acquisition_controllers[cls_name] = acquisition_controller

    def initialize_driver(self):
        """
            Puts driver into a known initial state. Further configuration will
            be done in the configure_driver and get_additional_pulses
            functions.
        """
        for k in range(self.instrument.n_channels):
            self.instrument.parameters[f'impedance_{k}'].set('50') # 50 Ohm impedance
            self.instrument.parameters[f'coupling_{k}'].set('DC')  # DC Coupled
            self.instrument.parameters[f'full_scale_{k}'].set(3.0)  # 3.0 Volts

    def get_additional_pulses(self):
        if not self.pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        else:
            # Add a single trigger pulse when starting acquisition
            t_start = min(pulse.t_start for pulse in
                          self.pulse_sequence.get_pulses(acquire=True))
            if (self.input_pulse_sequence.get_pulses(trigger=True, t_start=t_start)):
                logger.warning('Trigger manually defined for M3300A digitizer. '
                               'This is inadvisable as this could limit the responsiveness'
                               ' of the machine.')
                return []

            return [TriggerPulse(t_start=t_start, duration=15e-6,
                                 connection_requirements={
                                     'input_instrument': self.instrument_name(),
                                     'trigger': True
                                 })]

    def setup(self, samples=1, **kwargs):
        self.samples(samples)

        # Find all unique pulse_connections to choose which channels to acquire on
        channel_selection = [int(ch_name[-1]) for ch_name in self.acquisition_channels()]
        self.channel_selection(sorted(channel_selection))
        self.acquisition_controller().channel_selection(self.channel_selection())

        # Pulse averaging is done in the interface, not the controller
        self.acquisition_controller().average_mode('none')

        if isinstance(self.acquisition_controller(), Triggered_Controller):
            if self.input_pulse_sequence.get_pulses(name='Bayes'):
                bayesian_pulse = self.input_pulse_sequence.get_pulses(name='Bayes')[0]
                for ch in range(8):
                    self.instrument.parameters[f'DAQ_trigger_delay_{ch}'].set(int(bayesian_pulse.t_start*2*self.sample_rate()))

            self.acquisition_controller().sample_rate(self.sample_rate())
            self.acquisition_controller().traces_per_acquisition(self.samples())

            # Setup triggering
            trigger_pulse = self.input_pulse_sequence.get_pulse(trigger=True)
            trigger_channel = trigger_pulse.connection.input['channel'].id
            self.acquisition_controller().trigger_channel(trigger_channel)
            self.acquisition_controller().trigger_threshold(trigger_pulse.amplitude / 2)
            self.acquisition_controller().trigger_edge('rising')

            # Capture maximum number of samples on all channels
            t_start = min(self.pulse_sequence.t_start_list)
            t_stop = max(self.pulse_sequence.t_stop_list)
            samples_per_trace = (t_stop - t_start) * self.sample_rate()
            self.acquisition_controller().samples_per_trace(samples_per_trace)

            # Set read timeout interval, which is the interval for requesting
            # an acquisition. This allows us to interrupt an acquisition prematurely.
            timeout_interval = max(2.1 * self.pulse_sequence.duration,
                                   self.minimum_timeout_interval())
            self.acquisition_controller().timeout_interval(timeout_interval)
            # An error is raised if no data has been acquired within 64 seconds.
            self.acquisition_controller().timeout(64.0)

            # Traces per read should not be longer than the timeout interval
            traces_per_read = max(timeout_interval // self.pulse_sequence.duration, 1)
            self.acquisition_controller().traces_per_read(traces_per_read)
        else:
            raise RuntimeError('No setup configured for acquisition controller '
                               f'{self.acquisition_controller()}')

    def stop(self):
        # Stop all DAQs
        self.instrument.daq_stop_multiple((1 << 8) - 1)


