from typing import List
import numpy as np

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses.pulse_types import Pulse, TriggerPulse

from qcodes.utils import validators as vals
from qcodes.instrument_drivers.Keysight.SD_common.SD_acquisition_controller \
    import Triggered_Controller

import logging
logger = logging.getLogger(__name__)


class Keysight_SD_DIG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
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

        self.add_parameter(name='acquisition_controller',
                           set_cmd=None,
                           snapshot_value=False,
                           docstring='Acquisition controller for acquiring '
                                     'data with SD digitizer. '
                                     'Must be acquisition controller object.')

        self.add_parameter(name='acquisition_channels',
                           initial_value=[],
                           set_cmd=None,
                           vals=vals.Lists(),
                           docstring='Names of acquisition channels '
                                     '[chA, chB, etc.]. Set by the layout')

        self.add_parameter('sample_rate',
                           vals=vals.Numbers(),
                           set_cmd=None,
                           docstring='Acquisition sampling rate (Hz)')

        self.add_parameter('samples',
                           vals=vals.Numbers(),
                           set_cmd=None,
                           docstring='Number of times to acquire the pulse '
                                     'sequence.')

        self.add_parameter('points_per_trace',
                           get_cmd=lambda: self.acquisition_controller().samples_per_trace(),
                           docstring='Number of points in a trace.')

        self.add_parameter('channel_selection',
                           set_cmd=None,
                           docstring='Active channel indices to acquire. '
                                     'Zero-based index (chA -> 0, etc.). '
                                     'Set during setup and should not be set'
                                     'manually.')

        self.add_parameter('minimum_timeout_interval',
                           unit='s',
                           vals=vals.Numbers(),
                           initial_value=5,
                           set_cmd=None,
                           docstring='Minimum value for timeout when acquiring '
                                     'data. If 2.1 * pulse_sequence.duration '
                                     'is lower than this value, it will be '
                                     'used instead')

        self.add_parameter('trigger_in_duration',
                           unit='s',
                           vals=vals.Numbers(),
                           initial_value=15e-6,
                           set_cmd=None,
                           docstring="Duration for a receiving trigger signal. "
                                     "This is passed the the interface that is "
                                     "sending the triggers to this instrument.")

        self.add_parameter('capture_full_trace',
                           initial_value=False,
                           vals=vals.Bool(),
                           set_cmd=None,
                           docstring='Capture from t=0 to end of pulse '
                                     'sequence. False by default, in which '
                                     'case start and stop times correspond to '
                                     'min(t_start) and max(t_stop) of all '
                                     'pulses with the flag acquire=True, '
                                     'respectively.')
        # dict of raw unsegmented traces {ch_name: ch_traces}
        self.traces = {}
        # Segmented traces per pulse, {pulse_name: {channel_name: {ch_pulse_traces}}
        self.pulse_traces = {}

        # Set up the driver to a known default state
        self.initialize_driver()

    def acquisition(self):
        """Perform acquisition"""
        traces = self.acquisition_controller().acquisition()
        self.stop()
        self.traces = {ch: ch_traces for ch, ch_traces
                       in zip(self.acquisition_channels(), traces)}

        # The start of acquisition
        if self.capture_full_trace():
            t0 = 0
        else:
            t0 = min(pulse.t_start for pulse in self.pulse_sequence.get_pulses(acquire=True))

        # Split data into pulse traces
        pulse_traces = {}
        for pulse in self.pulse_sequence.get_pulses(acquire=True):
            name = pulse.full_name
            pulse_traces[name] = {}
            for k, ch_name in enumerate(self.acquisition_channels()):
                ch_data = self.traces[ch_name]
                start_idx = int(round((pulse.t_start - t0) * self.sample_rate()))
                stop_idx = start_idx + int(round(pulse.duration * self.sample_rate()))

                # Extract pulse data from the channel data
                pulse_traces[name][ch_name] = ch_data[:, start_idx:stop_idx]
                # Further average the pulse data
                if pulse.average == 'none':
                    pass
                elif pulse.average == 'trace':
                    pulse_traces[name][ch_name] = np.mean(pulse_traces[name][ch_name], axis=0)
                elif pulse.average == 'point':
                    pulse_traces[name][ch_name] = np.mean(pulse_traces[name][ch_name])
                elif 'point_segment' in pulse.average:
                    segments = int(pulse.average.split(':')[1])
                    split_arrs = np.array_split(pulse_traces[name][ch_name], segments, axis=1)
                    pulse_traces[name][ch_name] = np.mean(split_arrs, axis=(1,2))
                else:
                    raise SyntaxError(f'average mode {pulse.average} not configured')

        self.pulse_traces = pulse_traces
        return pulse_traces

    def initialize_driver(self):
        """
            Puts driver into a known initial state. Further configuration will
            be done in the configure_driver and get_additional_pulses
            functions.
        """
        self.instrument.channels.impedance('50') # 50 Ohm impedance
        self.instrument.channels.coupling('DC')  # DC Coupled
        self.instrument.channels.full_scale(3.0)  # 3.0 Volts

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses, empty by default.
        """
        if self.input_pulse_sequence.get_pulses(trigger=True):
            logger.warning('SD digitizer has a manual trigger pulse defined.'
                           'This should normally be done automatically instead.')
            return []
        elif not self.pulse_sequence.get_pulses(acquire=True):
            # No pulses need to be acquired
            return []
        else:
            # A trigger pulse is needed
            try:
                trigger_connection = next(connection for connection in connections
                                          if connection.trigger or connection.trigger_start)
            except StopIteration:
                logger.error('Could not find trigger connection for SD digitizer')

            connection_requirements = {'input_instrument': self.instrument_name()}

            if trigger_connection.trigger:
                connection_requirements['trigger'] = True
                t_start = min(pulse.t_start for pulse in
                              self.pulse_sequence.get_pulses(acquire=True))
            else: # connection.trigger_start or capture full trace
                connection_requirements['trigger_start'] = True
                t_start = 0

            if self.capture_full_trace():  # Override t_start to capture full trace
                t_start = 0

            return [TriggerPulse(t_start=t_start, duration=self.trigger_in_duration(),
                                 connection_requirements=connection_requirements)]

    def requires_setup(self, **kwargs) -> bool:
        requires_setup = super().requires_setup(**kwargs)
        if kwargs['samples'] != self.samples():
            requires_setup = True
        return requires_setup

    def setup(self, samples=1, input_connections=(), **kwargs):
        self.samples(samples)

        # Select the channels to acquire
        channel_selection = [int(ch_name[-1]) for ch_name in self.acquisition_channels()]
        self.channel_selection(channel_selection)
        self.acquisition_controller().channel_selection(channel_selection)

        # Pulse averaging is done in the interface, not the controller
        self.acquisition_controller().average_mode('none')

        if isinstance(self.acquisition_controller(), Triggered_Controller):
            self.acquisition_controller().sample_rate(self.sample_rate())
            self.acquisition_controller().traces_per_acquisition(self.samples())

            # Capture maximum number of samples on all channels
            if not self.capture_full_trace():
                t_start = min(self.pulse_sequence.t_start_list)
                t_stop = max(self.pulse_sequence.t_stop_list)
            else:
                # Capture full trace, even if no pulses have acquire=True
                # Mainly done so that the full trace can be stored.
                t_start = 0
                t_stop = self.pulse_sequence.duration

            #  !!! Changed np.ceil to np.round !!!
            samples_per_trace = int(np.round((t_stop - t_start) * self.sample_rate()))
            samples_per_trace += samples_per_trace % 2
            self.acquisition_controller().samples_per_trace(samples_per_trace)

            # Set read timeout interval
            # This is the interval for requesting an acquisition.
            # This allows us to interrupt an acquisition prematurely.
            timeout_interval = max(2.1 * self.pulse_sequence.duration,
                                   self.minimum_timeout_interval())
            self.acquisition_controller().timeout_interval(timeout_interval)
            # An error is raised if no data has been acquired within 64 seconds.
            self.acquisition_controller().timeout(64.0)

            # Traces per read should not be longer than the timeout interval
            traces_per_read = max(timeout_interval // self.pulse_sequence.duration, 1)
            self.acquisition_controller().traces_per_read(traces_per_read)

            self.setup_trigger(t_start, input_connections)
        else:
            raise RuntimeError('No setup configured for acquisition controller '
                               f'{self.acquisition_controller()}')

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def setup_trigger(self, t_start, input_connections):
        # Setup triggering
        trigger_pulse = self.input_pulse_sequence.get_pulse(trigger=True)
        if trigger_pulse is None:
            trigger_pulse = self.input_pulse_sequence.get_pulse(trigger_start=True)
        assert trigger_pulse is not None, "No trigger pulse found for digitizer"
        trigger_channel = trigger_pulse.connection.input['channel'].name
        # Also sets trigger mode, etc.
        self.acquisition_controller().trigger_channel(trigger_channel)
        if trigger_channel.startswith('ch'):
            self.acquisition_controller().analog_trigger_edge('rising')
            self.acquisition_controller().analog_trigger_threshold(
                trigger_pulse.amplitude / 2)

        elif trigger_channel == 'trig_in':
            self.acquisition_controller().digital_trigger_mode('rising')
        else:  # PXI channel
            self.acquisition_controller().digital_trigger_mode('rising')

        if self.input_pulse_sequence.get_pulses(name='Bayes'):
            bayesian_pulse = self.input_pulse_sequence.get_pulse(name='Bayes')
            trigger_delay = int(bayesian_pulse.t_start * 2 * self.sample_rate())
            trigger_delay_samples = int(round(trigger_delay * self.sample_rate()))
        elif any(connection.trigger_start for connection in input_connections):
            trigger_delay_samples = int(round(t_start * self.sample_rate()))
        else:
            trigger_delay_samples = 0
        self.acquisition_controller().trigger_delay_samples(trigger_delay_samples)

    def stop(self):
        # Stop all DAQs
        self.instrument.stop_channels(range(8))


