from qcodes.instrument_drivers.picotech.picoscope_alternative import *

from silq.instrument_interfaces import InstrumentInterface
from typing import List, Union, Dict
from qcodes.instrument.parameter import Parameter, MultiParameter
import numpy as np
from silq.instrument_interfaces import Channel
from qcodes.utils import validators as vals
from silq.pulses import Pulse, TriggerPulse

class PicoScopeInterface(InstrumentInterface):
    def __init__(self,
                 instrument_name: str,
                 default_settings={},
                 **kwargs):
        super().__init__(instrument_name, **kwargs)

        # Define channels
        self._acquisition_channels = {
            'ch'+idx: Channel(instrument_name=self.instrument_name(),
                               name='ch'+idx,
                               id=idx, input=True)
            for idx in ['A', 'B', 'C', 'D']}
        self._channels = {
            **self._acquisition_channels,
            'external':  Channel(instrument_name=self.instrument_name(),
                                name='external', id='external', input_trigger=True),
            }

        self.pulse_implementations = []

        # Names of acquisition channels [chA, chB, etc.]
        self.add_parameter(name='acquisition_channels',
                           set_cmd=None,
                           initial_value=[],
                           vals=vals.Anything(),
                           docstring='Names of acquisition channels '
                                     '[chA, chB, etc.]. Set by the layout')

        self.samples = self.instrument.samples

        self.points_per_trace = self.instrument.points_per_trace

        self.sample_rate = self.instrument.sample_rate

        self.add_parameter(name='trigger_channel',
                           set_cmd=None,
                           initial_value='external',
                           vals=vals.Enum('external', 'disable',
                                          *self._acquisition_channels.keys()))

        self.trigger_slope = self.instrument.trigger_direction

        self.trigger_threshold = self.instrument.trigger_threshold

        self.trigger_in_duration = Parameter('trigger_in_duration',
                                             initial_value=1e-6,
                                             vals=vals.Numbers(),
                                             set_cmd=None,
                                             )

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

        self.traces = {}
        self.pulse_traces = {}

    def setup(self, samples: Union[int, None] = None, **kwargs):
        if samples is not None:
            self.samples(samples)

        # Select the channels to acquire
        for ch in self.instrument.channels:
            # Enable or disable channels
            ch.enabled(ch.id in self.acquisition_channels())

        # Pulse averaging is done in the interface, not the controller
        self.instrument.average_mode(None)

        # Capture maximum number of samples on all channels
        if not self.capture_full_trace():
            t_start = min(self.pulse_sequence.t_start_list)
            t_stop = max(self.pulse_sequence.t_stop_list)
        else:
            # Capture full trace, even if no pulses have acquire=True
            # Mainly done so that the full trace can be stored.
            t_start = 0
            t_stop = self.pulse_sequence.duration

        samples_per_trace = (t_stop - t_start) * self.sample_rate()
        self.points_per_trace(samples_per_trace)

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

        # Set up the acquisition buffers to the right sizes needed
        self.instrument.initialize_buffers()

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses, empty by default.
        """
        if self.input_pulse_sequence.get_pulses(trigger=True):
            logger.warning('Picoscope has a manual trigger pulse defined.'
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
                logger.error('Could not find trigger connection for picoscope')

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

    def segment_traces(self, traces: Dict[str, np.ndarray]):
        """ Segment traces by acquisition pulses.

        For each pulse with ``acquire`` set to True (which should be all pulses
        passed along to the PicoscopeInterface), the relevant portion of each channel
        trace is segmented and returned in a new dict

        Args:
            traces: ``{channel_id: channel_traces}`` dict

        Returns:
            Dict[str, Dict[str, np.ndarray]:
            Dict format is
            ``{pulse.full_name: {channel_id: pulse_channel_trace}}``.

        """
        pulse_traces = {}

        if self.capture_full_trace():
            t_start_initial = 0
        else:
            t_start_initial = min(p.t_start for p in
                                  self.pulse_sequence.get_pulses(acquire=True))
        for pulse in self.pulse_sequence.get_pulses(acquire=True):
            delta_t_start = pulse.t_start - t_start_initial
            start_idx = int(round(delta_t_start * self.sample_rate()))
            pts = int(round(pulse.duration * self.sample_rate()))

            pulse_traces[pulse.full_name] = {}
            for ch, trace in traces.items():
                pulse_trace = trace[:, start_idx:start_idx + pts]
                if pulse.average == 'point':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace)
                elif pulse.average == 'trace':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace, 0)
                elif 'point_segment' in pulse.average:
                    # Extract number of segments to split trace into
                    segments = int(pulse.average.split(':')[1])

                    # average over number of samples, returns 1D trace
                    mean_arr = np.mean(pulse_trace, axis=0)

                    # Split 1D trace into segments
                    segmented_array = np.array_split(mean_arr, segments)
                    pulse_traces[pulse.full_name][ch] = [np.mean(arr) for arr in segmented_array]
                elif 'trace_segment' in pulse.average:
                    segments = int(pulse.average.split(':')[1])

                    segments_idx = [int(round(pts * idx / segments))
                                    for idx in np.arange(segments + 1)]

                    pulse_traces[pulse.full_name][ch] = np.zeros(segments)
                    for k in range(segments):
                        pulse_traces[pulse.full_name][ch][k] = \
                            pulse_trace[:, segments_idx[k]:segments_idx[k + 1]]
                elif pulse.average == 'none':
                    pulse_traces[pulse.full_name][ch] = pulse_trace
                else:
                    raise SyntaxError(f'Unknown average mode {pulse.average}')
        return pulse_traces


    def acquisition(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Perform an acquisition.

        Should only be called after the interface has been setup and all other
        instruments have been started (via `Layout.start`).

        Returns:
            Acquisition traces that have been segmented for each pulse.
            Returned dictionary format is:
            ``{pulse.full_name: {channel_id: pulse_channel_trace}}``.

        """
        self.instrument.acquisition()
        self.traces = self.instrument.buffers
        self.pulse_traces = self.segment_traces(self.instrument.buffers)
        return self.pulse_traces

    def start(self):
        pass

    def stop(self):
        self.instrument.stop()
