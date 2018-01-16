import numpy as np
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, SinePulse, MarkerPulse, \
    PulseImplementation
from silq.tools.general_tools import arreqclose_in_list

from qcodes import ManualParameter
from qcodes import validators as vals


logger = logging.getLogger(__name__)


class Keysight81180AInterface(InstrumentInterface):
    """

    Parameters that should be set in Instrument driver:
        output_coupling
        sample_rate
    """

    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, output=True)
            for k in [1, 2]}

        # TODO add marker outputs
        self._channels = {
            **self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True),
            'sync': Channel(instrument_name=self.instrument_name(),
                            name='sync', output=True)}

        self.pulse_implementations = []

        self.add_parameter('trigger_in_duration',
                           parameter_class=ManualParameter, unit='us',
                           initial_value=0.1)
        self.add_parameter('active_channels',
                           parameter_class=ManualParameter,
                           initial_value=[],
                           vals=vals.Lists(vals.Strings()))

    def get_additional_pulses(self, **kwargs):
        additional_pulses = []
        return additional_pulses

    def setup(self, **kwargs):
        self.active_channels(list({pulse.connection.output['channel'].name
                                   for pulse in self.pulse_sequence}))
        for ch in self.active_channels():
            instrument_channel = getattr(self.instrument, ch)

            instrument_channel.run_mode('sequenced')
            instrument_channel.sequence_mode('stepped')
            instrument_channel.trigger_source('external')
            instrument_channel.trigger_mode('override')

        self.generate_waveforms_sequences()

    def generate_waveforms_sequences(self):
        self.waveforms = {ch: [] for ch in self.active_channels()}
        self.sequences = {ch: [] for ch in self.active_channels()}

        for ch in self.active_channels():
            instrument_channel = getattr(self.instrument, ch)

            pulse_sequence = self.pulse_sequence(input_channel=ch)

            # Determine segments
            segments = [(pulse.t_start, pulse.t_stop)
                        for pulse in pulse_sequence(is_marker=False)]
            for marker_pulse in pulse_sequence(is_marker=True):
                t = marker_pulse.t_start
                while t < marker_pulse.t_stop:
                    for segment in segments:
                        if segment[0] <= t <= segment[1]:
                            t = segment[1]
                            break
                    else:
                        # No existing segment for this part of marker pulse
                        t_start_next = min(segment[0] for segment in segments
                                           if segment[0] > t)
                        t_next = min(t_start_next, marker_pulse.t_stop)
                        segments.append((t, t_next))
                        t = t_next


            # Add empty waveform, with minimum points (320)
            empty_segment = np.zeros(320)
            self.waveforms[ch].append(empty_segment)



            for pulse in pulse_sequence(is_marker=False):
                waveform = pulse.implement(
                    sample_rate=instrument_channel.sample_rate())

                # Check if waveform is already programmed
                waveform_idx = arreqclose_in_list(waveform, self.waveforms[ch])
                if waveform_idx is None:
                    # Add waveform to existing waveforms
                    self.waveforms[ch].append(waveform)
        pass

    def start(self):
        for ch_name in self.active_channels():
            instrument_channel = getattr(self.instrument, ch_name)
            instrument_channel.on()

    def stop(self):
        self.instrument.ch1.off()
        self.instrument.ch2.off()

    class SinePulseImplementation(PulseImplementation):
        pulse_class = SinePulse
        is_marker = False

        def implement(self):
            pass

    class MarkerPulseImplementation(PulseImplementation):
        pulse_class = MarkerPulse
        is_marker = True

        def implement(self):
            pass