import numpy as np
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, TriggerPulse, SinePulse, MarkerPulse, \
    PulseImplementation
from silq.pulses import PulseSequence
from silq.tools.general_tools import arreqclose_in_list
from silq.tools.pulse_tools import pulse_to_waveform_sequence


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

        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[('frequency', {'min': 0, 'max': 1e9}),
                                    ('amplitude', {'max': 0.5}),
                                    ('duration', {'min': 100e-9})]
            )
        ]

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
            instrument_channel.sequence_mode('automatic')
            # TODO: Trigger source needs to come from configuration
            instrument_channel.trigger_source('bus')
            instrument_channel.trigger_mode('override')

        self.generate_waveform_sequences()

    def generate_waveform_sequences(self):
        self.waveforms = {ch: [] for ch in self.active_channels()}
        self.sequences = {ch: [] for ch in self.active_channels()}

        for ch in self.active_channels():
            instrument_channel = getattr(self.instrument, ch)
            pulse_sequence = PulseSequence(self.pulse_sequence.get_pulses(output_channel=ch))

            # Determine segments
            # segments = [(pulse.t_start, pulse.t_stop)
            #             for pulse in pulse_sequence.get_pulses(is_marker=False)]
            # for marker_pulse in pulse_sequence.get_pulses(is_marker=True):
            #     t = marker_pulse.t_start
            #     while t < marker_pulse.t_stop:
            #         for segment_t_start, segment_t_stop in segments:
            #             if segment_t_start <= t <= segment_t_stop:
            #                 t = segment_t_stop
            #                 break
            #         else:
            #             # No existing segment for this part of marker pulse
            #             t_start_next = min(segment_t_start
            #                                for segment_t_start,_ in segments
            #                                if segment_t_start > t)
            #             t_next = min(t_start_next, marker_pulse.t_stop)
            #             segments.append((t, t_next))
            #             t = t_next


            # Add empty waveform, with minimum points (320)
            empty_segment = np.zeros(320)
            empty_idx = 1
            # Load blank waveform into memory
            instrument_channel.add_waveform(empty_segment, empty_idx)
            self.waveforms[ch].append(empty_segment)


            # Begin waveform sequencing from 2, after empty_segment loaded
            segment_idx = 2
            for pulse in pulse_sequence.get_pulses(is_marker=False):
                waveform_results = pulse.implementation.implement(
                    sample_rate=instrument_channel.sample_rate(),
                    max_points=self.instrument.waveform_max_length)

                waveform = waveform_results['waveform']
                loops = waveform_results['loops']
                waveform_tail = waveform_results['waveform_tail']

                # Check if waveform is already programmed
                waveform_idx = arreqclose_in_list(waveform, self.waveforms[ch])

                # Add blank pulse to wait for pre-trigger
                self.sequences[ch].append((empty_idx, 1, 1))

                if waveform_idx is None:
                    # Add waveform to existing waveforms
                    self.waveforms[ch].append(waveform)
                    # Load waveform into memory
                    instrument_channel.add_waveform(waveform, segment_idx)

                    sequence_step = (segment_idx, loops, 0)
                    segment_idx += 1
                else:
                    # Re-use previous segment, shift to 1-indexing
                    waveform_idx += 1
                    sequence_step = (waveform_idx, loops, 0)

                self.sequences[ch].append(sequence_step)

                # Add tail if one exists
                if waveform_tail is not None:
                    # Check if waveform_tail is already programmed
                    waveform_tail_idx = arreqclose_in_list(waveform_tail,
                                                           self.waveforms[ch])

                    if waveform_tail_idx is None:
                        # Add waveform to existing waveforms
                        self.waveforms[ch].append(waveform_tail)
                        # Load waveform into memory
                        instrument_channel.add_waveform(waveform_tail, segment_idx)
                        # Add segment to sequence
                        sequence_step = (segment_idx, 1, 0)

                        segment_idx += 1
                    else:
                        # Re-use previous segment, shift to 1-indexing
                        waveform_tail_idx += 1
                        sequence_step = (waveform_tail_idx, 1, 0)

                    self.sequences[ch].append(sequence_step)

            if len(self.sequences[ch]) < 3:
                # Add an extra blank segment which will automatically run to
                # the next segment (~ 70 ns offset)
                self.sequences[ch].append((empty_idx, 1, 0))

            # Sequence all loaded waveforms
            instrument_channel.set_sequence(self.sequences[ch])

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

    def implement(self, sample_rate, max_points):
        # NOTE: Output will hold at the DC value of the first sample
        #       if waveform is waiting for a trigger at the end.

        duration = self.pulse.t_stop - self.pulse.t_start

        # TODO: Is this the best way to choose samples in time?
        total_t_list = np.linspace(self.pulse.t_start,
                                   self.pulse.t_stop,
                                   num= 32 * np.floor(duration * sample_rate / 32))

        results = pulse_to_waveform_sequence(max_points=max_points,
                                   frequency= self.pulse.frequency,
                                   sampling_rate = sample_rate,
                                   frequency_threshold = 1,
                                   total_duration = duration,
                                   min_points=320,
                                   sample_points_multiple = 32)

        optimum = results['optimum']
        waveform_pts   = optimum['points']
        # TODO: Sometimes repetitions returns 0, but it should play at least once
        #       should I just add 1?
        waveform_loops = max(optimum['repetitions'], 1)

        # Get waveform points for repeated segment
        t_list = total_t_list[:waveform_pts]
        waveform = self.pulse.get_voltage(t_list)
        # print(len(total_t_list), len(t_list), len(np.setdiff1d(total_t_list, t_list)))

        # The remaining waveform to be played is drawn from the remaining time points of
        tail_t_list = np.setdiff1d(total_t_list, t_list)
        if len(tail_t_list) != 0:
            waveform_tail = self.pulse.get_voltage(np.setdiff1d(total_t_list, t_list))
        else:
            waveform_tail = None

        return {'waveform': waveform, 'loops': waveform_loops,
                'waveform_tail': waveform_tail}

class MarkerPulseImplementation(PulseImplementation):
        pulse_class = MarkerPulse
        is_marker = True

        def implement(self, sample_rate, max_points):
            # TODO: Should we force a final sample to 0? How does the output work.
            t_list = np.arange(self.pulse.t_start,
                               self.pulse.t_stop,
                               1 / sample_rate)
            waveform = self.pulse.get_voltage(t_list)
            return {'waveform': waveform, 'loops': n_loops,
                    'waveform_tail' : []}