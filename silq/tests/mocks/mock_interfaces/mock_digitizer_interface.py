import numpy as np

from qcodes.utils import validators as vals
from .mock_interface import MockInterface
from silq.instrument_interfaces import Channel


class MockDigitizerInterface(MockInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        # Override untargeted pulse adding (measurement pulses can be added)
        self.pulse_sequence.allow_untargeted_pulses = True

        # Define instrument channels
        # - Two input channels (ch1 and ch2)
        # - One input trigger channel (trig_in)
        self._acquisition_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, input=True)
            for k in range(1,11)
        }
        self._channels = {
            **self._acquisition_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True)
        }

        self.add_parameter(name='acquisition_channels',
                           set_cmd=None,
                           initial_value=[],
                           vals=vals.Anything(),
                           docstring='Names of acquisition channels '
                                     '[chA, chB, etc.]. Set by the layout')

        self.add_parameter(name='samples',
                           set_cmd=None,
                           initial_value=1,
                           docstring='Number of times to acquire the pulse '
                                     'sequence.')

        self.add_parameter(name='sample_rate',
                           set_cmd=None,
                           initial_value=int(50e3),
                           docstring='Number of samples captured per second.')

        self.add_parameter('points_per_trace',
                           get_cmd=lambda: int(self.sample_rate() * self.pulse_sequence.duration),
                           docstring='Number of points in a single trace.')

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
        # Noise factor used when generating traces
        self.noise_factor = 0.6


        self.blip_probability = 0.45  # Probability for fake blips during read pulse
        self.blip_start = 0.25  # Maximum fraction for blip to start
        self.blip_duration = 0.25  # Maximum duration for a blip
        self.blip_read_amplitude = 0.3  # Maximum read amplitude

        self.pulse_traces = {}

        # dict of raw unsegmented traces {ch_name: ch_traces}
        self.traces = {}
        # Segmented traces per pulse, {pulse_name: {channel_name: {
        # ch_pulse_traces}}
        self.pulse_traces = {}

    def setup(self, samples, **kwargs):
        self.samples(samples)

    def _generate_traces(self):
        """Generate fake traces with noise from self.noise_factor

        This data is usually acquired from the actual acquisition instrument
        """
        total_points = self.points_per_trace()
        traces = np.random.rand(len(self.acquisition_channels()),
                                self.samples(),
                                total_points) - 0.5
        traces *= self.noise_factor

        for pulse in self.pulse_sequence:
            idx_start = int(round(pulse.t_start * self.sample_rate()))
            idx_stop = int(round(pulse.t_stop * self.sample_rate()))
            traces[:, :, idx_start:idx_stop] += pulse.get_voltage(
                np.linspace(pulse.t_start, pulse.t_stop, idx_stop-idx_start))
            # print(idx_stop - idx_start)
            # Add fake blips to read pulse
            for trace in traces[0]:
                if pulse.name.startswith('read'):
                    if np.random.rand() > self.blip_probability:
                        # Probabilistically skip this trace
                        continue
                    elif np.random.rand() < pulse.amplitude / self.blip_read_amplitude:
                        continue
                    blip_start = np.random.randint(0, (idx_stop-idx_start)*self.blip_start)
                    blip_start += idx_start
                    blip_duration = np.random.randint(0, (idx_stop-idx_start)*self.blip_duration)
                    trace[blip_start:blip_start+blip_duration] += 1

        return traces

    def segment_traces(self, traces):
        """Segment traces per pulse, and optionally perform averaging"""
        pulse_traces = {}
        for pulse in self.pulse_sequence:
            pulse_traces[pulse.full_name] = {}
            for ch, channel_traces in zip(self.acquisition_channels(), traces):
                idx_start = int(round(pulse.t_start * self.sample_rate()))
                idx_stop = int(round(pulse.t_stop * self.sample_rate()))
                pulse_trace = channel_traces[:,idx_start:idx_stop]

                pts = int(round(pulse.duration * self.sample_rate()))

                # Perform averaging if defined in pulse.average
                if pulse.average == 'point':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace)
                elif pulse.average == 'trace':
                    pulse_traces[pulse.full_name][ch] = np.mean(pulse_trace, 0)
                elif 'point_segment' in pulse.average:
                    # import pdb; pdb.set_trace()
                    # Extract number of segments to split trace into
                    segments = int(pulse.average.split(':')[1])

                    # average over number of samples, returns 1D trace
                    mean_arr = np.mean(pulse_trace, axis=0)

                    # Split 1D trace into segments
                    segmented_array = np.array_split(mean_arr, segments)
                    pulse_traces[pulse.full_name][ch] = [np.mean(arr) for arr in
                                                         segmented_array]
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



    def acquisition(self):
        # Generate fake traces
        self.traces = self._generate_traces()

        # Segment traces per pulse
        self.pulse_traces = self.segment_traces(self.traces)
        self.traces = {ch: ch_traces for ch, ch_traces
                  in zip(self.acquisition_channels(), self.traces)}
        return self.pulse_traces

    def start(self):
        pass

    def stop(self):
        pass