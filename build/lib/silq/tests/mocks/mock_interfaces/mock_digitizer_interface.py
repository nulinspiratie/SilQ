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
            for k in [1,2]
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
                           initial_value=1,
                           docstring='Number of times to acquire the pulse '
                                     'sequence.')

        # Noise factor used when generating traces
        self.noise_factor = 0.6


        self.blip_probability = 0.45  # Probability for fake blips during read pulse
        self.blip_start = 0.25  # Maximum fraction for blip to start
        self.blip_duration = 0.25  # Maximum duration for a blip
        self.blip_read_amplitude = 0.3  # Maximum read amplitude

        self.pulse_traces = {}

    def setup(self, samples, **kwargs):
        self.samples(samples)

    def _generate_traces(self):
        """Generate fake traces with noise from self.noise_factor

        This data is usually acquired from the actual acquisition instrument
        """
        total_points = int(self.sample_rate() * self.pulse_sequence.duration)
        traces = np.random.rand(len(self.acquisition_channels()),
                                self.samples(),
                                total_points) - 0.5
        traces *= self.noise_factor

        for pulse in self.pulse_sequence:
            idx_start = int(pulse.t_start * self.sample_rate())
            idx_stop = int(pulse.t_stop * self.sample_rate())
            traces[:,:,idx_start:idx_stop] += max(-pulse.amplitude, 0)

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
            for channel, channel_traces in zip(self.acquisition_channels(),
                                               traces):
                idx_start = int(pulse.t_start * self.sample_rate())
                idx_stop = int(pulse.t_stop * self.sample_rate())
                pulse_trace = channel_traces[:,idx_start:idx_stop]

                # Perform averaging if defined in pulse.average
                if pulse.average == 'point':
                    pulse_trace = np.mean(pulse_trace)
                elif pulse.average == 'trace':
                    pulse_trace = np.mean(pulse_trace, axis=0)

                pulse_traces[pulse.full_name][channel] = pulse_trace

        return pulse_traces



    def acquisition(self):
        # Generate fake traces
        self.traces = self._generate_traces()

        # Segment traces per pulse
        self.pulse_traces = self.segment_traces(self.traces)
        return self.pulse_traces

    def start(self):
        pass

    def stop(self):
        pass