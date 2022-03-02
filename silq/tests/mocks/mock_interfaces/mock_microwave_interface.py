from .mock_interface import MockInterface
from silq.instrument_interfaces import Channel
from silq.pulses import SinePulse, PulseImplementation, FrequencyRampPulse, DCPulse
from qcodes import Parameter
import numpy as np


class MockMicrowaveInterface(MockInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        # Define instrument channels
        # - Two output channels (ch1 and ch2)
        # - One input trigger channel (trig_in)
        self._output_channels = {
            f'RF_out': Channel(instrument_name=self.instrument_name(),
                              name='RF_out', id=0, output=True)
        }
        self._input_channels = {
            label: Channel(instrument_name=self.instrument_name(),
                              name=label, id=k, input=True)
            for k, label in enumerate(['I', 'Q', 'pulse_mod'])
        }
        self._channels = {
            **self._output_channels,
            **self._input_channels,
        }

        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[
                    ('frequency', {'max': 500e6}),
                    ('power', {'min': -150, 'max': 25}),
                ]),

            FrequencyRampPulseImplementation(
                pulse_requirements=[
                    ('frequency_start', {'min': 20e9,'max': 44e9}),
                    ('frequency_stop', {'min': 20e9, 'max': 44e9}),
                    ('power', {'min': -150,'max': 25})
                ]),
        ]

        self.carrier_frequency = Parameter('carrier', unit='Hz', set_cmd=None)

    def get_additional_pulses(self, connections):
        """Additional pulses needed by instrument after targeting of main pulses

                Args:
                    connections: List of all connections in the layout

                Returns:
                    List of additional pulses, such as IQ modulation pulses
                """
        if not self.pulse_sequence:
            return []

        frequency_settings = self.determine_instrument_settings()

        additional_pulses = []
        phases = []
        frequencies = []
        for pulse in self.pulse_sequence:
            if isinstance(pulse, SinePulse):
                frequencies.append(pulse.frequency)
                phases.append(pulse.phase)
            elif isinstance(pulse, FrequencyRampPulse):
                frequencies.append(pulse.frequency_start)
                frequencies.append(pulse.frequency_stop)
                phases.append(pulse.phase)

        phases = set(phases)
        frequencies = set(frequencies)
        if len(phases) > 1 or len(frequencies) > 1:
            f0 = np.mean(frequencies)
            self.carrier_frequency(f0)
            for pulse in self.pulse_sequence:
                additional_pulses.extend(
                    [SinePulse('I_sideband', frequency=f0 - pulse.frequency,
                               duration=pulse.duration, t_start=pulse.t_start,
                               phase=pulse.phase(),
                               connection_requirements={
                                   'input_instrument': self.instrument_name(),
                                   'input_channel': 'I'}),
                     SinePulse('Q_sideband', frequency=f0 - pulse.frequency,
                               duration=pulse.duration, t_start=pulse.t_start,
                               phase=pulse.phase()-90,
                               connection_requirements={
                                   'input_instrument': self.instrument_name(),
                                   'input_channel': 'I'})
                     ]
                )
        else:
            self.carrier_frequency(next(frequencies))


        # # Ensure marker pulses are not overlapping
        # marker_pulses = [p for p in additional_pulses if
        #                  isinstance(p, MarkerPulse)]
        # if marker_pulses:
        #     marker_pulses = sorted(marker_pulses, key=lambda p: p.t_start)
        #     current_pulse = marker_pulses[0]
        #     merged_marker_pulses = [current_pulse]
        #     for pulse in marker_pulses[1:]:
        #         if pulse.t_start <= current_pulse.t_stop:
        #             # Marker pulse overlaps with previous pulse
        #             current_pulse.t_stop = max(pulse.t_stop,
        #                                        current_pulse.t_stop)
        #         else:
        #             # Pulse starts after previous pulse
        #             merged_marker_pulses.append(pulse)
        #             current_pulse = pulse
        #     nonmarker_pulses = [p for p in additional_pulses if
        #                         not isinstance(p, MarkerPulse)]
        #     additional_pulses = [*merged_marker_pulses, *nonmarker_pulses]

        return additional_pulses


    def setup(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

# Define pulse implementations

class DCPulseImplementation(PulseImplementation):
    pulse_class = DCPulse


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

class FrequencyRampPulseImplementation(PulseImplementation):
    pulse_class = FrequencyRampPulse
