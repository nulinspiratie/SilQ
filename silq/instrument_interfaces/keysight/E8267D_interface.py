from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection
from silq.pulses import DCPulse, DCRampPulse, SinePulse, FrequencyRampPulse, \
    TriggerPulse, MarkerPulse, PulseImplementation

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class E8267DInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        # Initialize channels
        self._input_channels = {
            'ext1': Channel(instrument_name=self.instrument_name(),
                            name='ext1', input=True),
            'ext2': Channel(instrument_name=self.instrument_name(),
                            name='ext2', input=True),
            'I': Channel(instrument_name=self.instrument_name(),
                            name='I', input=True),
            'Q': Channel(instrument_name=self.instrument_name(),
                            name='Q', input=True)
        }
        self._output_channels = {
            'RF_out': Channel(instrument_name=self.instrument_name(),
                              name='RF_out', output=True),
        }

        self._channels = {
            **self._input_channels,
            **self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input=True),
            'pattern_trig_in': Channel(instrument_name=self.instrument_name(),
                                       name='pattern_trig_in', input=True)
        }

        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[('frequency', {'min': 250e3, 'max': 44e9})]
            ),
            FrequencyRampPulseImplementation(
                pulse_requirements=[
                    ('frequency_start', {'min': 250e3, 'max': 44e9}),
                    ('frequency_stop', {'min': 250e3, 'max': 44e9})]
            )
        ]

        self.add_parameter('envelope_padding',
                           unit='ms',
                           parameter_class=ManualParameter,
                           initial_value=0)
        self.add_parameter('modulation_channel',
                           parameter_class=ManualParameter,
                           initial_value='ext1',
                           vals=vals.Enum(*self._input_channels))

    def get_final_additional_pulses(self, **kwargs):
        return []

    def setup(self, frequency=None, **kwargs):
        self.instrument.RF_output('off')
        self.instrument.phase_modulation('off')

        frequencies = {pulse.implement()['frequency']
                       for pulse in self.pulse_sequence}

        powers = {pulse.power for pulse in self.pulse_sequence}
        assert len(powers) == 1, "Cannot handle multiple pulse powers"

        frequency_deviations = {pulse.implement()['frequency_deviation']
                                for pulse in self.pulse_sequence
                                if 'frequency_deviation' in pulse.implement()}

        if len(frequencies) > 1 or frequency_deviations:
            # Need frequency modulation

            deviation = frequency_deviations[0]
            self.instrument.frequency_deviation(deviation)
            self.instrument.frequency_modulation('on')
            self.instrument.frequency_modulation_source(
                self.modulation_channel())
        else:
            self.instrument.frequency_deviation(0)
            self.instrument.frequency_modulation('off')

        frequency = frequencies[0]
        power = powers[0]
        self.instrument.frequency(frequency)
        self.instrument.power(power)

        self.instrument.pulse_modulation('on')
        self.instrument.pulse_modulation_source('ext')
        self.instrument.output_modulation('on')

    def start(self):
        self.instrument.RF_output('on')

    def stop(self):
        self.instrument.RF_output('off')



class SinePulseImplementation(PulseImplementation, SinePulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SinePulse, **kwargs)

    def target_pulse(self, pulse, interface, is_primary=False, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add an envelope pulse
        targeted_pulse.additional_pulses.append(
            MarkerPulse(t_start=pulse.t_start,
                        t_stop=pulse.t_stop,
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': 'trig_in'}
                        )
        )

        return targeted_pulse

    def implement(self):
        return {'frequency': self.frequency}


class FrequencyRampPulseImplementation(PulseImplementation, FrequencyRampPulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=FrequencyRampPulse,
                                     **kwargs)

    def target_pulse(self, pulse, interface, **kwargs):
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        assert targeted_pulse.power is not None, "Pulse must have power defined"

        if pulse.frequency_start < pulse.frequency_stop:
            amplitude_start, amplitude_stop = -1, 1
        else:
            amplitude_start, amplitude_stop = 1, -1

        # Determine the corresponding final amplitude from final frequency
        # Amplitude slope is dA/df
        amplitude_slope = \
            (amplitude_stop - amplitude_start) / \
            (targeted_pulse.frequency_stop - targeted_pulse.frequency_start)
        amplitude_final = \
            amplitude_start + amplitude_slope * \
            (targeted_pulse.frequency_final - targeted_pulse.frequency_start)


        # Add an envelope pulse with some padding on both sides.
        targeted_pulse.additional_pulses.extend((
            MarkerPulse(t_start=pulse.t_start - interface.envelope_padding()/2,
                        t_stop=pulse.t_stop + interface.envelope_padding()/2,
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': 'trig_in'}
                        ),
            # Add a ramping DC pulse for frequency modulation
            DCRampPulse(t_start=pulse.t_start,
                        t_stop=pulse.t_stop,
                        amplitude_start=amplitude_start,
                        amplitude_stop=amplitude_stop,
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': interface.modulation_channel()}
                        )
        ))

        if interface.envelope_padding() > 0:
            # Add padding DC pulses at start and end
            targeted_pulse.additional_pulses.extend((
                DCPulse(t_start=pulse.t_start - interface.envelope_padding(),
                        t_stop=pulse.t_start,
                        amplitude=amplitude_start,
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': interface.modulation_channel()}
                        ),
                DCPulse(t_start=pulse.t_stop,
                        t_stop=pulse.t_stop + interface.envelope_padding(),
                        amplitude=amplitude_final,
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': interface.modulation_channel()}
                        )
            ))

        if targeted_pulse.frequency_sideband is not None:
            targeted_pulse.additional_pulses.append(
                SinePulse(t_start=targeted_pulse.t_start,
                          t_stop=targeted_pulse.t_stop,
                          amplitude=1,
                          frequency=targeted_pulse.frequency_sideband,
                          connection_requirements={
                              'input_instrument': interface.instrument_name(),
                              'input_channel': ['I', 'Q']
                          })
            )

        return targeted_pulse

    def implement(self):
        return {'frequency': (self.frequency_start + self.frequency_stop) / 2,
                'frequency_deviation': abs(self.frequency_stop -
                                           self.frequency_start) / 2}