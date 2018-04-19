from typing import List

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, DCPulse, DCRampPulse, SinePulse, \
    FrequencyRampPulse, MarkerPulse, PulseImplementation

from qcodes.utils import validators as vals

class E8267DInterface(InstrumentInterface):
    """ Interface for the Keysight E8267D
    
    When a `PulseSequence` is targeted in the `Layout`, the 
    pulses are directed to the appropriate interface. Each interface is
    responsible for translating all pulses directed to it into instrument 
    commands. During the actual measurement, the instrument's operations will
    correspond to that required by the pulse sequence.
    
    Args:
        instrument_name: name of instrument for which this is an interface
        
    Note:    
        For a given instrument, its associated interface can be found using
            `get_instrument_interface` 
    """
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
                           unit='s',
                           set_cmd=None,
                           initial_value=0)
        self.add_parameter('modulation_channel',
                           set_cmd=None,
                           initial_value='ext1',
                           vals=vals.Enum(*self._input_channels))

        self.add_parameter('fix_frequency',
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('fix_frequency_deviation',
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('frequency',
                           unit='Hz',
                           set_cmd=None,
                           initial_value=None)
        self.add_parameter('frequency_deviation',
                           unit='Hz',
                           set_cmd=None,
                           initial_value=None)

    def get_additional_pulses(self) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses
        
        Returns:
            List of additional pulses, such as IQ modulation pulses
        """
        if not self.pulse_sequence:
            return []

        frequencies = list({pulse.frequency for pulse in self.pulse_sequence})

        frequency_deviations = {pulse.frequency_deviation
                                for pulse in self.pulse_sequence
                                if hasattr(pulse, 'frequency_deviation')}

        if len(frequencies) > 1 or frequency_deviations:
            # Find minimum and maximum frequency
            min_frequency = min(pulse.frequency - pulse.frequency_deviation
                                if hasattr(pulse, 'frequency_deviation')
                                else pulse.frequency
                                for pulse in self.pulse_sequence)
            max_frequency = max(pulse.frequency + pulse.frequency_deviation
                                if hasattr(pulse, 'frequency_deviation')
                                else pulse.frequency
                                for pulse in self.pulse_sequence)

            if not self.fix_frequency():
                self.frequency((min_frequency + max_frequency) / 2)

            if not self.fix_frequency_deviation():
                self.frequency_deviation(max(max_frequency - self.frequency(),
                                             self.frequency() - min_frequency))
        else:
            self.frequency(frequencies[0])
            self.frequency_deviation(0)

        additional_pulses = []
        for pulse in self.pulse_sequence:
            additional_pulses += pulse.implementation.get_additional_pulses(
                interface=self, frequency=self.frequency(),
                frequency_deviation=self.frequency_deviation())

        return additional_pulses

    def setup(self, **kwargs):
        """Set up instrument after layout has been targeted by pulse sequence.
        
        Args:
            **kwargs: Unused setup kwargs provided from Layout 
        """
        self.instrument.RF_output('off')
        self.instrument.phase_modulation('off')

        powers = list({pulse.power for pulse in self.pulse_sequence})
        assert len(powers) == 1, "Cannot handle multiple pulse powers"

        self.instrument.frequency(self.frequency())
        self.instrument.power(powers[0])

        self.instrument.frequency_deviation(self.frequency_deviation())
        if self.frequency_deviation() > 0:
            self.instrument.frequency_modulation('on')
        else:
            self.instrument.frequency_modulation('off')

        self.instrument.pulse_modulation('on')
        self.instrument.pulse_modulation_source('ext')
        self.instrument.output_modulation('on')

    def start(self):
        """Start instrument"""
        self.instrument.RF_output('on')

    def stop(self):
        """Stop instrument"""
        self.instrument.RF_output('off')


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def target_pulse(self, pulse, interface, **kwargs):
        assert pulse.power is not None, "Pulse must have power defined"
        return super().target_pulse(pulse, interface, **kwargs)

    def get_additional_pulses(self, interface, frequency, frequency_deviation):
        # Add an envelope pulse
        additional_pulses = [
            MarkerPulse(t_start=self.pulse.t_start, t_stop=self.pulse.t_stop,
                        amplitude=3,
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': 'trig_in'})]

        if frequency_deviation > 0:
            amplitude = (self.pulse.frequency - frequency) / frequency_deviation
            assert abs(amplitude) < 1 + 1e-13, \
                f'amplitude {amplitude} cannot be higher than 1'

            additional_pulses.append(
                DCPulse(
                    t_start=self.pulse.t_start - interface.envelope_padding(),
                    t_stop=self.pulse.t_start + interface.envelope_padding(),
                    amplitude=amplitude,
                    connection_requirements={
                        'input_instrument': interface.instrument_name(),
                        'input_channel': interface.modulation_channel()}))
        return additional_pulses

class FrequencyRampPulseImplementation(PulseImplementation):
    pulse_class = FrequencyRampPulse

    def target_pulse(self, pulse, interface, **kwargs):
        assert pulse.power is not None, "Pulse must have power defined"
        assert pulse.frequency_start != pulse.frequency_stop, \
            f"Pulse frequency_start must differ from frequency_stop {pulse}"
        return super().target_pulse(pulse, interface, **kwargs)

    def get_additional_pulses(self, interface, frequency, frequency_deviation):
        # TODO (additional_pulses) correct amplitudes

        amplitude_start = (self.pulse.frequency_start - frequency) / \
                          frequency_deviation
        amplitude_stop = (self.pulse.frequency_stop - frequency) / \
                          frequency_deviation

        # Determine the corresponding final amplitude from final frequency
        # Amplitude slope is dA/df
        frequency_difference = self.pulse.frequency_stop - self.pulse.frequency_start
        amplitude_slope = (amplitude_stop - amplitude_start) / frequency_difference
        amplitude_final = amplitude_start + amplitude_slope * frequency_difference


        # Add an envelope pulse with some padding on both sides.
        additional_pulses = [
            MarkerPulse(
                t_start=self.pulse.t_start - interface.envelope_padding()/2,
                t_stop=self.pulse.t_stop + interface.envelope_padding()/2,
                connection_requirements={
                    'input_instrument': interface.instrument_name(),
                    'input_channel': 'trig_in'}
            ),
            # Add a ramping DC pulse for frequency modulation
            DCRampPulse(
                t_start=self.pulse.t_start,
                t_stop=self.pulse.t_stop,
                amplitude_start=amplitude_start,
                amplitude_stop=amplitude_stop,
                connection_requirements={
                    'input_instrument': interface.instrument_name(),
                    'input_channel': interface.modulation_channel()}
            )
        ]

        if interface.envelope_padding() > 0:
            # Add padding DC pulses at start and end
            additional_pulses.extend((
                DCPulse(
                    t_start=self.pulse.t_start - interface.envelope_padding(),
                    t_stop=self.pulse.t_start,
                    amplitude=amplitude_start,
                    connection_requirements={
                        'input_instrument': interface.instrument_name(),
                        'input_channel': interface.modulation_channel()}
                ),
                DCPulse(
                    t_start=self.pulse.t_stop,
                    t_stop=self.pulse.t_stop+interface.envelope_padding(),
                    amplitude=amplitude_final,
                    connection_requirements={
                        'input_instrument': interface.instrument_name(),
                        'input_channel': interface.modulation_channel()}
                )
            ))

        if self.pulse.frequency_sideband is not None:
            additional_pulses.append(
                SinePulse(
                    t_start=self.pulse.t_start,
                    t_stop=self.pulse.t_stop,
                    amplitude=1,
                    frequency=self.pulse.frequency_sideband,
                    connection_requirements={
                        'input_instrument': interface.instrument_name(),
                        'input_channel': ['I', 'Q']
                    })
            )

        return additional_pulses