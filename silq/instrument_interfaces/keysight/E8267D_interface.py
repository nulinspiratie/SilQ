from typing import List
from time import sleep
import numpy as np

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, DCPulse, DCRampPulse, SinePulse, \
    MultiSinePulse, FrequencyRampPulse, MarkerPulse, PulseImplementation

from qcodes.utils import validators as vals
from qcodes.instrument.parameter import Parameter



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
            MultiSinePulseImplementation(),
            FrequencyRampPulseImplementation(
                pulse_requirements=[
                    ('frequency_start', {'min': 250e3, 'max': 44e9}),
                    ('frequency_stop', {'min': 250e3, 'max': 44e9})]
            )
        ]

        self.add_parameter('envelope_padding',
                           unit='s',
                           set_cmd=None,
                           initial_value=0,
                           vals=vals.Numbers(min_value=0, max_value=10e-3),
                           docstring="Padding for any pulses that use either "
                                     "IQ and/or FM modulation. This is to "
                                     "ensure that any such pulses start before "
                                     "the gate marker pulse, and end afterwards. "
                                     "This is ignored for chirp pulses where "
                                     "FM_mode = 'IQ'.")
        self.envelope_IQ = Parameter(
            set_cmd=None,
            vals=vals.Bool(),
            initial_value=True,
            docstring="Apply the envelope padding to the IQ pulse instead of marker pulse."
                      "If True, the marker pulse length equals the pulse length, and the "
                      "I and Q pulses are extended by the envelope padding "
                      "before (after) the pulse start (stop) time. "
                      "If False, the marker pulse starts before (after) the pulse start (stop) "
                      "time, and the IQ pulses starts (stops) at the pulse start (stop) time."
                      "This means that there might be some leakage at the carrier frequency "
                      "when the IQ is zero, but the marker pulse is still high."
        )
        self.add_parameter('I_phase_correction',
                           unit='deg',
                           set_cmd=None,
                           initial_value=0,
                           docstring="Additional phase added to I pulse, which compensates "
                                     "any phase mismatch between I and Q components. Only for "
                                     "FM_mode = 'IQ'.")
        self.add_parameter('Q_phase_correction',
                           unit='deg',
                           set_cmd=None,
                           initial_value=0,
                           docstring="Additional phase added to Q pulse, which compensates "
                                     "any phase mismatch between Q and I components. Only for "
                                     "FM_mode = 'IQ'.")
        self.add_parameter('I_amplitude_correction',
                           unit='V',
                           set_cmd=None,
                           initial_value=0,
                           vals=vals.Numbers(min_value=-1, max_value=0),
                           docstring="Amplitude correction added to the amplitude of I pulse"
                                     "to compensate for any mismatch in amplitude/power "
                                     "between I and Q components. The correction is restricted "
                                     "in value between -1V to 0V to ensure the I/Q inputs do not "
                                     "receive signals above 1V. Only for FM_mode = 'IQ'.")
        self.add_parameter('Q_amplitude_correction',
                           unit='V',
                           set_cmd=None,
                           initial_value=0,
                           vals=vals.Numbers(min_value=-1, max_value=0),
                           docstring="Amplitude correction added to the amplitude of Q pulse"
                                     "to compensate for any mismatch in amplitude/power "
                                     "between Q and I components. The correction is restricted "
                                     "in value between -1V to 0V to ensure the I/Q inputs do not "
                                     "receive signals above 1V. Only for FM_mode = 'IQ'.")
        self.add_parameter('marker_amplitude',
                           unit='V',
                           set_cmd=None,
                           initial_value=1.5,
                           docstring="Amplitude of marker pulse used for gating")
        self.add_parameter('modulation_channel',
                           set_cmd=None,
                           initial_value='ext1',
                           vals=vals.Enum(*self._input_channels),
                           docstring="Channel to use for FM.")
        self.add_parameter('fix_frequency',
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool(),
                           docstring="Whether to fix frequency to current "
                                     "value, or to dynamically choose frequency"
                                     " during setup")
        self.add_parameter('fix_frequency_deviation',
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool(),
                           docstring="Whether to fix frequency_deviation to "
                                     "current value, or to dynamically choose "
                                     "deviation during setup")
        self.add_parameter('frequency_carrier_choice',
                           set_cmd=None,
                           initial_value='center',
                           vals=vals.MultiType(vals.Enum('min', 'center', 'max'),
                                               vals.Numbers()),
                           docstring='The choice for the microwave frequency, '
                                     'This is used if pulses with multiple '
                                     'frequencies are used, or if frequency '
                                     'modulation is needed. Ignored if '
                                     'fix_frequency = True. Can be either "max",'
                                     '"min", or "center", or a number which is '
                                     'the offset from the center')
        self.add_parameter('frequency',
                           unit='Hz',
                           set_cmd=None,
                           initial_value=None)
        self.add_parameter('frequency_deviation',
                           unit='Hz',
                           set_cmd=None,
                           initial_value=None)
        self.add_parameter('IQ_modulation',
                           initial_value=None,
                           vals=vals.Enum('on', 'off'),
                           docstring='Whether to use IQ modulation. This '
                                     'cannot be directly set, but is determined '
                                     'by FM_mode and whether pulses have '
                                     'frequency_sideband not None.')
        self.add_parameter('force_IQ',
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool(),
                           docstring='Force IQ modulation to be enabled, even when'
                                     'outputting only a single frequency.')
        self.add_parameter('FM_mode',
                           set_cmd=None,
                           initial_value='ramp',
                           vals=vals.Enum('ramp', 'IQ'),
                           docstring="Type of frequency modulation used. "
                                     "Can be either 'ramp' in which case the "
                                     "internal FM is used by converting a DC "
                                     "amplitude from an ext port, or 'IQ', in "
                                     "which case the internal FM is turned off.")

    def determine_instrument_settings(self, update: bool = False) -> dict:
        """Determine the frequency settings from parameters and  pulse sequence

        Used to determine additional pulses and during setup

        Args:
            update: Update the interface parameters

        Returns:
            Dictionary with three items:
            - ``IQ_modulation``: Use IQ modulation
            - ``frequency``: carrier frequency
            - ``frequency_deviation``: frequency deviation
        """
        settings = {}

        frequency_sidebands = {int(round(pulse.frequency_sideband))
                               if pulse.frequency_sideband is not None else None
                               for pulse in self.pulse_sequence}

        # Ramp mode does not work with envelope IQ since the output can only be
        # turned off via gating (0V on ramp input corresponds to a pulse at carrier)
        assert self.FM_mode() != 'ramp' or self.envelope_IQ(), \
            "Cannot use 'ramp' FM mode while envelope_IQ == False"

        if self.FM_mode() == 'IQ':
            assert frequency_sidebands == {None}, \
                "pulse.frequency_sideband must be None when FM_mode is 'IQ'"
        else:
            assert frequency_sidebands == {None} or None not in frequency_sidebands, \
                f'Sideband frequencies must either all be None, or all not ' \
                f'None when FM_mode is "ramp" (0 Hz is allowed). ' \
                f'frequency_sidebands: {frequency_sidebands}'

        if None in frequency_sidebands:
            frequency_sidebands.remove(None)

        # Find minimum and maximum frequency
        min_frequency = max_frequency = None
        for pulse in self.pulse_sequence:
            frequency_deviation = getattr(pulse, 'frequency_deviation', None)
            multiple_frequencies = getattr(pulse, 'frequencies', None)
            frequency_sideband = pulse.frequency_sideband

            if multiple_frequencies is not None:
                pulse_min_frequency = min(multiple_frequencies)
                pulse_max_frequency = max(multiple_frequencies)
            else:
                pulse_min_frequency = pulse_max_frequency = pulse.frequency
            if frequency_deviation is not None:
                pulse_min_frequency -= pulse.frequency_deviation
                pulse_max_frequency += pulse.frequency_deviation
            if frequency_sideband is not None:
                pulse_min_frequency -= pulse.frequency_sideband
                pulse_max_frequency -= pulse.frequency_sideband

            if min_frequency is None or pulse_min_frequency < min_frequency:
                min_frequency = pulse_min_frequency
            if max_frequency is None or pulse_max_frequency > max_frequency:
                max_frequency = pulse_max_frequency

        min_frequency = int(round(min_frequency))
        max_frequency = int(round(max_frequency))

        if not self.fix_frequency():
            # Choose center frequency
            if self.frequency_carrier_choice() == 'center':
                frequency_carrier = int(round((min_frequency + max_frequency) / 2))
            elif self.frequency_carrier_choice() == 'min':
                frequency_carrier = min_frequency
            elif self.frequency_carrier_choice() == 'max':
                frequency_carrier = max_frequency
            else:  # value away from center
                frequency_carrier = int(round((min_frequency + max_frequency) / 2))
                frequency_carrier += self.frequency_carrier_choice()
            settings['frequency'] = frequency_carrier
        else:
            settings['frequency'] = self.frequency()

        if not self.fix_frequency_deviation():
            settings['frequency_deviation'] = (
                int(round(max([max_frequency - settings['frequency'],
                               settings['frequency'] - min_frequency]))))
        else:
            settings['frequency_deviation'] = self.frequency_deviation()

        assert settings['frequency_deviation'] < 80e6 or self.FM_mode() == 'IQ', \
            "Maximum FM frequency deviation is 80 MHz if FM_mode == 'ramp'. " \
            f"Current frequency deviation: {self.frequency_deviation() / 1e6} MHz"

        if frequency_sidebands or self.force_IQ() or (self.FM_mode() == 'IQ' and min_frequency != max_frequency):
            self.IQ_modulation._save_val('on')
        else:
            settings['IQ_modulation'] = 'off'


        if update:
            self.frequency(settings['frequency'])
            self.frequency_deviation(settings['frequency_deviation'])
            self.IQ_modulation._latest["raw_value"] = settings["IQ_modulation"]
            self.IQ_modulation.get()

        return settings

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses, such as IQ modulation pulses
        """
        if not self.pulse_sequence:
            return []

        powers = list({pulse.power for pulse in self.pulse_sequence})
        assert len(powers) == 1, "Cannot handle multiple pulse powers"

        frequency_settings = self.determine_instrument_settings()

        additional_pulses = []
        for pulse in self.pulse_sequence:
            additional_pulses += pulse.implementation.get_additional_pulses(
                interface=self, **frequency_settings
            )

        # Ensure marker pulses are not overlapping
        marker_pulses = [p for p in additional_pulses if isinstance(p, MarkerPulse)]
        if marker_pulses:
            marker_pulses = sorted(marker_pulses, key=lambda p: p.t_start)
            current_pulse = marker_pulses[0]
            merged_marker_pulses = [current_pulse]
            for pulse in marker_pulses[1:]:
                if pulse.t_start <= current_pulse.t_stop:
                    # Marker pulse overlaps with previous pulse
                    current_pulse.t_stop = max(pulse.t_stop, current_pulse.t_stop)
                else:
                    # Pulse starts after previous pulse
                    merged_marker_pulses.append(pulse)
                    current_pulse = pulse
            nonmarker_pulses = [p for p in additional_pulses if not isinstance(p, MarkerPulse)]
            additional_pulses = [*merged_marker_pulses, *nonmarker_pulses]

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

        # Determine frequency, frequency_deviation, and IQ_modulation
        self.determine_instrument_settings(update=True)

        self.instrument.frequency(self.frequency())
        self.instrument.power(powers[0])

        if self.frequency_deviation() > 0 and self.FM_mode() == 'ramp':
            self.instrument.frequency_modulation('on')
            self.instrument.frequency_deviation(self.frequency_deviation())
            self.instrument.frequency_modulation_source(self.modulation_channel())
        else:
            self.instrument.frequency_modulation('off')

        self.instrument.pulse_modulation('on')
        self.instrument.pulse_modulation_source('ext')
        self.instrument.output_modulation('on')

        if self.IQ_modulation() == 'on':
            self.instrument.internal_IQ_modulation('on')
        else:
            self.instrument.internal_IQ_modulation('off')

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def start(self):
        """Start instrument"""
        self.instrument.RF_output('on')
        sleep(0.4)  # Sleep a short while for the RF to output

    def stop(self):
        """Stop instrument"""
        self.instrument.RF_output('off')


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def target_pulse(self, pulse, interface, **kwargs):
        assert pulse.power is not None, "Pulse must have power defined"
        return super().target_pulse(pulse, interface, **kwargs)

    def get_additional_pulses(
            self,
            interface: InstrumentInterface,
            IQ_modulation,
            frequency,
            frequency_deviation,
            **kwargs
    ):
        # Add an envelope pulse
        t_offset = 0 if interface.envelope_IQ() else interface.envelope_padding()
        additional_pulses = [
            MarkerPulse(t_start=self.pulse.t_start - t_offset,
                        t_stop=self.pulse.t_stop + t_offset,
                        amplitude=interface.marker_amplitude(),
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': 'trig_in'})]

        if IQ_modulation == 'off':
            if frequency_deviation == 0:  # No IQ modulation nor FM
                amplitude_FM = None
                frequency_IQ = None
                pass
            else:  # No IQ modulation, but FM
                frequency_difference = self.pulse.frequency - frequency
                amplitude_FM = frequency_difference / frequency_deviation
                frequency_IQ = None
        else:  # IQ_modulation == 'on'
            if interface.FM_mode() == 'ramp':
                assert self.pulse.frequency_sideband is not None, \
                    "Pulse.frequency_sideband must be defined when " \
                    "FM_mode = 'ramp' and IQ_modulation = 'on'"

                frequency = self.pulse.frequency + self.pulse.frequency_sideband
                frequency_difference = frequency - frequency
                amplitude_FM = frequency_difference / frequency_deviation
                frequency_IQ = self.pulse.frequency_sideband
            else:  # interface.FM_mode() == 'IQ'
                amplitude_FM = None
                frequency_IQ = self.pulse.frequency - frequency

        if frequency_IQ is not None and frequency_IQ != 0:
            t_offset = interface.envelope_padding() if interface.envelope_IQ() else 0
            additional_pulses.extend([
                SinePulse(name='sideband_I',
                          t_start=self.pulse.t_start - t_offset,
                          t_stop=self.pulse.t_stop + t_offset,
                          frequency=frequency_IQ,
                          amplitude=1 + interface.I_amplitude_correction(),
                          phase=self.pulse.phase + interface.I_phase_correction(),
                          phase_reference=self.pulse.phase_reference,
                          offset=self.pulse.offset,
                          connection_requirements={
                              'input_instrument': interface.instrument_name(),
                              'input_channel': 'I'}),
                SinePulse(name='sideband_Q',
                          t_start=self.pulse.t_start - t_offset,
                          t_stop=self.pulse.t_stop + t_offset,
                          frequency=frequency_IQ,
                          phase=self.pulse.phase - 90 + interface.Q_phase_correction(),
                          amplitude=1 + interface.Q_amplitude_correction(),
                          phase_reference=self.pulse.phase_reference,
                          offset=self.pulse.offset,
                          connection_requirements={
                              'input_instrument': interface.instrument_name(),
                              'input_channel': 'Q'})])
        elif frequency_IQ == 0:
            # Frequency is zero, add DC pulses instead of sine pulses
            amplitudes = {
                'I': np.sin(2 * np.pi * (self.pulse.phase +
                                         interface.I_phase_correction()) / 360),
                'Q': np.sin(2 * np.pi * (self.pulse.phase - 90 +
                                         interface.Q_phase_correction()) / 360)
            }

            for quadrature, amplitude in amplitudes.items():
                # Pulse is probably not needed if amplitude is 0, but we leave it for now.
                # if amplitude == 0:
                #     continue
                additional_pulses.append(
                    DCPulse(name=f'sideband_{quadrature}',
                            t_start=self.pulse.t_start - interface.envelope_padding(),
                            t_stop=self.pulse.t_stop + interface.envelope_padding(),
                            amplitude=amplitude,
                            connection_requirements={
                                'input_instrument': interface.instrument_name(),
                                'input_channel': quadrature}
                            )
                )

        if amplitude_FM is not None:
            assert abs(amplitude_FM) <= 1 + 1e-13, \
                f'abs(amplitude) {amplitude_FM} cannot be higher than 1'

            # Note that amplitude_FM implies envelope_IQ == True
            additional_pulses.append(
                DCPulse(t_start=self.pulse.t_start - interface.envelope_padding(),
                        t_stop=self.pulse.t_stop + interface.envelope_padding(),
                        amplitude=amplitude_FM,
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': interface.modulation_channel()}))
        return additional_pulses


class FrequencyRampPulseImplementation(PulseImplementation):
    pulse_class = FrequencyRampPulse

    def target_pulse(self, pulse, interface, **kwargs):
        assert pulse.power is not None, f"Pulse must have power defined {pulse}"
        assert pulse.frequency_start != pulse.frequency_stop, \
            f"Pulse frequency_start must differ from frequency_stop {pulse}"
        return super().target_pulse(pulse, interface, **kwargs)

    def get_additional_pulses(
            self,
            interface: InstrumentInterface,
            IQ_modulation,
            frequency,
            frequency_deviation,
            **kwargs
    ):
        assert self.pulse.t_start >= interface.envelope_padding(), \
            f"Keysight E8267D uses envelope padding " \
            f"{interface.envelope_padding()} s before and after pulse for FM " \
            f"and IQ modulation, so this is the minimum pulse.t_start."

        # Add an envelope pulse
        additional_pulses = [
            MarkerPulse(t_start=self.pulse.t_start, t_stop=self.pulse.t_stop,
                        amplitude=interface.marker_amplitude(),
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': 'trig_in'},
                        name=f'{self.pulse.name}_marker')]

        if IQ_modulation == 'off':
            frequency_IQ = None
            frequency_IQ_start = None
            frequency_IQ_stop = None
            frequency_offset = frequency
        elif interface.FM_mode() == 'ramp':  # interface.IQ_modulation() == 'on'
            assert self.pulse.frequency_sideband is not None, \
                "Pulse.frequency_sideband must be defined when " \
                "FM_mode = 'ramp' and IQ_modulation = 'on'"
            frequency_IQ = self.pulse.frequency_sideband
            frequency_IQ_start = None
            frequency_IQ_stop = None
            frequency_offset = self.pulse.frequency + self.pulse.frequency_sideband
        else:  # interface.FM_mode() == 'IQ'
            frequency_IQ = None
            frequency_IQ_start = self.pulse.frequency_start - frequency
            frequency_IQ_stop = self.pulse.frequency_stop - frequency
            frequency_offset = None

        if frequency_IQ is not None:
            additional_pulses.extend([
                SinePulse(name='sideband_I',
                          t_start=self.pulse.t_start - interface.envelope_padding(),
                          t_stop=self.pulse.t_stop + interface.envelope_padding(),
                          frequency=frequency_IQ,
                          amplitude=1 + interface.I_amplitude_correction(),
                          phase=self.pulse.phase + interface.I_phase_correction(),
                          phase_reference=self.pulse.phase_reference,
                          offset=self.pulse.offset,
                          connection_requirements={
                              'input_instrument': interface.instrument_name(),
                              'input_channel': 'I'}),
                SinePulse(name='sideband_Q',
                          t_start=self.pulse.t_start - interface.envelope_padding(),
                          t_stop=self.pulse.t_stop + interface.envelope_padding(),
                          frequency=frequency_IQ,
                          phase=self.pulse.phase - 90 + interface.Q_phase_correction(),
                          phase_reference=self.pulse.phase_reference,
                          offset=self.pulse.offset,
                          amplitude=1 + interface.Q_amplitude_correction(),
                          connection_requirements={
                              'input_instrument': interface.instrument_name(),
                              'input_channel': 'Q'})])
        elif frequency_IQ_start is not None:
            additional_pulses.extend([
                FrequencyRampPulse(name='sideband_I',
                                   t_start=self.pulse.t_start,
                                   t_stop=self.pulse.t_stop,
                                   frequency_start=frequency_IQ_start,
                                   frequency_stop=frequency_IQ_stop,
                                   amplitude=1 + interface.I_amplitude_correction(),
                                   phase=self.pulse.phase + interface.I_phase_correction(),
                                   phase_reference=self.pulse.phase_reference,
                                   offset=self.pulse.offset,
                                   connection_requirements={
                                       'input_instrument': interface.instrument_name(),
                                       'input_channel': 'I'}),
                FrequencyRampPulse(name='sideband_Q',
                                   t_start=self.pulse.t_start,
                                   t_stop=self.pulse.t_stop,
                                   frequency_start=frequency_IQ_start,
                                   frequency_stop=frequency_IQ_stop,
                                   amplitude=1 + interface.Q_amplitude_correction(),
                                   phase=self.pulse.phase - 90 + interface.Q_phase_correction(),
                                   phase_reference=self.pulse.phase_reference,
                                   offset=self.pulse.offset,
                                   connection_requirements={
                                       'input_instrument': interface.instrument_name(),
                                       'input_channel': 'Q'})])

        if frequency_offset is not None:  # Add a DC ramp pulse for FM
            amplitude_start = (self.pulse.frequency_start - frequency_offset) \
                              / abs(frequency_deviation)
            amplitude_stop = (self.pulse.frequency_stop - frequency_offset) \
                             / abs(frequency_deviation)
            additional_pulses.append(
                DCRampPulse(t_start=self.pulse.t_start,
                            t_stop=self.pulse.t_stop,
                            amplitude_start=amplitude_start,
                            amplitude_stop=amplitude_stop,
                            connection_requirements={
                                'input_instrument': interface.instrument_name(),
                                'input_channel': interface.modulation_channel()}))

            if interface.envelope_padding() > 0:  # Add padding DC pulses at start and end
                additional_pulses.extend((
                    DCPulse(t_start=self.pulse.t_start - interface.envelope_padding(),
                            t_stop=self.pulse.t_start,
                            amplitude=amplitude_start,
                            connection_requirements={
                                'input_instrument': interface.instrument_name(),
                                'input_channel': interface.modulation_channel()}),
                    DCPulse(t_start=self.pulse.t_stop,
                            t_stop=self.pulse.t_stop + interface.envelope_padding(),
                            amplitude=amplitude_stop,
                            connection_requirements={
                                'input_instrument': interface.instrument_name(),
                                'input_channel': interface.modulation_channel()})))
        return additional_pulses


class MultiSinePulseImplementation(PulseImplementation):
    pulse_class = MultiSinePulse

    def target_pulse(self, pulse, interface, **kwargs):
        assert pulse.power is not None, "Pulse must have power defined"
        return super().target_pulse(pulse, interface, **kwargs)

    def get_additional_pulses(self, interface: InstrumentInterface):
        # Add an envelope pulse
        additional_pulses = [
            MarkerPulse(t_start=self.pulse.t_start, t_stop=self.pulse.t_stop,
                        amplitude=interface.marker_amplitude(),
                        connection_requirements={
                            'input_instrument': interface.instrument_name(),
                            'input_channel': 'trig_in'})]

        assert (interface.IQ_modulation() == 'on') and \
               (interface.FM_mode() == 'IQ'), 'FM_mode should be IQ and IQ_modulation should be ON ' \
                                              'for MultiSinePulse.'

        # To ensure the waveform is limited to +- 1V:
        amplitudes_I = list(np.array(self.pulse.amplitudes) / len(self.pulse.amplitudes) +
                            interface.I_amplitude_correction())
        amplitudes_Q = list(np.array(self.pulse.amplitudes) / len(self.pulse.amplitudes) +
                            interface.Q_amplitude_correction())
        frequencies_IQ = list(np.array(self.pulse.frequencies) - interface.frequency())
        # Shifting all phases in order to start after envelope padding:
        dphases = - np.array(frequencies_IQ) * interface.envelope_padding() * 360  # in degrees
        phases_I = list(np.array(self.pulse.phases) + dphases + interface.I_phase_correction())
        phases_Q = list(np.array(self.pulse.phases) + dphases - 90 + interface.Q_phase_correction())

        assert all(0 <= amp <= 1 for amp in
                   self.pulse.amplitudes), f"Not all amplitudes in MultiSinePulse list: " \
                                           f"{self.pulse.amplitudes} are between 0 and 1V."
        max_input_I = sum(amplitudes_I) + abs(self.pulse.offset)
        max_input_Q = sum(amplitudes_Q) + abs(self.pulse.offset)
        assert (max_input_I <= 1) and (max_input_Q <= 1), \
            f"Input_I ({max_input_I}) or Input_Q ({max_input_Q}) voltages are above 1V."

        additional_pulses.extend([
            MultiSinePulse(name='sideband_I',
                           t_start=self.pulse.t_start - interface.envelope_padding(),
                           t_stop=self.pulse.t_stop + interface.envelope_padding(),
                           frequencies=frequencies_IQ,
                           amplitudes=amplitudes_I,
                           phases=phases_I,
                           phase_reference=self.pulse.phase_reference,
                           offset=self.pulse.offset,
                           connection_requirements={
                               'input_instrument': interface.instrument_name(),
                               'input_channel': 'I'}),
            MultiSinePulse(name='sideband_Q',
                           t_start=self.pulse.t_start - interface.envelope_padding(),
                           t_stop=self.pulse.t_stop + interface.envelope_padding(),
                           frequencies=frequencies_IQ,
                           amplitudes=amplitudes_Q,
                           phases=phases_Q,
                           phase_reference=self.pulse.phase_reference,
                           offset=self.pulse.offset,
                           connection_requirements={
                               'input_instrument': interface.instrument_name(),
                               'input_channel': 'Q'})])
        return additional_pulses
