import logging
from typing import List, Union

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import (
    Pulse,
    SinePulse,
    MarkerPulse,
    PulseImplementation,
)

from qcodes import Parameter, ParameterNode
from qcodes import validators as vals


logger = logging.getLogger(__name__)


class SGS100AInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._input_channels = {
            "I": Channel(instrument_name=self.instrument_name(), name="I", input=True),
            "Q": Channel(instrument_name=self.instrument_name(), name="Q", input=True),
            "pulse_mod": Channel(
                instrument_name=self.instrument_name(), name="pulse_mod", input=True
            ),
        }

        self._output_channels = {
            "RF_out": Channel(
                instrument_name=self.instrument_name(), name="RF_out", output=True
            )
        }

        self._channels = {
            **self._input_channels,
            **self._output_channels,
            "sync": Channel(
                instrument_name=self.instrument_name(), name="sync", output=True
            ),
        }

        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[
                    ("frequency", {"min": 1e6, "max": 40e9}),
                    ("power", {"min": -120, "max": 25}),
                    ("duration", {"min": 100e-9}),
                ]
            ),
        ]

        self.envelope_padding = Parameter(
            "envelope_padding",
            unit="s",
            set_cmd=None,
            initial_value=0,
            vals=vals.Numbers(min_value=0, max_value=10e-3),
            docstring="Padding for any pulses that use IQ modulation. "
                      "This is to ensure that any IQ pulses such as sine waves "
                      "are applied a bit before the pulse starts. The marker pulse " 
                      "used for pulse modulation does not use any envelope padding."
        )
        self.marker_amplitude = Parameter(
            unit="V",
            set_cmd=None,
            initial_value=1.5,
            docstring="Amplitude of marker pulse used for gating",
        )
        self.fix_frequency = Parameter(
            set_cmd=None,
            initial_value=False,
            vals=vals.Bool(),
            docstring="Whether to fix frequency to current value, or to "
            "dynamically choose frequency during setup",
        )
        self.frequency_carrier_choice = Parameter(
            set_cmd=None,
            initial_value="center",
            vals=vals.MultiType(vals.Enum("min", "center", "max"), vals.Numbers()),
            docstring="The choice for the microwave frequency, This is used if "
            "pulses with multiple frequencies are used, or if frequency "
            "modulation is needed. Ignored if fix_frequency = True. "
            'Can be either "max", "min", or "center", or a '
            "number which is the offset from the center",
        )
        self.frequency = Parameter(
            unit="Hz", set_cmd=None, initial_value=self.instrument.frequency()
        )
        self.IQ_modulation = Parameter(
            initial_value=None,
            vals=vals.Bool(),
            docstring="Whether to use IQ modulation. Cannot be directly set, but "
                      "is set internally if there are pulses with multiple "
                      "frequencies, self.fix_frequency() is True, or "
                      "self.force_IQ_modulation() is True."
        )
        self.force_IQ_modulation = Parameter(
            initial_value=False,
            vals=vals.Bool(),
            set_cmd=None,
            docstring="Whether to force IQ modulation."
        )

        # Add parameters that are not set via setup
        self.additional_settings = ParameterNode()
        for parameter_name in [
            "phase",
            "maximum_power",
            "IQ_impairment",
            "I_leakage",
            "Q_leakage",
            "Q_offset",
            "IQ_ratio",
            "IQ_wideband",
            "IQ_crestfactor",
            "reference_oscillator_source",
            "reference_oscillator_output_frequency",
            "reference_oscillator_external_frequency",
        ]:
            parameter = getattr(self.instrument, parameter_name)
            setattr(self.additional_settings, parameter_name, parameter)

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses, such as IQ modulation pulses
        """
        if not self.pulse_sequence:
            return []

        assert all(pulse.frequency_sideband is None for pulse in self.pulse_sequence)

        powers = {pulse.power for pulse in self.pulse_sequence}
        assert len(powers) == 1, f"SGS100A pulses cannot have different powers {powers}"

        frequencies = {pulse.frequency for pulse in self.pulse_sequence}

        # Check whether to use IQ modulation
        if len(frequencies) > 1 or self.fix_frequency() or self.force_IQ_modulation():
            # Set protected IQ_modulation parameter
            self.IQ_modulation._latest['raw_value'] = True
            self.IQ_modulation.get()

            if not self.fix_frequency():
                if self.frequency_carrier_choice() == 'center':
                    self.frequency((min(frequencies) + max(frequencies)) / 2)
                elif self.frequency_carrier_choice() == 'min':
                    self.frequency(min(frequencies))
                elif self.frequency_carrier_choice() == 'max':
                    self.frequency(max(frequencies))
                else:
                    frequency = (min(frequencies) + max(frequencies)) / 2
                    frequency += self.frequency_carrier_choice()
                    self.frequency(frequency)
        else:
            # Set protected IQ_modulation parameter
            self.IQ_modulation._latest['raw_value'] = False
            self.IQ_modulation.get()

            self.frequency(next(iter(frequencies)))

        # If IQ modulation is used, ensure pulses are spaced by more than twice
        # the envelope padding
        if self.IQ_modulation():
            for pulse in self.pulse_sequence:
                overlapping_pulses = [p for p in self.pulse_sequence
                       if 2*self.envelope_padding() < p.t_stop - pulse.t_start < 0]
                if any(overlapping_pulses):
                    raise RuntimeError(f'Pulse {pulse} start time {pulse.t_start} '
                                       f'is less than 2*envelope_padding of '
                                       f'previous pulse {overlapping_pulses}')

        additional_pulses = []
        marker_pulse = None
        for pulse in self.pulse_sequence:
            if marker_pulse is not None and pulse.t_start == marker_pulse.t_stop:
                # Marker pulse already exists, extend the duration
                marker_pulse.t_stop = pulse.t_stop
            else:
                # Request a new marker pulse
                marker_pulse = MarkerPulse(
                    t_start=pulse.t_start,
                    t_stop=pulse.t_stop,
                    amplitude=self.marker_amplitude(),
                    connection_requirements={
                        'input_instrument': self.instrument_name(),
                        'input_channel': "pulse_mod"
                    }
                )
                additional_pulses.append(marker_pulse)

            # Handle any additional pulses such as those for IQ modulation
            additional_pulses += pulse.implementation.get_additional_pulses(self)

        return additional_pulses


    def setup(self, **kwargs):
        """Setup all instrument settings to output pulse sequence.
        Parameters that are not automatically set are in interface.additional_settings
        """
        self.stop()

        # Use normal operation mode, not baseband bypass
        self.instrument.operation_mode("normal")

        self.instrument.frequency(self.frequency())
        power = next(pulse.power for pulse in self.pulse_sequence)
        self.instrument.power(power)

        self.instrument.pulse_modulation_state('on')
        self.instrument.pulse_modulation_source('external')

        if self.IQ_modulation():
            self.instrument.IQ_modulation('on')
        else:
            self.instrument.IQ_modulation('off')

    def start(self):
        """Turn all active instrument channels on"""
        self.instrument.on()

    def stop(self):
        self.instrument.off()


class SinePulseImplementation(PulseImplementation):
    pulse_class = SinePulse

    def target_pulse(self, pulse, interface, **kwargs):
        assert pulse.power is not None, "Pulse must have power defined"
        return super().target_pulse(pulse, interface, **kwargs)

    def get_additional_pulses(self, interface: InstrumentInterface):
        if not interface.IQ_modulation():
            return []
        else:  # interface.IQ_modulation() == 'on'
            frequency_IQ = self.pulse.frequency - interface.frequency()
            additional_pulses = [
                SinePulse(name='sideband_I',
                          t_start=self.pulse.t_start - interface.envelope_padding(),
                          t_stop=self.pulse.t_stop + interface.envelope_padding(),
                          frequency=frequency_IQ,
                          amplitude=1,
                          phase=self.pulse.phase,
                          connection_requirements={
                              'input_instrument': interface.instrument_name(),
                              'input_channel': 'I'}),
                SinePulse(name='sideband_Q',
                          t_start=self.pulse.t_start - interface.envelope_padding(),
                          t_stop=self.pulse.t_stop + interface.envelope_padding(),
                          frequency=frequency_IQ,
                          phase=self.pulse.phase-90,
                          amplitude=1,
                          connection_requirements={
                              'input_instrument': interface.instrument_name(),
                              'input_channel': 'Q'})
            ]
            return additional_pulses