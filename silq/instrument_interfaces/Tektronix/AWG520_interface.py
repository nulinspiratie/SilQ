import numpy as np
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import DCPulse, DCRampPulse, TriggerPulse, SinePulse, \
    PulseImplementation
from silq.tools.general_tools import arreqclose_in_list

from qcodes.utils.validators import Lists, Enum, Numbers


logger = logging.getLogger(__name__)

class AWG520Interface(InstrumentInterface):
    """

    Notes:
        - Sets first point of each waveform to final voltage of previous
          waveform because this is the value used when the previous waveform
          ended and is waiting for triggers.
        - Amplitude significantly decreases above 400 MHz at 1 GHz sampling.
    Todo:
        Check if only channel 2 can be programmed
        Add marker channels
    """
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)
        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, output=True)
            for k in [1,2]
        }

        self._channels = {
            **self._output_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True)
        }

        # TODO: Add pulse implementations
        self.pulse_implementations = [
            DCPulseImplementation(pulse_requirements=[('amplitude', {'max': 2})]),
            SinePulseImplementation(pulse_requirements=[('frequency', {'max': 500e6}),
                                                        ('amplitude', {'max': 2})])
        ]

        self.add_parameter('final_delay',
                           unit='s',
                           set_cmd=None,
                           initial_value=.1e-6,
                           docstring='Time subtracted from each waveform to ensure '
                               'that it is finished once next trigger arrives.')
        self.add_parameter('trigger_in_duration',
                           unit='s',
                           set_cmd=None,
                           initial_value=.1e-6)
        self.add_parameter('active_channels',
                           set_cmd=None,
                           vals=Lists(Enum(1,2)))
        self.add_parameter('sampling_rate',
                           unit='1/s',
                           initial_value=1e9,
                           set_cmd=None,
                           vals=Numbers(max_value=1e9))

    def get_additional_pulses(self, **kwargs):
        # Return empty list if no pulses are in the pulse sequence
        if not self.pulse_sequence or self.is_primary():
            return []

        active_channels = {pulse.connection.output['channel']
                           for pulse in self.pulse_sequence}

        # Add DCPulse(amplitude=0) for any gaps between pulses
        gap_pulses = []
        for channel in active_channels:
            pulses = sorted(self.pulse_sequence.get_pulses(output_channel=channel),
                            key=lambda pulse: pulse.t_start)
            t = 0
            for pulse in pulses:
                if pulse.t_start < t:
                    raise RuntimeError(f'Pulse {pulse} starts before previous'
                                       f'pulse is finished')
                elif pulse.t_start > t:
                    # Add DCPulse to fill gap between pulses
                    gap_pulse = DCPulse(t_start=t,
                                        t_stop=pulse.t_start,
                                        amplitude=0)
                    gap_pulses.append(self.get_pulse_implementation(gap_pulse))
                t = pulse.t_stop

            if t < self.pulse_sequence.duration:
                # Add DCPulse to fill gap between pulses
                gap_pulse = DCPulse(t_start=t,
                                    t_stop=self.pulse_sequence.duration,
                                    amplitude=0,
                                    connection_requirements={
                                        'output_instrument': self.instrument.name,
                                        'output_channel': channel
                                    })
                gap_pulses.append(self.get_pulse_implementation(gap_pulse))

        self.pulse_sequence.add(*gap_pulses)

        # If a pulse starts on one channel and needs a trigger while another
        # pulse is still active on the other channel, this would cause the other
        # pulse to move onto the next pulse prematurely. This only happens if
        # the other pulse finished its waveform and is waiting for a trigger.
        # Here we check that this does not occur.
        if len(active_channels) > 1:
            t = 0
            for t_start in self.pulse_sequence.t_start_list:
                pulses = {ch: self.pulse_sequence.get_pulse(t_start=t_start,
                                                            output_channel=ch)
                          for ch in ['ch1', 'ch2']}
                assert t == pulses['ch1'].t_start, \
                    f"Gap between t_stop={t} and pulse.t_start{pulses['ch1'].t_start}"
                assert pulses['ch1'].t_stop == pulses['ch2'].t_stop, \
                    f"Pulses do not have same t_stop. Pulses: {pulses}"
                t = pulses['ch1'].t_stop

        # TODO test if first waveform needs trigger as well
        additional_pulses = [
            TriggerPulse(t_start=t_start,
                         duration=self.trigger_in_duration(),
                         connection_requirements={
                             'input_instrument': self.instrument_name(),
                             'trigger': True})
            for t_start in self.pulse_sequence.t_start_list
        ]

        return additional_pulses

    def setup(self, is_primary=False, **kwargs):
        if is_primary:
            raise RuntimeError('AWG520 cannot function as primary instrument')

        self.active_channels({pulse.connection.output['channel']
                              for pulse in self.pulse_sequence})

        # Create waveforms and sequence
        waveforms = []
        sequence = {ch: [] for ch in self.active_channels()}

        t = 0
        previous_voltages = {ch: 0 for ch in self.active_channels()}
        while t < self.pulse_sequence.duration:
            pulses = {ch: self.pulse_sequence.get_pulse(t_start=t, output_channel=ch)
                      for ch in ['ch1', 'ch2']}
            assert pulses['ch1'].t_stop == pulses['ch2'].t_stop, \
                f"Pulses do not have same t_stop. Pulses: {pulses}"

            # Add pulses to waveforms and sequences
            for ch, pulse in pulses.items():
                assert pulse is not None,\
                    f"Could not find unique pulse for channel {ch} at t_start={t}"

                waveform = pulse.implementation.implement(
                    first_point_voltage=previous_voltages[ch],
                    sampling_rate=self.sampling_rate())
                waveform_idx = arreqclose_in_list(waveform, self.waveforms[ch])
                if waveform_idx is None:
                    waveforms.append(waveform)
                    waveform_idx = len(waveforms) - 1
                sequence[ch].append(waveform_idx)

                # Update previous voltage to last point of current waveform
                previous_voltages[ch] = waveform[-1]

            t = pulse.t_stop

        self.instrument.stop()
        self.instrument.trigger_mode('ENH')
        self.instrument.trigger_level(1)
        self.instrument.clock_freq(self.sampling_rate())
        for ch in [self.instrument.ch1, self.instrument.ch2]:
            ch.offset(0)
            # TODO What happens when we increase channel amplitude?
            ch.amplitude(1)
            ch.status('OFF')

        # Upload waveforms
        self.instrument.change_folder('silq')
        filenames = []
        for k, waveform in enumerate(waveforms):
            filename = f'waveform_{k}.wfm'
            marker1 = marker2 = np.zeros(len(waveform))
            self.instrument.send_waveform(waveform,
                                          marker1,
                                          marker2,
                                          filename,
                                          clock=self.sampling_rate)
            filenames.append(filename)

        # Upload sequence
        sequence_waveform_names = [[filenames[waveform_idx]
                                    for waveform_idx in sequence[ch]]
                                   for ch in self.active_channels()]
        N_instructions = len(sequence_waveform_names[0])
        repetitions = np.ones(N_instructions)
        wait_trigger = np.ones(N_instructions)
        goto_one = np.zeros(N_instructions)
        logic_jump = np.zeros(N_instructions)
        if len(self.active_channels()) == 1:
            sequence_waveform_names = sequence_waveform_names[0]
        self.instrument.send_sequence(sequence_waveform_names,
                                      repetitions,
                                      wait_trigger,
                                      goto_one,
                                      logic_jump,
                                      'sequence.seq')

        for error in self.instrument.get_errors():
            logger.warning(error)

    def start(self):
        self.instrument.ch1.status('ON')
        self.instrument.ch2.status('ON')
        self.instrument.start()

    def stop(self):
        self.instrument.stop()


class DCPulseImplementation(PulseImplementation):
    # Number of points in a waveform (must be at least 256)
    pulse_class = DCPulse
    pts = 256

    def implement(self,
                  first_point_voltage: int = 0,
                  sampling_rate: float = 1e9, **kwargs) -> np.ndarray:
        """Implements the DC pulses for the AWG520

        Args:
            first_point_voltage: Voltage to set first point of waveform to.
                When the previous waveform ends, the voltage is set to the first
                point of the next voltage.
            sampling_rate: AWG sampling rate
            final_delay: Final part of waveform to skip. If this does not exist,
                the waveform may not have finished when next trigger arrives,
                in which case the trigger is ignored.
                
        Returns:
            waveform
        """

        # AWG520 requires waveforms of at least 256 points
        waveform = np. self.voltage * np.ones(self.pts)
        waveform[0] = first_point_voltage
        return waveform


class SinePulseImplementation(PulseImplementation):
    # Number of points in a waveform (must be at least 256)
    pulse_class = SinePulse

    def implement(self,
                  first_point_voltage: int = 0,
                  sampling_rate: float = 1e9,
                  final_delay: float = 0, **kwargs) -> np.ndarray:
        """Implements Sine pulses for the AWG520

        Args:
            first_point_voltage: Voltage to set first point of waveform to.
                When the previous waveform ends, the voltage is set to the first
                point of the next voltage.
            sampling_rate: AWG sampling rate
            final_delay: Final part of waveform to skip. If this does not exist,
                the waveform may not have finished when next trigger arrives,
                in which case the trigger is ignored.
        Returns:
            Waveform array
        """
        t_list = np.arange(self.pulse.t_start,
                           self.pulse.t_stop - final_delay,
                           1 / sampling_rate)
        waveform = self.pulse.get_voltage(t_list)
        waveform[0] = first_point_voltage
        return waveform
