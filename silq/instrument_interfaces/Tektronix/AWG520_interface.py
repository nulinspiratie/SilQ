import numpy as np
import logging
from time import sleep
from copy import copy

import silq
from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import DCPulse, DCRampPulse, TriggerPulse, SinePulse, \
    PulseImplementation
from qcodes.utils.helpers import arreqclose_in_list
from silq.tools.pulse_tools import pulse_to_waveform_sequence
from qcodes.utils.validators import Lists, Enum, Numbers


logger = logging.getLogger(__name__)

class AWG520Interface(InstrumentInterface):
    """

    Notes:
        - Sets first point of each waveform to final voltage of previous
          waveform because this is the value used when the previous waveform
          ended and is waiting for triggers.
        - Amplitude significantly decreases above 400 MHz at 1 GHz sampling.
        - The interface receives a trigger at the end of the sequence, ignoring
          any final_delay. This means that the pulse sequence will already have
          restarted during any final delay, which may cause issues.
    Todo:
        Check if only channel 2 can be programmed
        Add marker channels
    """

    max_waveforms = 200
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

        self.pulse_implementations = [
            DCPulseImplementation(pulse_requirements=[('amplitude', {'min': -1,
                                                                     'max': 1})]),
            SinePulseImplementation(pulse_requirements=[('frequency', {'max': 500e6}),
                                                        ('amplitude', {'min': -1,
                                                                       'max': 1})])
        ]

        self.add_parameter('pulse_final_delay',
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
                           vals=Lists(Enum('ch1', 'ch2')))
        self.add_parameter('sampling_rate',
                           unit='1/s',
                           initial_value=1e9,
                           set_cmd=None,
                           vals=Numbers(max_value=1e9))

        self.waveforms = {}
        self.waveform_filenames = {}
        self.sequence = {}

        # Create silq folder to place waveforms and sequences in
        self.instrument.change_folder('/silq', create_if_necessary=True)
        # Delete all files in folder
        self.instrument.delete_all_files()

    def get_additional_pulses(self, **kwargs):
        # Return empty list if no pulses are in the pulse sequence
        if not self.pulse_sequence or self.is_primary():
            return []

        active_channels = list(set(pulse.connection.output['channel'].name
                                   for pulse in self.pulse_sequence))

        connections = {ch: self.pulse_sequence.get_connection(
            output_instrument=self.instrument.name,
            output_channel=ch) for ch in active_channels}

        # If a pulse starts on one channel and needs a trigger while another
        # pulse is still active on the other channel, this would cause the other
        # pulse to move onto the next pulse prematurely. This only happens if
        # the other pulse finished its waveform and is waiting for a trigger.
        # Here we check that this does not occur.

        if len(active_channels) > 1:
            t = 0
            gap_pulses = []
            for t_start in self.pulse_sequence.t_start_list:

                if t_start < t:
                    raise RuntimeError(
                        f'Pulse starting before end of previous pulse {t}')
                elif t_start > t:
                    # Add gap pulses for each channel
                    for ch in active_channels:
                        gap_pulse = DCPulse(t_start=t,
                                            t_stop=t_start,
                                            amplitude=0,
                                            connection=connections[ch])
                        gap_pulses.append(self.get_pulse_implementation(gap_pulse))
                    t = t_start

                pulses = {ch: self.pulse_sequence.get_pulse(t_start=t_start,
                                                            output_channel=ch)
                          for ch in active_channels}

                if pulses['ch1'] is None and pulses['ch2'] is None:
                    raise RuntimeError(f"pulse sequence has t_start={t_start}, "
                                        "but couldn't find pulse for either channel")
                elif pulses['ch1'] is not None and pulses['ch2'] is not None:
                    assert pulses['ch1'].t_stop == pulses['ch2'].t_stop, \
                        f"t_stop of pulses starting at {t_start} must be equal." \
                        f"This is a current limitation of the AWG interface."
                    t = pulses['ch1'].t_stop
                elif pulses['ch1'] is not None:
                    # add gap pulse for ch2
                    gap_pulse = DCPulse(t_start=t_start,
                                        t_stop=pulses['ch1'].t_stop,
                                        amplitude=0,
                                        connection=connections['ch2'])
                    gap_pulses.append(self.get_pulse_implementation(gap_pulse))
                    t = pulses['ch1'].t_stop
                else:
                    # add gap pulse for ch1
                    gap_pulse = DCPulse(t_start=t_start,
                                        t_stop=pulses['ch2'].t_stop,
                                        amplitude=0,
                                        connection=connections['ch1'])
                    gap_pulses.append(self.get_pulse_implementation(gap_pulse))
                    t = pulses['ch2'].t_stop
        else:
            ch = active_channels[0]
            # only add gap pulses for active channel
            connection = self.pulse_sequence.get_connection(
                output_instrument=self.instrument.name,
                output_channel=ch)
            gap_pulses = []
            t = 0
            for t_start in self.pulse_sequence.t_start_list:
                if t_start < t:
                    raise RuntimeError('Pulse starting before previous pulse '
                                       f'finished {t} s')
                elif t_start > t:
                    # Add gap pulse
                    gap_pulse = DCPulse(t_start=t,
                                        t_stop=t_start,
                                        amplitude=0,
                                        connection=connection)
                    gap_pulses.append(self.get_pulse_implementation(gap_pulse))

                pulse = self.pulse_sequence.get_pulse(t_start=t_start,
                                                      output_channel=ch)
                # Pulse will be None if the pulse sequence has a final delay
                if pulse is not None:
                    t = pulse.t_stop

        if t != self.pulse_sequence.duration:
            for ch in active_channels:
                gap_pulse = DCPulse(t_start=t,
                                    t_stop=self.pulse_sequence.duration,
                                    amplitude=0,
                                    connection=connections[ch])
                gap_pulses.append(self.get_pulse_implementation(gap_pulse))

        if gap_pulses:
            self.pulse_sequence.add(*gap_pulses)

        # TODO test if first waveform needs trigger as well
        additional_pulses = [
            TriggerPulse(t_start=t_start,
                         duration=self.trigger_in_duration(),
                         connection_requirements={
                             'input_instrument': self.instrument_name(),
                             'trigger': True})
            for t_start in self.pulse_sequence.t_start_list + [self.pulse_sequence.duration]
        ]

        return additional_pulses

    def setup(self, **kwargs):
        assert not self.is_primary(), 'AWG520 not programmed as primary instrument'

        self.active_channels(list({pulse.connection.output['channel'].name
                                   for pulse in self.pulse_sequence}))

        # Get waveforms for all channels
        t = 0
        pulses = {ch: self.pulse_sequence.get_pulses(output_channel=ch)
                  for ch in self.active_channels()}
        if len(self.active_channels()) > 1:
            assert len(pulses['ch1']) == len(pulses['ch2']), \
                "Channel1 and channel2 do not have equal number of pulses"

        N_instructions = len(pulses[self.active_channels()[0]])

        waveforms = {ch: [0] * N_instructions for ch in self.active_channels()}
        repetitions = [0] * N_instructions
        for k in range(N_instructions):
            pulse = {ch: pulses[ch][k] for ch in self.active_channels()}
            # Get pulse of first channel
            pulse1 = pulse[self.active_channels()[0]]

            if len(self.active_channels()) > 1:
                assert t == pulse["ch1"].t_start == pulse["ch2"].t_start, \
                    f't={t} does not match pulse.t_start={pulse["ch1"].t_start}'
                assert pulse["ch1"].t_stop == pulse["ch2"].t_stop, \
                    f'pulse["ch1"].t_stop != pulse["ch2"].t_stop: ' \
                    f'{pulse["ch1"].t_stop} != {pulse["ch2"].t_stop}'
            else:
                assert t == pulse1.t_start, "t != pulse.t_start: {t} != {pulse1.t_start}"

            # Waveform points of both channels need to match
            # Here we find out the number of points needed
            pulse_pts = [pulse.implementation.pts for pulse in pulse.values()]
            if None in pulse_pts:
                waveform_pts = int(round((pulse1.t_stop - self.pulse_final_delay() -
                                          pulse1.t_start) * self.sampling_rate()))
            else:
                waveform_pts = int(round(max(pulse_pts)))

            # subtract 0.5 from waveform points to ensure correct length
            t_list = np.arange(pulse1.t_start,
                               pulse1.t_start + (waveform_pts-0.5)/self.sampling_rate(),
                               1/self.sampling_rate())

            if len(t_list) % 4:
                # Points need to be increment of 4
                t_list = t_list[:len(t_list) - (len(t_list) % 4)]
            assert len(t_list) >= 256, \
                f"Waveform has too few points at pulse.t_start={pulse1.t_start}"

            single_repetitions = [0 for _ in self.active_channels()]
            for ch_idx, ch in enumerate(self.active_channels()):
                waveforms[ch][k], single_repetitions[ch_idx] = pulse[ch].implementation.implement(
                    t_list=t_list, sampling_rate=self.sampling_rate())

            if all(repetition == None for repetition in single_repetitions):
                # None of the channel pulses cares about repetitions, set to 1
                repetitions[k] = 1
            else:
                specified_repetitions = {elem for elem in single_repetitions
                                         if elem is not None}
                if len(set(specified_repetitions)) > 1:
                    raise RuntimeError(f'Pulses {pulse} require different number'
                                       f'of repetitions: {single_repetitions}')
                repetitions[k] = next(iter(specified_repetitions))
            t = pulse1.t_stop

        # Set first point of each waveform to endpoint of previous waveform
        endpoint_voltages = {ch: [waveform[-1] for waveform in ch_waveforms]
                             for ch, ch_waveforms in waveforms.items()}
        for ch in self.active_channels():
            for k, waveform in enumerate(waveforms[ch]):
                waveform[0] = endpoint_voltages[ch][(k-1) % len(waveforms[ch])]

        # Create list of unique waveforms and corresponding sequence
        waveforms_list = []
        sequence = {ch: [] for ch in self.active_channels()}
        for ch in self.active_channels():
            for waveform in waveforms[ch]:
                waveform_idx = arreqclose_in_list(waveform, waveforms_list,
                                                  rtol=-1e-4, atol=1e-4)
                if waveform_idx is None:
                    waveforms_list.append(waveform)
                    waveform_idx = len(waveforms_list) - 1
                sequence[ch].append(waveform_idx)

        self.instrument.stop()
        self.instrument.trigger_mode('ENH')
        self.instrument.trigger_level(1)
        self.instrument.clock_freq(self.sampling_rate())

        for ch in ['ch1', 'ch2']:
            self.instrument[f'{ch}_offset'](0)
            self.instrument[f'{ch}_amp'](2)
            self.instrument[f'{ch}_status']('OFF')
        # for ch in [self.instrument.ch1, self.instrument.ch2]:
        #     ch.offset(0)
        #     ch.amplitude(1)
        #     ch.status('OFF')

        # Create silq folder to place waveforms and sequences in
        self.instrument.change_folder('/silq', create_if_necessary=True)

        total_waveform_points = sum(len(waveform) for waveform in waveforms_list)

        assert total_waveform_points < 3.99e6, \
            f'Too many total waveform points: {total_waveform_points}'

        total_existing_waveform_points = sum(
            len(waveform) for waveform in self.waveform_filenames.values())
        if (total_waveform_points + total_existing_waveform_points > 3.99e6) or \
                len(self.waveform_filenames) > self.max_waveforms:
            logger.info('Deleting existing waveforms from hard disk')
            self.instrument.delete_all_files(root=False)
            self.waveform_filenames.clear()

        waveform_filename_mapping = []
        # Copy waveform filenames because we only want to update the attribute
        # If the setup is complete
        waveform_filenames = copy(self.waveform_filenames)
        for waveform in waveforms_list:
            waveform_idx = arreqclose_in_list(waveform,
                                              waveform_filenames.values(),
                                              rtol=1e-4, atol=1e-4)
            if waveform_idx is None:
                # Waveform has not yet been uploaded
                waveform_idx = len(waveform_filenames)

                # Upload waveform
                filename = f'waveform_{waveform_idx}.wfm'
                marker1 = marker2 = np.zeros(len(waveform))
                self.instrument.send_waveform(waveform,
                                              marker1,
                                              marker2,
                                              filename,
                                              clock=self.sampling_rate())
                waveform_filenames[filename] = waveform
            else:
                filename = list(waveform_filenames)[waveform_idx]
            # Add waveform mapping
            waveform_filename_mapping.append(filename)


        # Upload sequence
        sequence_waveform_names = [
            [waveform_filename_mapping[waveform_idx]
             for waveform_idx in sequence[ch]]
            for ch in self.active_channels()]

        wait_trigger = np.ones(N_instructions)
        goto_one = np.zeros(N_instructions)
        logic_jump = np.zeros(N_instructions)
        if len(self.active_channels()) == 1:
            sequence_waveform_names = sequence_waveform_names[0]
        self.instrument.send_sequence('sequence.seq',
                                      sequence_waveform_names,
                                      repetitions,
                                      wait_trigger,
                                      goto_one,
                                      logic_jump)

        # Check for errors here because else it adds a 50% overhead because it's
        # still busy setting the sequence
        for error in self.instrument.get_errors():
            logger.warning(error)

        self.instrument.set_sequence('sequence.seq')

        self.waveforms = waveforms
        self.waveform_filenames = waveform_filenames
        self.sequence = sequence

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def start(self):
        for ch in self.active_channels():
            self.instrument[f'{ch}_status']('ON')
        # self.instrument.ch1.status('ON')
        # self.instrument.ch2.status('ON')
        self.instrument.start()

        # SLeep for a short time because else it sometimes misses first trigger
        sleep(0.2)

    def stop(self):
        self.instrument.ch1_status('OFF')
        self.instrument.ch2_status('OFF')
        self.instrument.stop()


class DCPulseImplementation(PulseImplementation):
    # Number of points in a waveform (must be at least 256)
    pulse_class = DCPulse
    pts = 256

    def implement(self,
                  t_list: np.ndarray,
                  **kwargs) -> np.ndarray:
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
        waveform = self.pulse.amplitude * np.ones(len(t_list))
        repetitions = None # Repetitions is irrelevant

        return waveform, repetitions


class SinePulseImplementation(PulseImplementation):
    # Number of points in a waveform (must be at least 256)
    pulse_class = SinePulse
    pts = None
    settings = {}

    def implement(self,
                  t_list: np.ndarray,
                  sampling_rate: float,
                  **kwargs) -> np.ndarray:
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
        settings = {**silq.config.properties.get('sine_waveform_settings', {}),
                    **self.settings}
        max_points = settings.pop('max_points', 50000)
        if len(t_list) <= max_points:
            waveform = self.pulse.get_voltage(t_list)
            repetitions = 1
        else:
            duration = np.max(t_list) - np.min(t_list)
            min_points = settings.pop('min_points',
                                      np.ceil(duration / 65536 * sampling_rate))
            use_modified_frequency = settings.pop('use_modified_frequency', False)
            results = pulse_to_waveform_sequence(max_points=max_points,
                                                 frequency=self.pulse.frequency,
                                                 sampling_rate=sampling_rate,
                                                 total_duration=duration,
                                                 min_points=max(256, min_points),
                                                 sample_points_multiple=4,
                                                 **settings
                                                 )
            # Store results for debugging purposes
            self.results = results

            if use_modified_frequency:
                # Temporarily change frequency to modified value
                original_frequency = self.pulse.frequency
                self.pulse.frequency = results['optimum']['modified_frequency']

            t_list_segment = t_list[:results['optimum']['points']]
            waveform = self.pulse.get_voltage(t_list_segment)
            repetitions = results['optimum']['repetitions']

            if use_modified_frequency:
                # Revert frequency to original
                self.pulse.frequency = original_frequency

        return waveform, repetitions
