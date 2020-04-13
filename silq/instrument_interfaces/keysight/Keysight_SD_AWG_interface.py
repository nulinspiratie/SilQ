from typing import List, Union
from time import sleep
import numpy as np
from threading import Timer
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, SinePulse, PulseImplementation, TriggerPulse, \
    AWGPulse, CombinationPulse, DCPulse, DCRampPulse, MarkerPulse
from silq.tools.pulse_tools import pulse_to_waveform_sequence
from silq.tools.general_tools import find_approximate_divisor
from qcodes.utils.helpers import arreqclose_in_list

from qcodes import validators as vals


logger = logging.getLogger(__name__)


class Keysight_SD_AWG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            f'ch{ch}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{ch}', id=ch, output=True)
            for ch in self.instrument.channel_idxs}

        self._pxi_channels = {
            f'pxi{k}': Channel(instrument_name=self.instrument_name(),
                               name=f'pxi{k}', id=4000 + k,
                               input_trigger=True, output=True, input=True)
            for k in range(self.instrument.n_triggers)}

        self._channels = {
            **self._output_channels,
            **self._pxi_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True,
                               input_TTL=(0, 5.0)),
            'trig_out': Channel(instrument_name=self.instrument_name(),
                                name='trig_out', output_TTL=(0, 3.3))}

        self.pulse_implementations = [
            # TODO fix sinepulseimplementation by using pulse_to_waveform_sequence
            SinePulseImplementation(
                pulse_requirements=[('frequency', {'min': 0, 'max': 200e6}),
                                    ('amplitude', {'max': 1.5})]),
            AWGPulseImplementation(
                pulse_requirements=[]),
            CombinationPulseImplementation(
                pulse_requirements=[]),
            DCPulseImplementation(
                pulse_requirements=[('amplitude', {'min': -1.5, 'max': 1.5})]),
            DCRampPulseImplementation(
                pulse_requirements=[]),
            TriggerPulseImplementation(
                pulse_requirements=[]),
            MarkerPulseImplementation(
                pulse_requirements=[])
        ]

        self.add_parameter('channel_selection',
                           vals=vals.Lists(),
                           get_cmd=self._get_active_channel_names)

        self.add_parameter('default_sampling_rates', set_cmd=None,
                           initial_value=[500e6] * len(self.instrument.channel_idxs))

        self.add_parameter('trigger_mode',
                           set_cmd=None,
                           initial_value='software',
                           vals=vals.Enum('none', 'hardware', 'software'),
                           docstring='Selects the method to run through the AWG queue.')

        self.trigger_thread = None
        self.waveforms = None
        self.waveform_queue = None
        self.started = False

    def _get_active_channel_names(self):
        """Get sorted list of active channels"""
        # First create a set to ensure unique elements
        active_channels = {pulse.connection.output['channel'].name
                           for pulse in self.pulse_sequence}
        return sorted(active_channels)

    @property
    def active_channel_ids(self):
        """Sorted list of active channel id's"""
        # First create a set to ensure unique elements
        active_channel_ids = {pulse.connection.output['channel'].id
                              for pulse in self.pulse_sequence}
        return sorted(active_channel_ids)

    @property
    def active_instrument_channels(self):
        return self.instrument.channels[self.active_channel_ids]

    def stop(self):
        # stop all AWG channels and sets FG channels to 'No Signal'
        self.started = False
        self.instrument.off()

        if self.trigger_thread is not None and self.trigger_thread.is_alive():
            logger.debug('Waiting for trigger thread to close...')
            while self.trigger_thread.is_alive():
                sleep(.1)

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Trigger pulses are requested if trigger mode is hardware.
        Trigger at t_start is requested if there is a connection.trigger.
        Trigger at t=0 is requested if there is a connection.trigger_start/

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses, empty by default.
        """
        if self.is_primary():
            # Instrument does not require triggers
            return []
        elif not self.pulse_sequence:
            # AWG does not output pulses
            return []
        elif self.trigger_mode() in ['software', 'none']:
            return []
        else:  # Hardware trigger
            if self.input_pulse_sequence.get_pulses(trigger=True) \
                    or self.input_pulse_sequence.get_pulses(trigger_start=True):
                logger.warning(f'Trigger(s) manually defined for {self.name}: '
                               f'{self.input_pulse_sequence.get_pulses(trigger=True)}')
                return []

            trigger_connection = next(
                connection for connection in connections
                if (connection.trigger or connection.trigger_start) and
                connection.input['instrument'] == self.instrument_name())

            return [TriggerPulse(name=f'{self.name}_trigger',
                                 t_start=0,
                                 duration=15e-6,
                                 connection=trigger_connection)]

    def setup(self, error_threshold=1e-6, **kwargs):
        # TODO: startdelay of first waveform
        # TODO: Handle sampling rates different from default
        # TODO: think about how to configure queue behaviour (cyclic/one shot for example)

        self.instrument.off()

        for channel in self.active_instrument_channels:
            channel.wave_shape('arbitrary')
            channel.amplitude(1.5)
            channel.queue_mode('cyclic')

        self.setup_trigger()

        self.waveforms, self.waveform_queue = self.create_waveforms(error_threshold)

        self.load_waveforms(self.waveforms)

        self.load_waveform_queue(self.waveform_queue)

        # targeted_pulse_sequence is the pulse sequence that is currently setup
        self.targeted_pulse_sequence = self.pulse_sequence
        self.targeted_input_pulse_sequence = self.input_pulse_sequence

    def setup_trigger(self):
        """Sets up triggering of the AWG.

        Triggering setup is only necessary if trigger_mode == 'hardware'.
        If the AWG is setup as the primary instrument, the trigger_mode cannot
        be hardware, and so will be reset to 'software' here.
        Similarly, if the AWG is not the primary instrument, the trigger_mode
        will be reset to hardware.
        """
        if self.is_primary() and self.trigger_mode() not in ['software', 'none']:
            logger.warning('AWG.trigger_mode must be software or none when '
                           'configured as primary instrument. setting to '
                           'software')
            self.trigger_mode('software')
        elif not self.is_primary() and self.trigger_mode() != 'hardware':
            logger.warning('AWG.trigger_mode must be hardware because the AWG '
                           'is not the primary instrument, and must therefore '
                           'receive external triggers. Setting to hardware')
            self.trigger_mode('hardware')

        # Only hardware mode needs configuration
        if self.trigger_mode() == 'hardware':  # AWG is not primary instrument
            try:
                # First check if there is a triggering connection
                trigger_connection = self.input_pulse_sequence.get_connection(
                    trigger=True)
            except AssertionError:
                # Check if there is a connection that can trigger, but only
                # at the start of the sequence
                trigger_connection = self.input_pulse_sequence.get_connection(
                    trigger_start=True)

            trigger_source = trigger_connection.input['channel'].name

            assert trigger_source in ['trig_in', *self._pxi_channels], \
                f"Trigger source {trigger_source} not allowed."

            self.active_instrument_channels.trigger_source(trigger_source)
            self.active_instrument_channels.trigger_mode('rising')

            if trigger_source == 'trig_in':
                self.instrument.trigger_direction('in')

    def create_waveforms(self, error_threshold):
        waveform_queue = {ch: [] for ch in self.channel_selection()}

        # Sort the list of waveforms for each channel and calculate delays or
        # throw error on overlapping waveforms.
        for channel in self.active_instrument_channels:
            default_sampling_rate = self.default_sampling_rates()[channel.id]
            prescaler = 0 if default_sampling_rate == 500e6 else int(
                100e6 / default_sampling_rate)

            # Handle delays between waveforms
            t = 0
            clock_rate = 100e6
            total_samples = 0  # counted at clock rate
            for pulse in self.pulse_sequence.get_pulses(
                    output_channel=channel.name):
                # TODO: pulse implementation should return single channel only
                assert pulse.t_start >= t, \
                    f"Pulse {pulse} starts {pulse.t_start} < t={t}." \
                    f"This likely means that pulses are overlapping"

                pulse_samples_start = max(
                    int(round(pulse.t_start * clock_rate)),
                    total_samples)

                if pulse.t_start > t:  # Add waveform at 0V
                    logger.info(
                        f'Ch{channel.id}: No pulse defined between t={t} s and next '
                        f'{pulse} (pulse.t_start={pulse.t_start} s), '
                        f'Adding DC pulse at 0V')
                    # Use maximum value because potentially total samples could
                    # be higher than t (rounding errors etc.)
                    samples_start_0V = max(int(round(t * clock_rate)),
                                           total_samples)
                    samples_0V = (pulse_samples_start - samples_start_0V) * default_sampling_rate / clock_rate
                    waveform_0V = self.create_DC_waveform(voltage=0,
                                                          samples=samples_0V,
                                                          prescaler=prescaler,
                                                          t_start=t)
                    if waveform_0V is not None:
                        # Add any potential delay samples after previous pulse
                        waveform_0V['delay'] = max(0, (
                                    samples_start_0V - total_samples) * default_sampling_rate / clock_rate)
                        waveform_queue[channel.name].append(waveform_0V)

                        # Increase total samples to include 0V pulse points
                        total_samples += waveform_0V['delay']
                        total_samples += waveform_0V['points_100MHz'] * \
                                         waveform_0V['cycles']

                pulse_waveforms = pulse.implementation.implement(
                    interface=self,
                    instrument=self.instrument,
                    default_sampling_rate=default_sampling_rate,
                    threshold=error_threshold)

                for waveform in pulse_waveforms:
                    sampling_rate = 500e6 if waveform['prescaler'] == 0 else \
                        100e6 / waveform['prescaler']
                    start_samples = int(round(waveform['t_start'] * clock_rate))

                    waveform['delay'] = max((start_samples - total_samples) * sampling_rate / clock_rate, 0)
                    waveform_queue[channel.name].append(waveform)

                    total_samples += waveform['delay']
                    total_samples += waveform['points_100MHz'] * waveform['cycles']

                t = pulse.t_stop

            if t <= self.pulse_sequence.duration:
                final_samples = int(
                    round(self.pulse_sequence.duration * clock_rate))
                remaining_samples = (final_samples - total_samples) * default_sampling_rate / clock_rate
                waveform_0V = self.create_DC_waveform(voltage=0,
                                                      samples=remaining_samples,
                                                      prescaler=prescaler,
                                                      t_start=t)
                if waveform_0V:
                    waveform_queue[channel.name].append(waveform_0V)

        waveforms = []
        for channel, channel_waveform_queue in waveform_queue.items():
            for waveform_info in channel_waveform_queue:
                waveform = waveform_info.pop('waveform')
                waveform_idx = arreqclose_in_list(waveform,
                                                  waveforms,
                                                  rtol=1e-4, atol=1e-4)
                if waveform_idx is None:  # Add waveform to list
                    waveforms.append(waveform)
                    waveform_idx = len(waveforms) - 1
                waveform_info['idx'] = waveform_idx

        return waveforms, waveform_queue

    def load_waveforms(self, waveforms):
        self.instrument.flush_waveforms()

        for waveform_idx, waveform in enumerate(waveforms):
            waveform_object = self.instrument.new_waveform_from_double(
                waveform_type=0, waveform_data_a=waveform)
            self.instrument.load_waveform(waveform_object, waveform_idx)

    def load_waveform_queue(self, waveform_queue):
        self.instrument.flush_waveforms()

        for channel in self.active_instrument_channels:
            channel_waveforms = waveform_queue[channel.name]

            # always play a priming pulse first
            trigger_mode = ['none', 'software', 'hardware'].index(self.trigger_mode())
            for waveform in channel_waveforms:

                channel.queue_waveform(waveform_number=waveform['idx'],
                                       trigger_mode=trigger_mode,
                                       start_delay=int(waveform['delay']),
                                       cycles=int(waveform['cycles']),
                                       prescaler=int(waveform['prescaler']))
                trigger_mode = 0  # auto trigger for every wf that follows first

    def start(self):
        """Start selected channels, and auto-triggering if primary instrument

        Auto-triggering is performed by creating a triggering thread
        """
        for channel in self.active_instrument_channels:
            channel.wave_shape('arbitrary')
        self.instrument.start_channels(self.active_channel_ids)
        self.started = True
        if self.trigger_mode() == 'software':
            trigger_period = self.pulse_sequence.duration * 1.1
            logger.debug(f'Starting self triggering of the M3201 AWG with '
                         f'interval {trigger_period*1e3:.3f}ms.')
            self.start_auto_trigger(trigger_period)

    def start_auto_trigger(self, trigger_period: float):
        """Starts auto-triggering of AWG, used if trigger_mode == 'software'

        This method first restarts and triggers the AWG, and then starts a
        thread that calls this method again after trigger_period.

        If self.started == False, this method will not do anything, and so no
        thread is started

        Args:
            trigger_period: Period before calling this method again
        """
        if self.started:
            # Restart current waveform queue
            self.instrument.stop_channels(self.active_channel_ids)
            self.instrument.start_channels(self.active_channel_ids)
            self.instrument.trigger_channels(self.active_channel_ids)

            # Create a threaded Timer that retriggers after trigger_period
            self.trigger_thread = Timer(interval=trigger_period,
                                        function=self.start_auto_trigger,
                                        args=[trigger_period])
            self.trigger_thread.start()
        else:
            logger.debug('Not continuing auto-triggering because '
                         'AWG_interface.started == false')

    def create_DC_waveform(self,
                           voltage: int,
                           samples: int,
                           prescaler: int,
                           t_start: float,
                           max_cycles: int = 2**16,
                           final_voltage=None):
        if samples < self.instrument.waveform_minimum:
            return None

        assert -1.5 <= voltage <= 1.5

        # Determine waveform with whose points and cycles satisfy:
        # - Low points (iteratively increasing max_points until match is found)
        # - Points >= waveform_minimum & points % waveform_multiple = 0
        # - At most 2**16 cycles
        # - Remaining points is 6000 (max) divided by prescaler since these
        #   are included in the next pulses start delay
        for max_points in [1000, 3000, 10000, 30000, 100000]:
            approximate_divisor = find_approximate_divisor(
                N=samples, max_cycles=max_cycles,
                points_multiple=self.instrument.waveform_multiple,
                min_points=self.instrument.waveform_minimum,
                max_points=max_points,
                max_remaining_points=int(6000 / (prescaler+1)))

            if approximate_divisor is not None:
                points = approximate_divisor['points']
                cycles = approximate_divisor['cycles']
                break
        else:
            logger.warning('Could not find suitable points, cycles for 0V '
                           f'pulse with {samples} points. Using single cycle '
                           f'with {samples} points, which may be very long')
            logger.warning(dict(
                N=samples, max_cycles=max_cycles,
                points_multiple=self.instrument.waveform_multiple,
                min_points=self.instrument.waveform_minimum,
                max_points=max_points,
                max_remaining_points=int(6000 / prescaler)))
            points = samples - samples % self.instrument.waveform_multiple
            cycles = 1

        waveform_points = voltage / 1.5 * np.ones(points)
        if final_voltage is not None:
            waveform_points[-1] = final_voltage
        waveform = {
            'name': 'zero_pulse',
            'waveform': waveform_points,
            'points': points,
            'points_100MHz': int(points / 5) if prescaler == 0 else points * prescaler,
            'cycles': cycles,
            'delay': 0,
            'prescaler': prescaler,
            't_start': t_start
        }
        return waveform


class SinePulseImplementation(PulseImplementation):
    # TODO fix sinepulseimplementation by using pulse_to_waveform_sequence
    pulse_class = SinePulse

    def implement(self, interface, instrument, default_sampling_rate, threshold):
        """
        This function takes the targeted pulse (i.e. an interface specific pulseimplementation) and converts
        it to a set of pulse-independent instructions/information that can be handled by interface.setup().

        For example::

            SinePulseImplementation()
                initializes a generic (interface independent) pulseimplementation for the sinepulse
            SinePulseImplementation.implement()
                takes the targeted pulse and returns information/instructions that are independent of the pulse type
                (DCPulse, SinePulse, ...) and can be handled by interface.setup() to send instructions to the driver.

        Args:
            instrument (Instrument): the M3201A instrument
            sampling_rates (dict): dictionary containing sampling rates for each channel
            threshold (float): threshold in frequency error in Hz

        Returns:
            waveforms (dict): dictionary containing waveform objects for each channel

        example return value::

            waveforms =
                {
                'ch_1':
                    [
                        {
                            'waveform': waveform1,
                            'cycles': 10,
                            'delay': 0.0
                        },
                        {
                            'waveform': waveform2,
                            'cycles': 1,
                            'delay': 0.0
                        }
                    ]
                }

        """
        # TODO: what is the most useful threshold definition for the user (rel. error/abs. error in period/frequency)
        logger.debug('implementing SinePulse for the M3201A interface')
        # use t_start, t_stop, sampling_rate, ... to make a waveform object that can be queued in interface.setup()
        # basically, each implement in all PulseImplementations will be a waveform factory

        full_name = self.pulse.full_name or 'none'

        # channel independent parameters
        sampling_rate = default_sampling_rate
        duration = self.pulse.duration
        period = 1 / self.pulse.frequency
        cycles = duration // period
        # TODO: maybe make n_max an argument? Or even better: make max_samples a parameter?
        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15  # the minimum size of a waveform


        prescaler = 0 if sampling_rate == 500e6 else \
            int(100e6 / sampling_rate)
        period_sample = 1 / sampling_rate
        assert self.pulse.frequency < sampling_rate/ 2, \
            f'Sine frequency {self.pulse.frequency} is too high for' \
            f'the sampling frequency {sampling_rate}'

        # This factor determines the number of points needed in the waveform
        # as the number of waveform cycles is limited to (2 ** 16 - 1 = 65535)

        results = pulse_to_waveform_sequence(
            max_points=duration*sampling_rate,
            frequency=self.pulse.frequency,
            sampling_rate=sampling_rate,
            frequency_threshold=threshold,
            total_duration=duration,
            min_points=waveform_minimum,
            sample_points_multiple=waveform_multiple)

        waveform_samples = results['optimum']['points']
        waveform_repeated_cycles = results['optimum']['repetitions']

        waveform_repeated_period = period_sample * waveform_samples

        t_list_1 = np.linspace(self.pulse.t_start,
                               self.pulse.t_start + waveform_repeated_period,
                               waveform_samples, endpoint=False)

        waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

        # {'error': 0.0, 'final_delay': 0.0005000000000000004, 'periods': 1,
        #  'repetitions': 12, 'points': 1000, 'idx': (0, 1),
         # 'modified_frequency': 1000.0}

        if waveform_samples < waveform_minimum:
            raise RuntimeError(f'Waveform too short for {full_name}: '
                               f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                               f'{waveform_minimum/sampling_rate*1e3:.3f}ms')

        waveform_tail_start = self.pulse.t_start + waveform_repeated_duration
        waveform_tail_samples = waveform_multiple * round(
            ((self.pulse.t_stop - waveform_tail_start) / period_sample) / waveform_multiple)

        if waveform_tail_samples < waveform_minimum:
            # logger.debug('tail is too short, removing tail (tail size was: {})'.format(waveform_tail_samples))
            waveform_tail_samples = 0

        t_list_2 = np.linspace(waveform_tail_start,
                               self.pulse.t_stop,
                               waveform_tail_samples,
                               endpoint=True)

        waveform_repeated = {}
        waveform_tail = {}

        waveform_repeated_data = self.pulse.get_voltage(t_list_1) / 1.5

        waveform_repeated['waveform'] = waveform_repeated_data
        waveform_repeated['name'] = full_name
        waveform_repeated['points'] = waveform_samples
        waveform_repeated['points_100MHz'] = (int(waveform_samples / 5) if prescaler == 0
                                              else waveform_samples * prescaler)
        waveform_repeated['cycles'] = waveform_repeated_cycles
        waveform_repeated['t_start'] = self.pulse.t_start
        waveform_repeated['t_stop'] = waveform_tail_start
        waveform_repeated['prescaler'] = prescaler

        if len(t_list_2) == 0:
            waveforms = [waveform_repeated]
        else:
            waveform_tail_data = self.pulse.get_voltage(t_list_2) / 1.5

            waveform_tail['waveform'] = waveform_tail_data
            waveform_tail['name'] = full_name + '_tail'
            waveform_tail['points'] = waveform_tail_samples
            waveform_tail['points_100MHz'] = (int(waveform_tail_samples / 5) if prescaler == 0
                                              else waveform_tail_samples * prescaler)
            waveform_tail['cycles'] = 1
            waveform_tail['t_start'] = waveform_tail_start
            waveform_tail['t_stop'] = self.pulse.t_stop
            waveform_tail['prescaler'] = prescaler

            waveforms = [waveform_repeated, waveform_tail]

        return waveforms


class DCPulseImplementation(PulseImplementation):
    pulse_class = DCPulse

    def implement(self, interface, instrument, default_sampling_rate, threshold):
        sampling_rate = default_sampling_rate
        prescaler = 0 if sampling_rate == 500e6 else int(100e6 / sampling_rate)
        samples = int(self.pulse.duration * sampling_rate)
        assert samples >= instrument.waveform_minimum, \
            f"pulse {self.pulse} too short"

        waveform = interface.create_DC_waveform(voltage=self.pulse.amplitude,
                                                samples=samples,
                                                prescaler=prescaler,
                                                t_start=self.pulse.t_start)

        if self.pulse.full_name:
            waveform['name'] = self.pulse.full_name

        return [waveform]


class DCRampPulseImplementation(PulseImplementation):
    pulse_class = DCRampPulse

    def implement(self, interface, instrument, default_sampling_rate, threshold):
        full_name = self.pulse.full_name or 'none'

        sampling_rate = default_sampling_rate
        prescaler = 0 if sampling_rate == 500e6 else int(100e6 / sampling_rate)

        samples = int(self.pulse.duration * sampling_rate)
        samples -= samples % instrument.waveform_multiple
        assert samples >= instrument.waveform_minimum, \
            f"pulse {self.pulse} too short"

        t_list = np.linspace(self.pulse.t_start, self.pulse.t_stop, samples)

        waveform_data = self.pulse.get_voltage(t_list) / 1.5

        waveform = {'waveform': waveform_data,
                    'points': samples,
                    'points_100MHz': (int(samples / 5) if prescaler == 0
                                      else samples * prescaler),
                    'name': full_name,
                    'points': samples,
                    'cycles': 1,
                    't_start': self.pulse.t_start,
                    't_stop': self.pulse.t_stop,
                    'prescaler': prescaler}

        return [waveform]


class AWGPulseImplementation(PulseImplementation):
    pulse_class = AWGPulse

    def implement(self, interface, instrument, default_sampling_rate, threshold):
        full_name = self.pulse.full_name or 'none'

        sampling_rate = default_sampling_rate
        prescaler = 0 if sampling_rate == 500e6 else (100e6 / sampling_rate)

        samples = int(self.pulse.duration * sampling_rate)
        samples -= samples % instrument.waveform_multiple
        assert samples >= instrument.waveform_minimum, \
            f"pulse {self.pulse} too short"

        t_list = np.linspace(self.pulse.t_start, self.pulse.t_stop, samples)

        waveform_data = self.pulse.get_voltage(t_list) / 1.5

        waveform = {'waveform': waveform_data,
                    'points': samples,
                    'points_100MHz': (int(samples / 5) if prescaler == 0
                                      else samples * prescaler),
                    'name':full_name,
                    'cycles': 1,
                    't_start': self.pulse.t_start,
                    't_stop': self.pulse.t_start,
                    'prescaler':prescaler}

        return [waveform]


class CombinationPulseImplementation(PulseImplementation):
    pulse_class = CombinationPulse

    def implement(self, interface, instrument, default_sampling_rate):
        full_name = self.pulse.full_name or 'none'

        sampling_rate = default_sampling_rate
        prescaler = 0 if sampling_rate == 500e6 else int(100e6 / sampling_rate)

        samples = int(self.pulse.duration * sampling_rate)
        samples -= samples % instrument.waveform_multiple
        assert samples >= instrument.waveform_minimum, \
            f"pulse {self.pulse} too short"

        t_list = np.linspace(self.pulse.t_start, self.pulse.t_stop, samples)

        waveform_data = self.pulse.get_voltage(t_list) / 1.5

        waveform = {'waveform': waveform_data,
                    'points': samples,
                    'points_100MHz': (int(samples / 5) if prescaler == 0
                                      else samples * prescaler),
                    'name': full_name,
                    'cycles': 1,
                    't_start': self.pulse.t_start,
                    't_stop': self.pulse.t_stop,
                    'prescaler': prescaler}

        return [waveform]


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse

    def implement(self, interface, instrument, default_sampling_rate, **kwargs):
        # sampling_rate = default_sampling_rate
        # prescaler = 0 if sampling_rate == 500e6 else int(100e6 / sampling_rate)
        logger.info('Trigger pulse implementation overrides default sampling rates and uses 100 MSa/s.')
        sampling_rate = 100e6
        prescaler = 1
        samples = int(self.pulse.duration * sampling_rate)
        if samples < instrument.waveform_minimum:
            logger.warning(f'Trigger pulse {self.pulse} too short, setting to '
                           f'minimum duration of 15 samples')
            samples = 15

        # Set max cycles to 1 since trigger pulses should be very short
        waveform = interface.create_DC_waveform(voltage=self.pulse.amplitude,
                                                samples=samples,
                                                prescaler=prescaler,
                                                t_start=self.pulse.t_start,
                                                max_cycles=1)
                                                # final_voltage=0)
        if self.pulse.full_name:
            waveform['name'] = self.pulse.full_name

        return [waveform]


class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse

    def implement(self, interface, instrument, default_sampling_rate, **kwargs):
        sampling_rate = default_sampling_rate
        prescaler = 0 if sampling_rate == 500e6 else int(100e6 / sampling_rate)
        samples = int(self.pulse.duration * sampling_rate)
        assert samples >= instrument.waveform_minimum, \
            f"pulse {self.pulse} too short"

        waveform = interface.create_DC_waveform(voltage=self.pulse.amplitude,
                                                samples=samples,
                                                prescaler=prescaler,
                                                t_start=self.pulse.t_start)

        if self.pulse.full_name:
            waveform['name'] = self.pulse.full_name

        return [waveform]
