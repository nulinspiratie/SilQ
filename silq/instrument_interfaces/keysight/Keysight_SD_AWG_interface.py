from typing import List
from time import sleep
import numpy as np
from threading import Timer
import logging

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import Pulse, SinePulse, PulseImplementation, TriggerPulse, \
    AWGPulse, CombinationPulse, DCPulse, DCRampPulse, MarkerPulse
from silq.meta_instruments.layout import SingleConnection
from silq.tools.pulse_tools import pulse_to_waveform_sequence

from qcodes import validators as vals


logger = logging.getLogger(__name__)


def first_factor_above_N(x, N, step):
    for i in range(N, x, step):
        if not x % i:
            return i
    # If no factor found, just use N
    return N


def find_approximate_divisor(N: int,
                             max_cycles: int = 65535,
                             points_multiple: int = 1,
                             min_points: int = 15,
                             max_points: int = 6000,
                             max_remaining_points: int = 1000) -> Union[tuple, None]:
    """Find an approximate divisor for a number

    The divisor (points) is chosen such that points * cycles <= N, with
    cycles as close as possible to max_cycles, with a low number of remaining
    points

    Args:
        N: Number for which to find a divisor
        max_cycles: Maximum number of cycles (for points * cycles)
        points_multiple: Optional value that points must be a multiple of
        max_points
        max_remaining_points: Maximum number of remaining points.
            Set to 0 to find an exact divisor

    Returns:
        If successful, a dict containing {'points', 'cycles', 'remaining_points'}
        If unsuccessful, None
    """
    # Maximum cycles shouldn't be higher than N/points_multiple
    max_cycles = min(max_cycles, int(N/points_multiple))
    for cycles in range(max_cycles, 0, -1):
        # Find points floor such that points*cycles <= N,
        # but is as close to N as possible
        points = N // cycles
        # Ensure points is always a multiple of points_multiple
        points -= points % points_multiple
        remaining_points = N - points * cycles
        if min_points <= points <= max_points & \
                remaining_points <= max_remaining_points:
            return int(points), int(cycles), int(remaining_points)
    else:
        return None


"""
AWG behaviour:
# Output voltage
The waveform points are between -1 and 1, which correspond to the minimum, 
maximum voltages, respectively.
When waiting for a trigger to proceed to the next waveform, the output voltage
of the final point in the previous waveform is maintained. If there is no 
previous waveform, the voltage is zero. Similar for a waveform start_delay 
(see below).
Note that this can cause odd behaviour, namely that before the waveform queue is 
played for the first time, the voltage is 0V, but after the queue has finished,
the output voltage will be the that of the last point. This may affect the pulse
sequence, if not handled correctly.

# Start delay:
The start delay is time between when a waveform should start and when it 
actually starts. If the waveform requires a trigger, it will wait for 
waveform_delay after it received a trigger.
If the waveform doesn't need a trigger, it will still wait for start_delay after
the previous waveform finished.
During start_delay, the output voltage is the last point of the previous 
waveform. If there is no previous waveform, output voltage is zero.
Start delay is independent of prescaler!

# Trigger
Triggers are ignored if the system is not waiting for one. It is therefore 
important to have some small dead-time built-in before the AWG receives the 
trigger to ensure that the AWG is waiting for one.
"""


class Keysight_SD_AWG_Interface(InstrumentInterface):
    waveform_multiple = 5
    waveform_minimum = 15

    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'ch{k}', id=k, output=True)
            for k in range(self.instrument.n_channels)}

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
            # SinePulseImplementation(
            #     pulse_requirements=[('frequency', {'min': 0, 'max': 200e6}),
            #                         ('amplitude', {'max': 1.5})]),
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
                           vals=vals.Lists,
                           get_cmd=self._get_active_channel_names)

        self.add_parameter('default_sampling_rates', set_cmd=None,
                           initial_value=[500e6] * self.instrument.n_channels)

        self.add_parameter('trigger_mode',
                           set_cmd=None,
                           vals=vals.Enum('none', 'hardware', 'software'),
                           docstring='Selects the method to run through the AWG queue.')

        self.trigger_thread = None
        self.waveforms = None
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
            t_start = min(self.pulse_sequence.t_start_list)
            if self.input_pulse_sequence.get_pulses(trigger=True):
                logger.warning(f'Trigger(s) manually defined for {self.name}: '
                               f'{self.input_pulse_sequence.get_pulses(trigger=True)}')
                return []

            trigger_connection = next(
                (connection.trigger or connection.trigger_start) and
                connection.input['instrument'] == self.instrument_name()
                for connection in connections)

            if trigger_connection.trigger:
                # Add a single trigger pulse when starting sequence
                logger.debug(f'Creating trigger for Keysight SD AWG: {self.name}')
                return [TriggerPulse(name=f'{self.name}_trigger',
                                     t_start=t_start,
                                     duration=15e-6,
                                     connection_requirements={
                                         'input_instrument': self.instrument_name(),
                                         'trigger': True})]
            else:  # trigger_connection.trigger_start
                # Add a single trigger pulse at t=0, a trigger delay until the
                # start of first pulse is configured for the first waveform
                logger.debug(f'Requesting trigger at t=0 (connection.trigger_start)')
                return [TriggerPulse(name=f'{self.name}_trigger',
                                     t_start=0,
                                     duration=15e-6,
                                     connection_requirements={
                                         'input_instrument': self.instrument_name(),
                                         'trigger_start': True})]

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

        self.waveforms = self.create_waveforms(error_threshold=error_threshold)

        self.load_waveforms(self.waveforms)

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
        waveforms = {ch: [] for ch in self.channel_selection()}

        # Sort the list of waveforms for each channel and calculate delays or
        # throw error on overlapping waveforms.
        for channel in self.active_instrument_channels:
            sampling_rate = self.default_sampling_rates()[channel.id]
            prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate

            # Handle delays between waveforms
            t = 0 # TODO: what if trigger_start? t should depend on input trigger pulse
            total_samples = 0 # TODO ensure properly updated
            # TODO: ensure total_samples is always an int
            for pulse in self.pulse_sequence.get_pulses(output_channel=channel.name):
                # TODO: pulse implementation should return single channel only
                pulse_waveforms = pulse.implementation.implement(
                    instrument=self.instrument,
                    default_sampling_rates=self.default_sampling_rates(),
                    threshold=error_threshold)

                for waveform in pulse_waveforms:
                    start_samples = int(round((waveform['t_start'] * 1e9) / 10))
                    delay_samples = start_samples - total_samples

                    # Handle any delay between pulse and previous pulse
                    if delay_samples <= 6000:
                        waveform['delay'] = delay_samples
                    else:  # Delay too long -> separate waveform needed at 0V
                        logger.debug(f'Delay waveform needed for {waveform["name"]}'
                                     f': duration {delay_samples-duration:.3f} s')
                        zero_waveform = self.create_zero_waveform(samples=delay_samples,
                                                                  prescaler=prescaler,
                                                                  t_start=t)
                        waveforms[channel.name].append(zero_waveform)
                        total_samples += zero_waveform['delay'] + zero_waveform['points']

                        waveform['delay'] = min(start_samples - total_samples, 0)

                    waveforms[channel.name].append(waveform)

                    t = waveform['t_stop']

            if self.trigger_mode() != 'hardware':
                # Final 0V waveform may be necessary until end of pulse sequence
                end_samples = int(round((self.pulse_sequence.duration * 1e9) / 10))
                delay_samples = end_samples - total_samples
                zero_waveform = self.create_zero_waveform(t_start=t,
                                                          samples=delay_samples,
                                                          prescaler=prescaler)
                if zero_waveform is not None:
                    waveforms[channel.name].append(zero_waveform)
        return waveforms

    def load_waveforms(self, waveforms):
        self.instrument.flush_waveforms()

        for channel in self.active_instrument_channels:
            channel_waveforms = waveforms[channel.name]

            # always play a priming pulse first
            trigger_mode = ['none', 'software', 'hardware'].index(self.trigger_mode())
            for waveform_idx, waveform in enumerate(channel_waveforms):
                self.instrument.load_waveform(waveform['waveform'], waveform_idx)

                channel.queue_waveform(waveform_number=waveform_idx,
                                       trigger_mode=trigger_mode,
                                       start_delay=0,
                                       cycles=waveform['cycles'],
                                       prescaler=waveform['prescaler'])
                trigger_mode = 0  # auto trigger for every wf that follows first

    def start(self):
        """Start selected channels, and auto-triggering if primary instrument

        Auto-triggering is performed by creating a triggering thread
        """

        self.instrument.start_channels(self.active_channel_ids)
        if self.trigger_mode() == 'software':
            self.started = True
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

    def create_zero_waveform(self, samples, prescaler, t_start):
        # TODO: Test if startDelay actually sets to zero.
        if samples < self.waveform_minimum:
            return None

        # Determine optimal combination of points & cycles for zero pulse
        # optimal means minimal points and remaining_points
        # Remaining points become the waveform start_delay
        # Iteratively loosen constraints until an appropriate divisor is found
        for (max_points, max_remaining_points) in [(2000, 0), (2000, 10),
                                                   (2000, 100), (10000, 1000)]:
            approximate_divisor = find_approximate_divisor(
                N=samples, max_cycles=2**16,
                points_multiple=self.waveform_multiple,
                min_points=self.waveform_minimum,
                max_points=max_points,
                max_remaining_points=max_remaining_points)

            if approximate_divisor is not None:
                points, cycles, remaining_points = approximate_divisor
                break
        else:
            logger.warning('Could not find suitable points, cycles for 0V '
                           f'pulse with {samples} points. Using single cycle '
                           f'with {samples} points, which may be very long')
            points = samples - samples % self.waveform_multiple
            cycles = 1
            remaining_points = samples % self.waveform_multiple

        waveform = {
            'name': 'zero_pulse',
            'waveform': self.instrument.new_waveform_from_double(
                waveform_type=0, waveform_data_a=np.zeros(samples)),
            'points': points,
            'cycles': cycles,
            'delay': remaining_points,
            'prescaler': prescaler,
            't_start': t_start
        }
        return waveform


class SinePulseImplementation(PulseImplementation):
    # TODO fix sinepulseimplementation by using pulse_to_waveform_sequence
    pulse_class = SinePulse

    def implement(self, instrument, default_sampling_rates, threshold):
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

        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.pulse.connection))
        waveforms = {}

        full_name = self.pulse.full_name or 'none'

        # channel independent parameters
        duration = self.pulse.duration
        period = 1 / self.pulse.frequency
        cycles = duration // period
        # TODO: maybe make n_max an argument? Or even better: make max_samples a parameter?
        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15  # the minimum size of a waveform


        for ch_idx, ch in enumerate(channels):

            sampling_rate = default_sampling_rates[ch_idx]
            prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
            period_sample = 1 / sampling_rate
            assert self.pulse.frequency < sampling_rate / 2, \
                f'Sine frequency {self.pulse.frequency} is too high for' \
                f'the sampling frequency {sampling_rate}'

            # This factor determines the number of points needed in the waveform
            # as the number of waveform cycles is limited to (2 ** 16 - 1 = 65535)
            n_min = int(np.ceil(cycles / 2**16))

            n, error, waveform_samples = pulse_to_waveform_sequence(duration, self.pulse.frequency,
                                                                    sampling_rate, threshold,
                                                           n_min=n_min, n_max=1000,
                                                           sample_points_multiple=waveform_multiple)

            if waveform_samples < waveform_minimum:
                raise RuntimeError(f'Waveform too short for {full_name}: '
                                   f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                                   f'{waveform_minimum/sampling_rate*1e3:.3f}ms')


            # the first waveform (waveform_repeated) is repeated n times
            # the second waveform is for the final part of the total wave so the total wave looks like:
            #   n_cycles * waveform_repeated + waveform_tail
            # This is done to minimise the number of data points written to the AWG
            waveform_repeated_period = period_sample * waveform_samples
            t_list_1 = np.linspace(self.pulse.t_start,
                                   self.pulse.t_start + waveform_repeated_period,
                                   waveform_samples, endpoint=False)

            waveform_repeated_cycles = cycles // n
            waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

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

            waveform_repeated['waveform'] = instrument.new_waveform_from_double(
                waveform_type=0,
                waveform_data_a=waveform_repeated_data)
            waveform_repeated['name'] = full_name
            waveform_repeated['points'] = waveform_samples
            waveform_repeated['cycles'] = waveform_repeated_cycles
            waveform_repeated['t_start'] = self.pulse.t_start
            waveform_repeated['t_stop'] = waveform_tail_start
            waveform_repeated['prescaler'] = prescaler

            if len(t_list_2) == 0:
                waveforms[ch] = [waveform_repeated]
            else:
                waveform_tail_data = self.pulse.get_voltage(t_list_2) / 1.5

                waveform_tail['waveform'] = instrument.new_waveform_from_double(
                    waveform_type=0,
                    waveform_data_a=waveform_tail_data)
                waveform_tail['name'] = full_name + '_tail'
                waveform_tail['points'] = waveform_tail_samples
                waveform_tail['cycles'] = 1
                waveform_tail['t_start'] = waveform_tail_start
                waveform_tail['t_stop'] = self.pulse.t_stop
                waveform_tail['prescaler'] = prescaler

                waveforms[ch] = [waveform_repeated, waveform_tail]

        return waveforms


class DCPulseImplementation(PulseImplementation):
    pulse_class = DCPulse

    def implement(self, instrument, default_sampling_rates, threshold):
        assert isinstance(self.pulse.connection, SingleConnection), \
            "DC pulse can only be implemented for SingleConnection. " \
            f"Connection: {self.pulse.connection}"
        channel = self.pulse.connection.output['channel'].name

        waveforms = {}

        full_name = self.pulse.full_name or 'none'

        # channel independent parameters
        duration = self.pulse.t_stop - self.pulse.t_start

        for ch_idx, ch in enumerate(channels):

            sampling_rate = default_sampling_rates[ch_idx]
            prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
            period_sample = 1 / sampling_rate

            period = period_sample * self.waveform_multiple
            max_cycles = int(duration // period)

            # This factor determines the number of points needed in the waveform
            # as the number of waveform cycles is limited to (2 ** 16 - 1 = 65535)
            n = int(np.ceil(max_cycles / 2 ** 16))
            if n < 3:
                waveform_samples = first_factor_above_N(int(round(duration / period_sample)),
                                                        self.waveform_minimum, self.waveform_multiple)
            else:
                waveform_samples = n * self.waveform_multiple

            cycles = int(duration // (period_sample * waveform_samples))
            if duration + threshold < self.waveform_minimum * period_sample:
                raise RuntimeError(f'Waveform too short for {full_name}: '
                                   f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                                   f'{self.waveform_minimum/sampling_rate*1e3:.3f}ms')

            # the first waveform (waveform_repeated) is repeated n times
            # the second waveform is for the final part of the total wave so the total wave looks like:
            #   n_cycles * waveform_repeated + waveform_tail
            # This is done to minimise the number of data points written to the AWG
            waveform_repeated_period = period_sample * waveform_samples
            t_list_1 = np.linspace(self.pulse.t_start, self.pulse.t_start + waveform_repeated_period, waveform_samples, endpoint=False)
            waveform_repeated_cycles = cycles
            waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

            waveform_tail_start = self.pulse.t_start + waveform_repeated_duration
            waveform_tail_samples = self.waveform_multiple * round(
                ((self.pulse.t_stop - waveform_tail_start) / period_sample) / self.waveform_multiple)

            if waveform_tail_samples < self.waveform_minimum:
                waveform_tail_samples = 0

            t_list_2 = np.linspace(waveform_tail_start,
                                   self.pulse.t_stop,
                                   waveform_tail_samples,
                                   endpoint=True)

            waveform_repeated = {}
            waveform_tail = {}

            waveform_repeated_data = self.pulse.get_voltage(t_list_1) / 1.5

            waveform_repeated['waveform'] = instrument.new_waveform_from_double(
                waveform_type=0,
                waveform_data_a=waveform_repeated_data)
            waveform_repeated['name'] = full_name
            waveform_repeated['points'] = waveform_samples
            waveform_repeated['cycles'] = waveform_repeated_cycles
            waveform_repeated['t_start'] = self.pulse.t_start
            waveform_repeated['t_stop'] = waveform_tail_start
            waveform_repeated['prescaler'] = prescaler

            if waveform_tail_samples == 0:
                waveform_repeated['t_stop'] = self.pulse.t_stop
                waveforms[ch] = [waveform_repeated]
            else:
                waveform_tail_data = self.pulse.get_voltage(t_list_2) / 1.5

                waveform_tail['waveform'] = instrument.new_waveform_from_double(
                    waveform_type=0,
                    waveform_data_a=waveform_tail_data)

                waveform_tail['name'] = full_name + '_tail'
                waveform_tail['points'] = waveform_tail_samples
                waveform_tail['cycles'] = 1
                waveform_tail['t_start'] = waveform_tail_start
                waveform_tail['t_stop'] = self.pulse.t_stop
                waveform_tail['prescaler'] = prescaler

                waveforms[ch] = [waveform_repeated, waveform_tail]

        return waveforms


class DCRampPulseImplementation(PulseImplementation):
    pulse_class = DCRampPulse
    def implement(self, instrument, default_sampling_rates, threshold):
        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.pulse.connection))

        waveforms = {}

        full_name = self.pulse.full_name or 'none'

        # channel independent parameters
        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15

        duration = self.pulse.duration
        for ch_idx, ch in enumerate(channels):

            sampling_rate = default_sampling_rates[ch_idx]
            prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
            period_sample = 1 / sampling_rate

            waveform_samples = waveform_multiple * round(
                (self.pulse.duration / period_sample) / waveform_multiple)
            t_list = np.linspace(self.pulse.t_start, self.pulse.t_stop,
                                 waveform_samples, endpoint=True)

            if duration + threshold < waveform_minimum * period_sample:
                raise RuntimeError(f'Waveform too short for {full_name}: '
                                   f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                                   f'{waveform_minimum/sampling_rate*1e3:.3f}ms')
            waveform_data = self.pulse.get_voltage(t_list) / 1.5

            waveform = {
                'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                waveform_data_a=waveform_data),
                'name': full_name,
                'points':waveform_samples,
                'cycles': 1,
                't_start': self.pulse.t_start,
                't_stop': self.pulse.t_stop,
                'prescaler': prescaler}

            waveforms[ch] = [waveform]

        return waveforms


class AWGPulseImplementation(PulseImplementation):
    pulse_class = AWGPulse
    def implement(self, instrument, default_sampling_rates, threshold):
        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.pulse.connection))

        waveforms = {}

        full_name = self.pulse.full_name or 'none'

        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15

        for ch_idx, ch in enumerate(channels):

            sampling_rate = default_sampling_rates[ch_idx]
            prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
            period_sample = 1 / sampling_rate

            waveform_samples = waveform_multiple * round(
                (self.pulse.duration / period_sample) / waveform_multiple)
            t_list = np.linspace(self.pulse.t_start,
                                 self.pulse.t_stop,
                                 waveform_samples,
                                 endpoint=True)

            waveform_data = self.pulse.get_voltage(t_list) / 1.5

            waveform = {'waveform': instrument.new_waveform_from_double(
                waveform_type=0,
                waveform_data_a=waveform_data),
                        'name':full_name,
                        'cycles': 1,
                        't_start': self.pulse.t_start,
                        't_stop': self.pulse.t_start,
                        'prescaler':prescaler}

            waveforms[ch] = [waveform]

        return waveforms


class CombinationPulseImplementation(PulseImplementation):
    pulse_class = CombinationPulse
    def implement(self, instrument, default_sampling_rates, threshold):
        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.pulse.connection))

        waveforms = {}

        full_name = self.pulse.full_name or 'none'

        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15

        duration = self.pulse.duration

        for ch_idx, ch in enumerate(channels):

            sampling_rate = default_sampling_rates[ch_idx]
            prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
            period_sample = 1 / sampling_rate

            waveform_samples = waveform_multiple * round(
                (duration/ period_sample) / waveform_multiple)

            if duration + threshold < waveform_minimum * period_sample:
                raise RuntimeError(f'Waveform too short for {full_name}: '
                               f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                               f'{waveform_minimum/sampling_rate*1e3:.3f}ms')

            t_list = np.linspace(self.pulse.t_start,
                                 self.pulse.t_stop,
                                 waveform_samples,
                                 endpoint=True)

            waveform_data = self.pulse.get_voltage(t_list) / 1.5

            waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                        waveform_data_a=waveform_data),
                        'name': full_name,
                        'cycles': 1,
                        't_start': self.pulse.t_start,
                        't_stop': self.pulse.t_stop,
                        'prescaler': prescaler}

            waveforms[ch] = [waveform]

        return waveforms


class TriggerPulseImplementation(PulseImplementation):
    pulse_class = TriggerPulse
    @property
    def amplitude(self):
        return 1.0

    def implement(self, instrument, default_sampling_rates, threshold):
        if isinstance(self.pulse.connection, SingleConnection):
            channel = self.pulse.connection.output['channel'].name
        else:
            raise Exception('No implementation for connection {}'.format(self.pulse.connection))

        waveform_multiple = 5
        waveform_minimum = 15

        waveforms = {}

        full_name = self.pulse.full_name or 'none'
        duration = self.pulse.duration


        sampling_rate = default_sampling_rates[int(channel[-1])]
        prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
        period_sample = 1 / sampling_rate

        waveform_samples = waveform_multiple * round(
            (self.pulse.duration / period_sample ) / waveform_multiple)

        if duration + threshold < waveform_minimum * period_sample:
            raise RuntimeError(f'Waveform too short for {full_name}: '
                f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                f'{waveform_minimum/sampling_rate*1e3:.3f}ms')
        t_list = np.linspace(self.pulse.t_start, self.pulse.t_stop, waveform_samples, endpoint=True)

        waveform_data = np.append(self.pulse.get_voltage(t_list[:-1]), [0]) / 1.5
        assert len(waveform_data) == waveform_samples, 'Waveform data length' \
                    f'{len(waveform_data)} does not match needed samples {waveform_samples}'

        waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                    waveform_data_a=waveform_data),
                    'name': full_name,
                    'points': waveform_samples,
                    'cycles': 1,
                    't_start': self.pulse.t_start,
                    't_stop': self.pulse.t_stop,
                    'prescaler': prescaler}

        waveforms[channel] = [waveform]

        return waveforms


class MarkerPulseImplementation(PulseImplementation):
    pulse_class = MarkerPulse
    amplitude = 1.0

    def implement(self, instrument, default_sampling_rates, threshold):
        if isinstance(self.pulse.connection, SingleConnection):
            channel = self.pulse.connection.output['channel'].name
        else:
            raise Exception('No implementation for connection {}'.format(self.pulse.connection))

        waveform_multiple = 5
        waveform_minimum = 15

        waveforms = {}

        full_name = self.pulse.full_name or 'none'
        duration = self.pulse.duration

        sampling_rate = default_sampling_rates[int(channel[-1])]
        prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
        period_sample = 1 / sampling_rate

        # Waveform must have at least waveform_multiple samples
        waveform_samples = waveform_multiple * round(
            (self.pulse.duration / period_sample) / waveform_multiple)
        if duration + threshold < waveform_minimum * period_sample:
            raise RuntimeError(f'Waveform too short for {full_name}: '
                f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                f'{waveform_minimum/sampling_rate*1e3:.3f}ms')
        t_list = np.linspace(self.pulse.t_start, self.pulse.t_stop, waveform_samples, endpoint=True)

        waveform_data = np.append(self.pulse.get_voltage(t_list[:-1]), [0]) / 1.5
        assert len(waveform_data) == waveform_samples, f'Waveform data length' \
                    f'{len(waveform_data)} does not match needed samples {waveform_samples}'
        waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                    waveform_data_a=waveform_data),
                    'name': full_name,
                    'points': waveform_samples,
                    'cycles': 1,
                    't_start': self.pulse.t_start,
                    't_stop': self.pulse.t_stop,
                    'prescaler': prescaler}

        waveforms[channel] = [waveform]

        return waveforms
