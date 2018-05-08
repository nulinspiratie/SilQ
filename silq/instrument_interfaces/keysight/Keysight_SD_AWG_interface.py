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


class Keysight_SD_AWG_Interface(InstrumentInterface):
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

        # By default run in cyclic mode
        self.cyclic_mode = True
        self.trigger_thread = None

        self.pulse_implementations = [
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
        else: # Hardware trigger
            t_start = min(self.pulse_sequence.t_start_list)
            if self.input_pulse_sequence.get_pulses(trigger=True, t_start=t_start):
                logger.warning(f'Trigger manually defined for {self.name}.')
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
        self.instrument.flush_waveforms()

        self.setup_trigger()

        self.waveforms = self.create_waveforms(error_threshold=error_threshold)

        self.load_waveforms(self.waveforms)

    def setup_trigger(self):
        """Sets up triggering of the AWG.

        Triggering setup is only necessary if trigger_mode == 'hardware'.
        If the AWG is setup as the primary instrument, the trigger_mode cannot
        be hardware, and so will be reset to 'software' here.
        """
        if self.is_primary() and self.trigger_mode() not in ['software', 'none']:
            logger.warning('AWG.trigger_mode must be software or none when '
                           'configured as primary instrument. setting to '
                           'software')
            self.trigger_mode('software')

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

            assert trigger_source in ['trig_in'] + self._pxi_channels.keys(), \
                f"Trigger source {trigger_source} not allowed."

            self.active_instrument_channels.trigger_source(trigger_source)
            self.active_instrument_channels.trigger_mode('rising')

            if trigger_source == 'trig_in':
                self.instrument.trigger_direction('in')

    def setup_waveforms(self, error_threshold):
        waveforms = {ch: [] for ch in self.channel_selection()}

        # Collect all the pulse implementations
        for pulse in self.pulse_sequence:
            channel_waveforms = pulse.implementation.implement(
                instrument=self.instrument,
                default_sampling_rates=self.default_sampling_rates(),
                threshold=error_threshold)

            for ch in channel_waveforms:
                waveforms[ch] += channel_waveforms[ch]

        # Sort the list of waveforms for each channel and calculate delays or
        # throw error on overlapping waveforms.
        for ch_idx, ch in enumerate(self.channel_selection()):
            waveforms[ch] = sorted(waveforms[ch], key=lambda k: k['t_start'])
            global_t_start = min(pulse.t_start for pulse in self.pulse_sequence)

            default_sampling_rate = self.default_sampling_rates()[ch_idx]
            prescaler = 0 if default_sampling_rate == 500e6 else 100e6 / default_sampling_rate

            insert_points = []
            for k, waveform in enumerate(waveforms[ch]):
                if k == 0:
                    # delay_duration = wf['t_start'] - global_t_start
                    delay_duration = waveform['t_start']
                    # print(wf['name'], wf['t_start'])
                    # print('global_t_start', global_t_start)
                else:
                    delay_duration = waveform['t_start'] - waveforms[ch][k-1]['t_stop']

                # a waveform delay is expressed in tens of ns
                delay = int(round((delay_duration * 1e9) / 10))

                if delay > 6000:
                    # create a zero pulse and keep track of where to insert it later
                    # (as a replacement for the long delay)
                    logger.debug('Delay waveform needed for "{}" : duration {:.3f} s'.format(waveform['name'], delay_duration))
                    zero_waveforms = self.create_zero_waveform(duration=delay_duration,
                                                               prescaler=prescaler)

                    if k == 0:
                        zero_waveforms[0]['name'] = f'padding_pulse[{ch[-1]}]'
                        zero_waveforms[0]['t_start'] = global_t_start
                        if len(zero_waveforms) == 2:
                            zero_waveforms[1]['name'] = f'padding_pulse_tail[{ch[-1]}]'
                            zero_waveforms[1]['t_start'] = waveform['t_start']
                    else:
                        zero_waveforms[0]['name'] = f'padding_pulse[{ch[-1]}]'
                        zero_waveforms[0]['t_start'] = waveforms[ch][k-1]['t_stop']
                        if len(zero_waveforms) == 2:
                            zero_waveforms[1]['name'] = f'padding_pulse_tail[{ch[-1]}]'
                            zero_waveforms[1]['t_start'] = waveform['t_start']

                    insertion = {'index': k, 'waveforms': zero_waveforms}
                    insert_points.append(insertion)
                    waveform['delay'] = 0
                else:
                    waveform['delay'] = delay

            # Add final waveform, should fill in space to the end of the whole pulse sequence.
            zero_waveforms = self.create_zero_waveform(
                    duration=self.pulse_sequence.duration - waveform['t_stop'],
                    prescaler=prescaler)
            # Only insert when a waveform is needed
            if zero_waveforms is not None:
                zero_waveforms[0]['name'] = f'padding_pulse[{ch[-1]}]'
                zero_waveforms[0]['t_start'] = waveform['t_stop']
                if len(zero_waveforms) == 2:
                    zero_waveforms[1]['name'] = f'padding_pulse_tail[{ch[-1]}]'
                    zero_waveforms[1]['t_start'] = self.pulse_sequence.duration
                duration = self.pulse_sequence.duration - waveform['t_stop']
                logger.info(f'Adding a final delay waveform to {ch} for ' \
                            f'{duration}s following {waveform["name"]}')
                insertion = {'index': k+1, 'waveforms': zero_waveforms}
                insert_points.append(insertion)

            insert_points = sorted(insert_points, key=lambda k: k['index'], reverse=True)

            for insertion in insert_points:
                k = insertion['index']
                waveforms[ch][k:k] = insertion['waveforms']

    def load_waveforms(self, waveforms, waveform_counter=0):
        for ch in sorted(waveforms):
            ch_idx = self._channels[ch].id
            self.instrument.channels[ch_idx].flush_waveforms()
            self.instrument.set_channel_wave_shape(wave_shape=6, channel_number=ch_idx)
            self.instrument.set_channel_amplitude(amplitude=1.5, channel_number=ch_idx)
            waveform_array = waveforms[ch]

            ch_wf_counter = 0
            message = f'\n{ch} AWG Waveforms:\n'
            total_samples = 0
            # always play a priming pulse first
            next_wf_trigger = True
            for waveform in waveform_array:
                # logger.debug('loading waveform-object {} in M3201A with waveform id {}'.format(id(waveform['waveform']),
                #                                                                         waveform_counter))

                self.instrument.load_waveform(waveform['waveform'], waveform_counter)
                # import pdb; pdb.set_trace()
                if waveform['t_start'] < 0:
                    trigger_mode = 0
                elif next_wf_trigger:
                    next_wf_trigger = False
                    # await trigger for first wf if trigger mode
                    if self.trigger_mode() == 'hardware':
                        trigger_mode = 2
                    elif self.trigger_mode() == 'software':
                        trigger_mode = 1
                    else:
                        trigger_mode = 0
                else:
                    trigger_mode = 0  # auto trigger for every wf that follows

                self.instrument.channels[ch_idx].queue_waveform(
                    waveform_number=waveform_counter,
                    trigger_mode=trigger_mode,
                    start_delay=0,
                    cycles=int(waveform['cycles']),
                    prescaler=int(waveform.get('prescaler', 0)))
                waveform_counter += 1
                ch_wf_counter += 1

                message += f'\t{waveform.get("name"): <20}' + \
                           f'\tpoints = {int(waveform.get("points",-1)) : <20}' + \
                           f'\tcycles = {int(waveform.get("cycles", -1)): <20}\n'
                message += f'\t{" "* 20}' \
                           f'\tprescaler = {int(waveform.get("prescaler", -1)): <20}' + \
                           f'delay  = {int(waveform.get("delay", -1)): <20}' + \
                           f'trigger_mode = {trigger_mode}\n'
                total_samples += int(waveform.get("points", 0) * waveform.get("cycles", 0))
            sample_rate = self.default_sampling_rates()[ch_idx]
            duration = total_samples / sample_rate

            message += f'\tTotal samples = {total_samples} = {duration*1e3} ms @ {sample_rate/1e6} MSPS'
            logger.debug(message)

            self.instrument.awg.AWGqueueConfig(nAWG=self._channels[ch].id, mode=self.cyclic_mode)

    def start(self):
        """Start selected channels, and auto-triggering if primary instrument

        Auto-triggering is performed by creating a triggering thread
        """
        for channel in self.active_instrument_channels:
            channel.wave_shape('arbitrary')

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

    def create_zero_waveform(self, duration, prescaler):
        # TODO: Check if right
        waveform_multiple = 5
        waveform_minimum = 15  # the minimum size of a waveform

        sampling_rate = 500e6 if prescaler == 0 else 100e6 / prescaler

        if (duration < waveform_minimum / sampling_rate):
            return None

        period_sample = 1 / sampling_rate

        period = period_sample * waveform_multiple
        cycles = int(duration // period)
        if (cycles < 1):
            return None

        n = int(np.ceil(cycles / 2 ** 16))

        if n < 3:
            samples = first_factor_above_N(
                int(round(duration / period_sample)),
                waveform_minimum, waveform_multiple)
        else:
            samples = n * waveform_multiple

        cycles = int(duration // (period_sample * samples))

        waveform_repeated_period = period_sample * samples
        waveform_repeated_cycles = cycles
        waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

        waveform_tail_samples = waveform_multiple * int(round(
            ((duration - waveform_repeated_duration) / period_sample) / waveform_multiple))

        if waveform_tail_samples < waveform_minimum:
            waveform_tail_samples = 0

        waveform_repeated = {}
        waveform_tail = {}

        waveform_repeated_data = np.zeros(samples)

        waveform_repeated['waveform'] = self.instrument.new_waveform_from_double(waveform_type=0,
                                                                          waveform_data_a=waveform_repeated_data)
        waveform_repeated['name'] = 'zero_pulse'
        waveform_repeated['points'] = samples
        waveform_repeated['cycles'] = waveform_repeated_cycles
        waveform_repeated['samples'] = samples
        waveform_repeated['delay'] = 0
        waveform_repeated['prescaler'] = prescaler
        logger.debug(f'Delay waveform attrs: '
                     f'cyc={waveform_repeated_cycles} len={samples} n={n}')
        if waveform_tail_samples == 0:
            return [waveform_repeated]
        else:
            waveform_tail_data = np.zeros(waveform_tail_samples)

            waveform_tail['waveform'] = self.instrument.new_waveform_from_double(
                waveform_type=0, waveform_data_a=waveform_tail_data)
            waveform_tail['name'] = 'zero_pulse_tail'
            waveform_tail['points'] = waveform_tail_samples
            waveform_tail['cycles'] = 1
            waveform_tail['delay'] = 0
            waveform_tail['prescaler'] = prescaler

            return [waveform_repeated, waveform_tail]


class SinePulseImplementation(PulseImplementation):
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
    waveform_multiple = 5
    waveform_minimum = 15  # the minimum size of a waveform
    def implement(self, instrument, default_sampling_rates, threshold):
        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.pulse.connection))

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
