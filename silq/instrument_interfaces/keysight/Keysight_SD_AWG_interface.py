from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation, TriggerPulse, AWGPulse,\
                        CombinationPulse, DCPulse, DCRampPulse, MarkerPulse
from silq.meta_instruments.layout import SingleConnection
from silq.tools.pulse_tools import pulse_to_waveform_sequence

from qcodes import ManualParameter
from qcodes import validators as vals

from functools import partial
import numpy as np

import threading
import logging
logger = logging.getLogger(__name__)


def firstFactorAboveN(x, N, step):
    for i in range(N, x, step):
        if (x % i == 0):
            return i
    # If no factor found, just use N
    return N


class Keysight_SD_AWG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'ch{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='ch{}'.format(k), id=k,
                        output=True)
            for k in range(self.instrument.n_channels)}

        self._pxi_channels = {
            'pxi{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='pxi{}'.format(k), id=4000 + k,
                        input_trigger=True, output=True, input=True)
            for k in range(self.instrument.n_triggers)}

        self._channels = {
            **self._output_channels,
            **self._pxi_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input_trigger=True, input_TTL=(0, 5.0)),
            'trig_out': Channel(instrument_name=self.instrument_name(),
                                name='trig_out', output_TTL=(0, 3.3))}

        # By default run in cyclic mode
        self.cyclic_mode = True
        self.trigger_thread = None

        # TODO: how does the power parameter work? How can I set requirements on the amplitude?
        self.pulse_implementations = [
            SinePulseImplementation(
                pulse_requirements=[('frequency', {'min': 0, 'max': 200e6}),
                                    ('power', {'max': 1.5})]
            ),
            AWGPulseImplementation(
                pulse_requirements=[]
            ),
            CombinationPulseImplementation(
                pulse_requirements=[]
            ),
            DCPulseImplementation(
                pulse_requirements=[('amplitude', {'min': -1.5, 'max': 1.5})]
            ),
            DCRampPulseImplementation(
                pulse_requirements=[]
            ),
            TriggerPulseImplementation(
                pulse_requirements=[]
            ),
            MarkerPulseImplementation(
                pulse_requirements=[]
            )
        ]

        self.add_parameter('active_channels',
                           get_cmd=self._get_active_channels)

        self.add_parameter('default_sampling_rates', set_cmd=None,
                           initial_value=[500e6] * self.instrument.n_channels)

        self.add_parameter('trigger_mode',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('none', 'hardware', 'software'),
                           docstring='Selects the method to run through the AWG queue.')

    # TODO: is this device specific? Does it return [0,1,2] or [1,2,3]?
    def _get_active_channels(self):
        active_channels = [pulse.connection.output['channel'].name
                           for pulse in self.pulse_sequence]
        # Transform into set to ensure that elements are unique
        active_channels = list(set(active_channels))
        return active_channels

    def _get_active_channel_ids(self):
        active_channel_ids = [pulse.connection.output['channel'].id
                              for pulse in self.pulse_sequence]
        # Transform into set to ensure that elements are unique
        active_channel_ids = list(set(active_channel_ids))
        return active_channel_ids

    def stop(self):
        # stop all AWG channels and sets FG channels to 'No Signal'
        self.started = False
        self.instrument.off()
        if (self.trigger_thread != None):
            logger.debug('Waiting for trigger thread to close...')
            while(self.trigger_thread.is_alive()):
                pass
            logger.debug('Done.')

    def setup(self, **kwargs):
        # TODO: Handle sampling rates different from default
        # TODO: figure out how we want to configure error_threshold
        error_threshold = 1e-6
        # TODO: think about how to configure queue behaviour (cyclic/one shot for example)

        # flush the onboard RAM and reset waveform counter
        self.instrument.flush_waveform()
        waveform_counter = 0

        # Configure triggering
        if (self.trigger_mode() == 'hardware'):
            trigger_pulse = self.input_pulse_sequence.get_pulses(trigger=True)[0]
            trigger_connection = trigger_pulse.connection
            # import pdb; pdb.set_trace()

            if not trigger_connection.input['channel'].name in self._channels.keys():
                err = 'Cannot find trigger connection in channels'
                logger.error(err)
                raise RuntimeError(err)

            # Configure for PXI triggers
            if trigger_connection.input['channel'].name in self._pxi_channels.keys():
                for ch in self._get_active_channel_ids():
                    self.instrument.awg_config_external_trigger(ch, trigger_connection.input['channel'].id, 3)
            # Otherwise use external trigger port
            else:
                for ch in self._get_active_channel_ids():
                    self.instrument.awg_config_external_trigger(ch, 0, 3)
                self.instrument.config_trigger_io(1, 1)
        elif self.trigger_mode() == 'software':
            pass
        elif self.trigger_mode() == 'none':
            pass

        # for each pulse:
        #   - implement
        # for each channel:
        #   - load waveforms
        #   - queue waveforms

        waveforms = dict()

        for pulse in self.pulse_sequence:
            channel_waveforms = pulse.implementation.implement(
                instrument=self.instrument,
                default_sampling_rates=self.default_sampling_rates(),
                threshold=error_threshold)

            for ch in channel_waveforms:
                if ch in waveforms:
                    waveforms[ch] += channel_waveforms[ch]
                else:
                    waveforms[ch] = channel_waveforms[ch]

        # Sort the list of waveforms for each channel and calculate delays or throw error on overlapping waveforms.
        for ch_idx, ch in enumerate(sorted(self._get_active_channels())):
            waveforms[ch] = sorted(waveforms[ch], key=lambda k: k['t_start'])
            global_t_start = min(pulse.t_start for pulse in self.pulse_sequence)

            default_sampling_rate = self.default_sampling_rates()[ch_idx]
            prescaler = 0 if default_sampling_rate == 500e6 else 100e6 / default_sampling_rate

            insert_points = []
            for i, wf in enumerate(waveforms[ch]):
                if i == 0:
                    # delay_duration = wf['t_start'] - global_t_start
                    delay_duration = wf['t_start']
                    # print(wf['name'], wf['t_start'])
                    # print('global_t_start', global_t_start)
                else:
                    delay_duration = wf['t_start'] - waveforms[ch][i-1]['t_stop']

                # a waveform delay is expressed in tens of ns
                delay = int(round((delay_duration * 1e9) / 10))

                if delay > 6000:
                    # create a zero pulse and keep track of where to insert it later
                    # (as a replacement for the long delay)
                    logger.debug('Delay waveform needed for "{}" : duration {:.3f} s'.format(wf['name'], delay_duration))
                    zero_waveforms = self.create_zero_waveform(duration=delay_duration,
                                                               prescaler=prescaler)

                    if i == 0:
                        zero_waveforms[0]['name'] = f'padding_pulse[{ch[-1]}]'
                        zero_waveforms[0]['t_start'] = global_t_start
                        if len(zero_waveforms) == 2:
                            zero_waveforms[1]['name'] = f'padding_pulse_tail[{ch[-1]}]'
                            zero_waveforms[1]['t_start'] = wf['t_start']
                    else:
                        zero_waveforms[0]['name'] = f'padding_pulse[{ch[-1]}]'
                        zero_waveforms[0]['t_start'] = waveforms[ch][i-1]['t_stop']
                        if len(zero_waveforms) == 2:
                            zero_waveforms[1]['name'] = f'padding_pulse_tail[{ch[-1]}]'
                            zero_waveforms[1]['t_start'] = wf['t_start']

                    insertion = {'index': i, 'waveforms': zero_waveforms}
                    insert_points.append(insertion)
                    wf['delay'] = 0
                else:
                    wf['delay'] = delay

            # Add final waveform, should fill in space to the end of the whole pulse sequence.
            zero_waveforms = self.create_zero_waveform(
                    duration=self.pulse_sequence.duration - wf['t_stop'],
                    prescaler=prescaler)
            # Only insert when a waveform is needed
            if zero_waveforms is not None:
                zero_waveforms[0]['name'] = f'padding_pulse[{ch[-1]}]'
                zero_waveforms[0]['t_start'] = wf['t_stop']
                if len(zero_waveforms) == 2:
                    zero_waveforms[1]['name'] = f'padding_pulse_tail[{ch[-1]}]'
                    zero_waveforms[1]['t_start'] = self.pulse_sequence.duration
                duration = self.pulse_sequence.duration - wf['t_stop']
                logger.info(f'Adding a final delay waveform to {ch} for ' \
                            f'{duration}s following {wf["name"]}')
                insertion = {'index': i+1, 'waveforms': zero_waveforms}
                insert_points.append(insertion)

            insert_points = sorted(insert_points, key=lambda k: k['index'], reverse=True)

            for insertion in insert_points:
                i = insertion['index']
                waveforms[ch][i:i] = insertion['waveforms']


            # message = f'\n{ch} AWG Waveforms:\n'
            # total = 0
            # for wf in waveforms[ch]:
            #     message += f'\t{wf.get("name"): <20}' + \
            #                f'\tpoints = {int(wf.get("points",-1)) : <20}' + \
            #                f'\tcycles = {int(wf.get("cycles", -1)): <20}\n'
            #     message += f'\t{" "* 20}' \
            #                f'\tprescaler = {int(wf.get("prescaler", -1)): <20}' + \
            #                f'\tdelay     = {int(wf.get("delay", -1)): <20}\n'
            #     total += int(wf.get("points", 0) * wf.get("cycles", 0))
            # message += f'\tTotal samples = {total} = {total/500e3}ms @ 500 MSPS'
            # logger.debug(message)


        self.instrument.off()
        self.instrument.flush_waveform()

        for ch in sorted(waveforms):
            self.instrument.awg_flush(self._channels[ch].id)
            self.instrument.set_channel_wave_shape(wave_shape=6, channel_number=self._channels[ch].id)
            self.instrument.set_channel_amplitude(amplitude=1.5, channel_number=self._channels[ch].id)
            waveform_array = waveforms[ch]

            ch_wf_counter = 0
            message = f'\n{ch} AWG Waveforms:\n'
            total = 0
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

                self.instrument.awg_queue_waveform(self._channels[ch].id, waveform_counter, trigger_mode,
                                                   0, int(waveform['cycles']), prescaler=int(waveform.get('prescaler', 0)))
                waveform_counter += 1
                ch_wf_counter += 1



                message += f'\t{waveform.get("name"): <20}' + \
                           f'\tpoints = {int(waveform.get("points",-1)) : <20}' + \
                           f'\tcycles = {int(waveform.get("cycles", -1)): <20}\n'
                message += f'\t{" "* 20}' \
                           f'\tprescaler = {int(waveform.get("prescaler", -1)): <20}' + \
                           f'delay  = {int(waveform.get("delay", -1)): <20}' + \
                           f'trigger_mode = {trigger_mode}\n'
                total += int(waveform.get("points", 0) * waveform.get("cycles", 0))

            message += f'\tTotal samples = {total} = {total/500e3}ms @ 500 MSPS'
            logger.debug(message)

            self.instrument.awg.AWGqueueConfig(nAWG=self._channels[ch].id, mode=self.cyclic_mode)

        # for ch in waveforms:
        #     waveform_array = waveforms[ch]
        #     for wf in waveform_array:
        #         del(wf['waveform'])

    def start(self):
        mask = 0
        for c in self._get_active_channel_ids():
            self.instrument.set_channel_wave_shape(wave_shape=6, channel_number=c)
            mask |= 1 << c

        self.instrument.awg_start_multiple(mask)
        if self.trigger_mode() == 'software':
            self.started = True
            duration = self.pulse_sequence.duration
            trigger_period = duration * 1.1
            logger.debug(f'Starting self triggering of the M3201 AWG with interval {trigger_period*1100:.3f}ms.')
            self.trigger_self(trigger_period)

    def trigger_self(self, trigger_period):
        self.software_trigger()
        if self.started:
            self.trigger_thread = threading.Timer(trigger_period, partial(self.trigger_self, trigger_period))
            self.trigger_thread.start()

    def get_additional_pulses(self, **kwargs):
        if self.is_primary():
            # Instrument does not require triggers
            return []

        if not self.pulse_sequence:
            # Instrument not needed
            return []
        else:
            if (self.trigger_mode() == 'hardware'):
                # Add a single trigger pulse when starting sequence
                t_start = min(self.pulse_sequence.t_start_list)
                if (self.input_pulse_sequence.get_pulses(trigger=True,
                                                         t_start=t_start)):
                    logger.info(
                        f'Trigger manually defined for Keysight SD AWG : {self.name}.')
                    return []
                logger.info(f'Creating trigger for Keysight SD AWG: {self.name}')
                trigger_pulse = \
                    TriggerPulse(name=self.name + '_trigger', t_start=t_start, duration=15e-6,
                                 connection_requirements={
                                     'input_instrument': self.instrument_name(),
                                     'trigger': True
                                 })
                return [trigger_pulse]
            elif self.trigger_mode() == 'software':
                # No extra pulses needed
                return []
            elif self.trigger_mode() == 'none':
                # No extra pulses needed
                return []

    def write_raw(self, cmd):
        pass

    def ask_raw(self, cmd):
        pass

    def software_trigger(self):
        mask = 0
        for c in self._get_active_channel_ids():
            mask |= 1 << c
        self.instrument.awg_stop_multiple(mask)
        self.instrument.awg_start_multiple(mask)
        self.instrument.awg_trigger_multiple(mask)

    def create_zero_waveform(self, duration, prescaler):
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
            samples = firstFactorAboveN(
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
        logger.debug('Delay waveform attrs: cyc={cycles} len={samples} n={n}'.format(cycles=waveform_repeated_cycles,
                                                                                      samples=samples, n=n))
        if waveform_tail_samples == 0:
            return [waveform_repeated]
        else:
            waveform_tail_data = np.zeros(waveform_tail_samples)

            waveform_tail['waveform'] = self.instrument.new_waveform_from_double(waveform_type=0,
                                                                              waveform_data_a=waveform_tail_data)
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

        For example:
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

        example return value:
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
        if isinstance(self.pulse.connection, SingleConnection):
            channels = [self.pulse.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.pulse.connection))

        waveforms = {}

        full_name = self.pulse.full_name or 'none'

        # channel independent parameters
        duration = self.pulse.t_stop - self.pulse.t_start
        waveform_multiple = 5
        waveform_minimum = 15  # the minimum size of a waveform

        for ch_idx, ch in enumerate(channels):

            sampling_rate = default_sampling_rates[ch_idx]
            prescaler = 0 if sampling_rate == 500e6 else 100e6 / sampling_rate
            period_sample = 1 / sampling_rate

            period = period_sample * waveform_multiple
            max_cycles = int(duration // period)

            # This factor determines the number of points needed in the waveform
            # as the number of waveform cycles is limited to (2 ** 16 - 1 = 65535)
            n = int(np.ceil(max_cycles / 2 ** 16))
            if n < 3:
                waveform_samples = firstFactorAboveN(int(round(duration / period_sample)),
                                                     waveform_minimum, waveform_multiple)
            else:
                waveform_samples = n * waveform_multiple

            cycles = int(duration // (period_sample * waveform_samples))
            if duration + threshold < waveform_minimum * period_sample:
                raise RuntimeError(f'Waveform too short for {full_name}: '
                                   f'{waveform_samples/sampling_rate*1e3:.3f}ms < '
                                   f'{waveform_minimum/sampling_rate*1e3:.3f}ms')

            # the first waveform (waveform_repeated) is repeated n times
            # the second waveform is for the final part of the total wave so the total wave looks like:
            #   n_cycles * waveform_repeated + waveform_tail
            # This is done to minimise the number of data points written to the AWG
            waveform_repeated_period = period_sample * waveform_samples
            t_list_1 = np.linspace(self.pulse.t_start, self.pulse.t_start + waveform_repeated_period, waveform_samples, endpoint=False)
            waveform_repeated_cycles = cycles
            waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

            waveform_tail_start = self.pulse.t_start + waveform_repeated_duration
            waveform_tail_samples = waveform_multiple * round(
                ((self.pulse.t_stop - waveform_tail_start) / period_sample) / waveform_multiple)

            if waveform_tail_samples < waveform_minimum:
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
