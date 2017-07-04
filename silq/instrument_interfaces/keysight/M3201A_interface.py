import numpy as np

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation, TriggerPulse, AWGPulse,\
                        CombinationPulse, DCPulse, DCRampPulse, MarkerPulse
from silq.meta_instruments.layout import SingleConnection
from silq.tools.pulse_tools import pulse_to_waveform_sequence
from functools import partial
import threading
import logging
logger = logging.getLogger(__name__)


class M3201AInterface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        self._output_channels = {
            'ch{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='ch{}'.format(k), id=k,
                        output=True) for k in range(4)}

        self._pxi_channels = {
            'pxi{}'.format(k):
                Channel(instrument_name=self.instrument_name(),
                        name='pxi{}'.format(k), id=4000 + k,
                        input_trigger=True, output=True, input=True) for k in range(8)}

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
        # TODO: figure out how/if we want to configure channel-specific sampling rates
        sampling_rates = {ch: 500e6 for ch in self.active_channels()}
        # TODO: figure out how we want to configure error_threshold
        error_threshold = 1e-6
        # TODO: think about how to configure queue behaviour (cyclic/one shot for example)

        # flush the onboard RAM and reset waveform counter
        prescaler = 0
        sampling_rate = 500e6 if prescaler == 0 else 100e6 / prescaler
        self.instrument.flush_waveform()
        waveform_counter = 0

        # for each pulse:
        #   - implement
        # for each channel:
        #   - load waveforms
        #   - queue waveforms
        #   - start awg channel

        waveforms = dict()

        for pulse in self.pulse_sequence:
            channel_waveforms = pulse.implement(instrument=self.instrument,
                                                sampling_rates=sampling_rates,
                                                threshold=error_threshold)

            for ch in channel_waveforms:
                if ch in waveforms:
                    waveforms[ch] += channel_waveforms[ch]
                else:
                    waveforms[ch] = channel_waveforms[ch]

        # Sort the list of waveforms for each channel and calculate delays or throw error on overlapping waveforms.
        for ch in sorted(self._get_active_channels()):
            waveforms[ch] = sorted(waveforms[ch], key=lambda k: k['t_start'])

            insert_points = []
            for i, wf in enumerate(waveforms[ch]):
                if i == 0:
                    delay_duration = wf['t_start']
                else:
                    delay_duration = wf['t_start'] - waveforms[ch][i-1]['t_stop']

                # a waveform delay is expressed in tens of ns
                delay = int(round((delay_duration * 1e9) / 10))

                if delay > 6000:
                    # create a zero pulse and keep track of where to insert it later
                    # (as a replacement for the long delay)

                    logger.info('Delay waveform needed for "{}" : duration {:.3f} s'.format(wf['name'], delay_duration))
                    zero_waveforms = self.create_zero_waveform(duration=delay_duration,
                                                               prescaler=prescaler)
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
                duration = self.pulse_sequence.duration - wf['t_stop']
                logger.info(f'Adding a final delay waveform to {ch} for ' \
                            f'{self.pulse_sequence.duration - wf["t_stop"]}s')
                insertion = {'index': i+1, 'waveforms': zero_waveforms}
                insert_points.append(insertion)

            insert_points = sorted(insert_points, key=lambda k: k['index'], reverse=True)

            for insertion in insert_points:
                i = insertion['index']
                waveforms[ch][i:i] = insertion['waveforms']

            logger.debug(f'\n{ch} AWG Waveforms:\n' +
                         '\n'.join(f'\t{wf["name"]}' \
                                    f'\tt_start={wf.get("t_start",-1):.3f}' \
                                    f'\tt_stop={wf.get("t_stop", -1):.3f}' for wf in waveforms[ch]))


        self.instrument.off()
        self.instrument.flush_waveform()
        # print('Loading waveforms onto AWG')
        for ch in waveforms:
            self.instrument.awg_flush(self._channels[ch].id)
            self.instrument.set_channel_wave_shape(wave_shape=6, channel_number=self._channels[ch].id)
            self.instrument.set_channel_amplitude(amplitude=1.5, channel_number=self._channels[ch].id)
            waveform_array = waveforms[ch]
            # print(waveform_array)
            ch_wf_counter = 0
            for waveform in waveform_array:
                logger.debug('loading waveform-object {} in M3201A with waveform id {}'.format(id(waveform['waveform']),
                                                                                        waveform_counter))

                self.instrument.load_waveform(waveform['waveform'], waveform_counter)
                if ch_wf_counter == 0:
                    # await software trigger for first wf if not in cyclic mode
                    trigger_mode = not self.cyclic_mode
                else:
                    trigger_mode = 0  # auto trigger for every wf that follows
                logger.debug('queueing waveform {} with id {} to awg channel {} for {} cycles with prescaler {}, delay {} and trigger {}'
                      .format(waveform['name'], waveform_counter, self._channels[ch].id,
                              int(waveform['cycles']), int(waveform['prescaler']),
                              int(waveform['delay']), trigger_mode))
                self.instrument.awg_queue_waveform(self._channels[ch].id, waveform_counter, trigger_mode,
                                                   0, int(waveform['cycles']), prescaler=int(waveform.get('prescaler', 0)))
                waveform_counter += 1
                ch_wf_counter += 1

            # print('starting awg channel {}'.format(self._channels[ch].id))
            self.instrument.awg.AWGqueueConfig(nAWG=self._channels[ch].id, mode=self.cyclic_mode)
            self.instrument.awg_start(self._channels[ch].id)
        pass

    def start(self):
        mask = 0
        for c in self._get_active_channel_ids():
            mask |= 1 << c
        if not self.cyclic_mode:
            self.started = True
            duration = self.pulse_sequence.duration
            trigger_period = duration * 1.1
            logger.info(f'Starting self triggering of the M3201 AWG with interval {trigger_period*1100:.3f}ms.')
            self.trigger_self(trigger_period)
        else:
            self.software_trigger()

    def trigger_self(self, trigger_period):
        self.software_trigger()
        if self.started:
            self.trigger_thread = threading.Timer(trigger_period, partial(self.trigger_self, trigger_period))
            self.trigger_thread.start()

    def get_final_additional_pulses(self, **kwargs):
        return []

    def write_raw(self, cmd):
        pass

    def ask_raw(self, cmd):
        pass

    def software_trigger(self):
        # from time import time
        # if not hasattr(self, 't_stamp'):
        #     self.t_stamp = time()
        # print(f'{(time()-self.t_stamp)*1000}')
        mask = 0
        for c in self._get_active_channel_ids():
            mask |= 1 << c
        self.instrument.awg_stop_multiple(mask)
        self.instrument.awg_start_multiple(mask)
        self.instrument.awg_trigger_multiple(mask)

    def create_zero_waveform(self, duration, prescaler):
        wave_form_multiple = 5
        wave_form_minimum = 15  # the minimum size of a waveform

        sampling_rate = 500e6 if prescaler == 0 else 100e6 / prescaler

        if (duration < wave_form_minimum / sampling_rate):
            return None

        period_sample = 1 / sampling_rate

        period = period_sample * wave_form_multiple

        cycles = duration // period
        if (cycles < 1):
            return None

        n = int(-(-cycles // 2 ** 16))

        samples = n * wave_form_multiple
        samples = max(samples, wave_form_minimum)
        cycles = int(duration // (period_sample * samples))

        waveform_repeated_period = period_sample * samples
        waveform_repeated_cycles = cycles
        waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

        waveform_tail_samples = wave_form_multiple * int(round(
            ((duration - waveform_repeated_duration) / period_sample + 1) / wave_form_multiple))

        if waveform_tail_samples < wave_form_minimum:
            waveform_tail_samples = 0

        waveform_repeated = {}
        waveform_tail = {}

        waveform_repeated_data = np.zeros(samples)

        waveform_repeated['waveform'] = self.instrument.new_waveform_from_double(waveform_type=0,
                                                                          waveform_data_a=waveform_repeated_data)
        waveform_repeated['name'] = 'zero_pulse'
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
            waveform_tail['cycles'] = 1
            waveform_tail['delay'] = 0
            waveform_tail['prescaler'] = prescaler

            return [waveform_repeated, waveform_tail]


class SinePulseImplementation(PulseImplementation, SinePulse):
    def __init__(self, prescaler=0, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SinePulse, **kwargs)
        self.prescaler = prescaler

    def target_pulse(self, pulse, interface, **kwargs):
        logger.debug('targeting SinePulse for M3201A interface {}'.format(interface))
        is_primary = kwargs.pop('is_primary', False)
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
        if (not is_primary) and targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-3,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))

        return targeted_pulse

    def implement(self, instrument, sampling_rates, threshold):
        """
        This function takes the targeted pulse (i.e. an interface specific pulseimplementation) and converts
        it to a set of pulse-independent instructions/information that can be handled by interface.setup().

        For example:
            SinePulseImplementation()
                initializes a generic (interface independent) pulseimplementation for the sinepulse
            SinePulseImplementation.target_pulse()
                target the generic (interface independent) pulseimplementation to this specific interface, also adds
                trigger requirements, which are communicated back to the layout. The trigger requirements are also
                interface specific. The pulse in now called a 'targeted pulse'.
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

        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.connection))

        sampling_rate = 500e6 if self.prescaler == 0 else 100e6 / self.prescaler
        period_sample = 1 / sampling_rate

        assert self.frequency < sampling_rate / 2, \
            f'Sine frequency {self.frequency} is too high for' \
            f'the sampling frequency {sampling_rate}'

        waveforms = {}

        full_name = self.full_name or 'none'

        # channel independent parameters
        duration = self.duration
        period = 1 / self.frequency
        cycles = duration // period
        # TODO: maybe make n_max an argument? Or even better: make max_samples a parameter?
        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15  # the minimum size of a waveform

        sampling_rate = 500e6 if self.prescaler == 0 else 100e6 / self.prescaler
        period_sample = 1 / sampling_rate

        for ch in channels:
            # This factor determines the number of points needed in the waveform
            # as the number of waveform cycles is limited to (2 ** 16 - 1 = 65535)
            n_min = int(-(-cycles // 2**16))

            n, error, waveform_samples = pulse_to_waveform_sequence(duration, self.frequency, sampling_rate, threshold,
                                                           n_min=n_min, n_max=1000,
                                                           sample_points_multiple=waveform_multiple)

            if waveform_samples < waveform_minimum:
                                   f'{waveform_samples*sampling_rate*1e3:.3f}ms < '
                                   f'{waveform_minimum*sampling_rate*1e3:.3f}ms')
                raise RuntimeError(f'Waveform too short for {full_name}: '


            # the first waveform (waveform_repeated) is repeated n times
            # the second waveform is for the final part of the total wave so the total wave looks like:
            #   n_cycles * waveform_repeated + waveform_tail
            # This is done to minimise the number of data points written to the AWG
            waveform_repeated_period = period_sample * waveform_samples
            t_list_1 = np.linspace(self.t_start,
                                   self.t_start + waveform_repeated_period,
                                   waveform_samples, endpoint=False)

            waveform_repeated_cycles = cycles // n
            waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

            waveform_tail_start = self.t_start + waveform_repeated_duration
            waveform_tail_samples = waveform_multiple * round(
                ((self.t_stop - waveform_tail_start) / period_sample + 1) / waveform_multiple)

            if waveform_tail_samples < waveform_minimum:
                # logger.debug('tail is too short, removing tail (tail size was: {})'.format(waveform_tail_samples))
                waveform_tail_samples = 0

            t_list_2 = np.linspace(waveform_tail_start, self.t_stop, waveform_tail_samples, endpoint=True)

            waveform_repeated = {}
            waveform_tail = {}

            waveform_repeated_data = [voltage/1.5 for voltage in self.get_voltage(t_list_1)]

            waveform_repeated['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                         waveform_data_a=waveform_repeated_data)
            waveform_repeated['name'] = full_name,
            waveform_repeated['cycles'] = waveform_repeated_cycles
            waveform_repeated['t_start'] = self.t_start
            waveform_repeated['t_stop'] = waveform_tail_start
            waveform_repeated['prescaler'] = self.prescaler

            if len(t_list_2) == 0:
                waveforms[ch] = [waveform_repeated]
            else:
                waveform_tail_data = [voltage/1.5 for voltage in self.get_voltage(t_list_2)]

                waveform_tail['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                             waveform_data_a=waveform_tail_data)
                waveform_tail['name'] = full_name + '_tail',
                waveform_tail['cycles'] = 1
                waveform_tail['t_start'] = waveform_tail_start
                waveform_tail['t_stop'] = waveform_tail_stop
                waveform_tail['prescaler'] = self.prescaler

                waveforms[ch] = [waveform_repeated, waveform_tail]

        return waveforms


class DCPulseImplementation(PulseImplementation, DCPulse):
    def __init__(self, prescaler=0, **kwargs):
        # Default sampling rate of 1 MSPS
        PulseImplementation.__init__(self, pulse_class=DCPulse, **kwargs)
        self.prescaler = prescaler

    def target_pulse(self, pulse, interface, **kwargs):
        logger.debug('targeting DCPulse for {}'.format(interface))
        is_primary = kwargs.pop('is_primary', False)
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence.get_pulses(
            t_start=pulse.t_start, trigger=True
        )

        # Add a trigger requirement, which is sent back to the Layout
        if (not is_primary) and targeted_pulse.t_start == 0 and not trigger_pulses:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-5,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))

        return targeted_pulse

    def implement(self, instrument, sampling_rates, threshold):
        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.connection))

        waveforms = {}

        full_name = self.full_name or 'none'

        # channel independent parameters
        duration = self.t_stop - self.t_start
        waveform_multiple = 5
        waveform_minimum = 15  # the minimum size of a waveform

        sampling_rate = 500e6 if self.prescaler == 0 else 100e6 / self.prescaler
        period_sample = 1 / sampling_rate

        for ch in channels:
            period = period_sample * waveform_minimum
            max_cycles = int(duration // period)

            # This factor determines the number of points needed in the waveform
            # as the number of waveform cycles is limited to (2 ** 16 - 1 = 65535)
            n = int(np.ceil(max_cycles / 2 ** 16))

            waveform_samples = n * waveform_minimum
            if waveform_samples < waveform_minimum:
                                   f'{waveform_samples*sampling_rate*1e3:.3f}ms < '
                                   f'{waveform_minimum*sampling_rate*1e3:.3f}ms')
                raise RuntimeError(f'Waveform too short for {full_name}: '

            # the first waveform (waveform_repeated) is repeated n times
            # the second waveform is for the final part of the total wave so the total wave looks like:
            #   n_cycles * waveform_repeated + waveform_tail
            # This is done to minimise the number of data points written to the AWG
            waveform_repeated_period = period_sample * waveform_samples
            t_list_1 = np.linspace(self.t_start, self.t_start + waveform_repeated_period, waveform_samples, endpoint=False)
            waveform_repeated_cycles = max_cycles // n
            waveform_repeated_duration = waveform_repeated_period * waveform_repeated_cycles

            # print(f'{self.name}')
            # print(f'\tn={n}, samples={samples}')
            # print(f'\twf1: {self.t_start} to {self.t_start + waveform_repeated_period} repeated {waveform_repeated_cycles} times')


            waveform_tail_start = self.t_start + waveform_repeated_duration
            waveform_tail_samples = waveform_multiple * round(
                ((self.t_stop - waveform_tail_start) / period_sample + 1) / waveform_multiple)

            if waveform_tail_samples < waveform_minimum:
                # print('tail is too short, removing tail (tail size was: {})'.format(waveform_tail_samples))
                waveform_tail_samples = 0

            # waveform_tail_stop = waveform_tail_start + period_sample * (waveform_tail_samples - 1)
            t_list_2 = np.linspace(waveform_tail_start, self.t_stop, waveform_tail_samples, endpoint=True)
            # print(f'\twf2: {waveform_tail_start} to {self.t_stop}\n')

            waveform_repeated = {}
            waveform_tail = {}

            waveform_repeated_data = [voltage/1.5 for voltage in self.get_voltage(t_list_1)]

            waveform_repeated['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                         waveform_data_a=waveform_repeated_data)
            waveform_repeated['name'] = full_name
            waveform_repeated['cycles'] = waveform_repeated_cycles
            waveform_repeated['t_start'] = self.t_start
            waveform_repeated['t_stop'] = waveform_tail_start
            waveform_repeated['prescaler'] = self.prescaler

            if len(t_list_2) == 0:
                waveforms[ch] = [waveform_repeated]
            else:
                waveform_tail_data = [voltage/1.5 for voltage in self.get_voltage(t_list_2)]

                waveform_tail['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                             waveform_data_a=waveform_tail_data)

                waveform_tail['name'] = full_name + '_tail'
                waveform_tail['cycles'] = 1
                waveform_tail['t_start'] = waveform_tail_start
                waveform_tail['t_stop'] = self.t_stop
                waveform_tail['prescaler'] = self.prescaler

                waveforms[ch] = [waveform_repeated, waveform_tail]

        return waveforms


class DCRampPulseImplementation(PulseImplementation, DCRampPulse):
    def __init__(self, prescaler=0, **kwargs):
        # Default sampling rate of 1 MSPS
        PulseImplementation.__init__(self, pulse_class=DCRampPulse, **kwargs)
        self.prescaler = prescaler

    def target_pulse(self, pulse, interface, **kwargs):
        logger.debug('targeting DCRampPulse for {}'.format(interface))
        is_primary = kwargs.pop('is_primary', False)
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Check if there are already trigger pulses
        trigger_pulses = interface.input_pulse_sequence.get_pulses(
            t_start=pulse.t_start, trigger=True
        )

        # Add a trigger requirement, which is sent back to the Layout
        if (not is_primary) and targeted_pulse.t_start == 0 and not trigger_pulses:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-5,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))

        return targeted_pulse

    def implement(self, instrument, sampling_rates, threshold):
        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.connection))

        waveforms = {}

        full_name = self.full_name or 'none'

        # channel independent parameters
        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15

        sampling_rate = 500e6 if self.prescaler == 0 else 100e6 / self.prescaler
        period_sample = 1 / sampling_rate

        for ch in channels:
            waveform_samples = waveform_multiple * round(
                (self.duration / period_sample + 1) / waveform_multiple)
            t_list = np.linspace(self.t_start, self.t_stop,
                                 waveform_samples, endpoint=True)

            if waveform_samples < waveform_minimum:
                                   f'{waveform_samples*sampling_rate*1e3:.3f}ms < '
                                   f'{waveform_minimum*sampling_rate*1e3:.3f}ms')
                raise RuntimeError(f'Waveform too short for {full_name}: '
            waveform_data = [voltage / 1.5 for voltage in
                             self.get_voltage(t_list)]

            waveform = {
                'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                waveform_data_a=waveform_data),
                'name': full_name,
                'cycles': 1,
                't_start': self.t_start,
                't_stop': self.t_stop,
                'prescaler': self.prescaler}

            waveforms[ch] = [waveform]

        return waveforms


class AWGPulseImplementation(PulseImplementation, AWGPulse):
    def __init__(self, prescaler=0, **kwargs):
        PulseImplementation.__init__(self, pulse_class=AWGPulse, **kwargs)
        self.prescaler = prescaler

    def target_pulse(self, pulse, interface, **kwargs):
        logger.debug('targeting AWGPulse for {}'.format(interface))
        is_primary = kwargs.pop('is_primary', False)
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
        if (not is_primary) and targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-5,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))

        return targeted_pulse

    def implement(self, instrument, sampling_rates, threshold):
        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.connection))

        waveforms = {}

        full_name = self.full_name or 'none'

        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15

        for ch in channels:
            sampling_rate = 500e6 if self.prescaler == 0 else 100e6 / self.prescaler
            period_sample = 1 / sampling_rate


            waveform_samples = waveform_multiple * round(
                (self.duration / period_sample + 1) / waveform_multiple)
            t_list = np.linspace(self.t_start, self.t_stop, waveform_samples, endpoint=True)

            waveform_data = [voltage/1.5 for voltage in self.get_voltage(t_list)]

            waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                        waveform_data_a=waveform_data),
                        'name':full_name,
                        'cycles': 1,
                        't_start': self.t_start,
                        't_stop': self.t_start,
                        'prescaler':self.prescaler}

            waveforms[ch] = [waveform]

        return waveforms


class CombinationPulseImplementation(PulseImplementation, CombinationPulse):
    def __init__(self, prescaler=0, **kwargs):
        PulseImplementation.__init__(self, pulse_class=CombinationPulse, **kwargs)
        self.prescaler = prescaler

    def target_pulse(self, pulse, interface, **kwargs):
        logger.debug('targeting CombinationPulse for {}'.format(interface))
        is_primary = kwargs.pop('is_primary', False)
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
        if (not is_primary) and targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-5,
                             connection_requirements={
                                 'input_instrument': interface.instrument_name(),
                                 'trigger': True}))

        return targeted_pulse

    def implement(self, instrument, sampling_rates, threshold):
        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.connection))

        waveforms = {}

        full_name = self.full_name or 'none'

        waveform_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5
        waveform_minimum = 15

        for ch in channels:
            sampling_rate = 500e6 if self.prescaler == 0 else 100e6 / self.prescaler
            period_sample = 1 / sampling_rate


            waveform_samples = waveform_multiple * round(
                (self.duration/ period_sample + 1) / waveform_multiple)

            if waveform_samples < waveform_minimum:
                               f'{waveform_samples*sampling_rate*1e3:.3f}ms < '
                               f'{waveform_minimum*sampling_rate*1e3:.3f}ms')
                raise RuntimeError(f'Waveform too short for {full_name}: '

            t_list = np.linspace(self.t_start, self.t_stop, waveform_samples, endpoint=True)

            waveform_data = [voltage/1.5 for voltage in self.get_voltage(t_list)]

            waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                        waveform_data_a=waveform_data),
                        'name': full_name,
                        'cycles': 1,
                        't_start': self.t_start,
                        't_stop': self.t_stop,
                        'prescaler': self.prescaler}

            waveforms[ch] = [waveform]

        return waveforms


class TriggerPulseImplementation(PulseImplementation, TriggerPulse):
    def __init__(self, prescaler = 0, **kwargs):
        PulseImplementation.__init__(self, pulse_class=TriggerPulse, **kwargs)
        self.prescaler = prescaler

    @property
    def amplitude(self):
        return 1.0

    def implement(self, instrument, sampling_rates, threshold):
        if isinstance(self.connection, SingleConnection):
            channel = self.connection.output['channel'].name
        else:
            raise Exception('No implementation for connection {}'.format(self.connection))

        waveform_multiple = 5
        waveform_minimum = 15

        waveforms = {}

        full_name = self.full_name or 'none'

        sampling_rate = 500e6 if self.prescaler == 0 else 100e6/self.prescaler
        period_sample = 1 / sampling_rate

        waveform_samples = waveform_multiple * round(
            (self.duration / period_sample + 1) / waveform_multiple)
        if waveform_samples < waveform_minimum:
                f'{waveform_samples*sampling_rate*1e3:.3f}ms < '
                f'{waveform_minimum*sampling_rate*1e3:.3f}ms')
            raise RuntimeError(f'Waveform too short for {full_name}: '
        t_list = np.linspace(self.t_start, self.t_stop, waveform_samples, endpoint=True)

        waveform_data = [voltage/1.5 for voltage in self.get_voltage(t_list)] + [0]

        waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                    waveform_data_a=waveform_data),
                    'name': full_name,
                    'cycles': 1,
                    't_start': self.t_start,
                    't_stop': self.t_stop,
                    'prescaler': self.prescaler}

        waveforms[channel] = [waveform]

        return waveforms


class MarkerPulseImplementation(PulseImplementation, MarkerPulse):
    def __init__(self, prescaler = 0, **kwargs):
        PulseImplementation.__init__(self, pulse_class=MarkerPulse, **kwargs)
        self.prescaler = prescaler

    @property
    def amplitude(self):
        return 1.0

    def implement(self, instrument, sampling_rates, threshold):
        if isinstance(self.connection, SingleConnection):
            channel = self.connection.output['channel'].name
        else:
            raise Exception('No implementation for connection {}'.format(self.connection))

        waveform_multiple = 5
        waveform_minimum = 15

        waveforms = {}

        full_name = self.full_name or 'none'

        sampling_rate = 500e6 if self.prescaler == 0 else 100e6/self.prescaler
        period_sample = 1 / sampling_rate

        # Waveform must have at least wave_form_multiple samples
        waveform_samples = waveform_multiple * round(
            (self.duration / period_sample + 1) / waveform_multiple)
        if waveform_samples < waveform_minimum:
                f'{waveform_samples*sampling_rate*1e3:.3f}ms < '
                f'{waveform_minimum*sampling_rate*1e3:.3f}ms')
            raise RuntimeError(f'Waveform too short for {full_name}: '
        t_list = np.linspace(self.t_start, self.t_stop, waveform_samples, endpoint=True)

        waveform_data = [voltage/1.5 for voltage in self.get_voltage(t_list)] + [0]

        waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                    waveform_data_a=waveform_data),
                    'name': full_name,
                    'cycles': 1,
                    't_start': self.t_start,
                    't_stop': self.t_stop,
                    'prescaler': self.prescaler}

        waveforms[channel] = [waveform]

        return waveforms
