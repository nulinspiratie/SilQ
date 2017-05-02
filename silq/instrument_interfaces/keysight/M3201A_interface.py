import numpy as np

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation, TriggerPulse, AWGPulse, CombinationPulse, DCPulse
from silq.meta_instruments.layout import SingleConnection
from silq.tools.pulse_tools import pulse_to_waveform_sequence


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
            )
        ]

        self.add_parameter('active_channels',
                           get_cmd=self._get_active_channels)

    # TODO: is this device specific? Does it return [0,1,2] or [1,2,3]?
    def _get_active_channels(self):
        active_channels = [pulse.connection.output['channel'].name
                           for pulse in self.pulse_sequence()]
        # Transform into set to ensure that elements are unique
        active_channels = list(set(active_channels))
        return active_channels

    def _get_active_channel_ids(self):
        active_channel_ids = [pulse.connection.output['channel'].id
                              for pulse in self.pulse_sequence()]
        # Transform into set to ensure that elements are unique
        active_channel_ids = list(set(active_channel_ids))
        return active_channel_ids

    def stop(self):
        # stop all AWG channels and sets FG channels to 'No Signal'
        self.instrument.off()

    def setup(self, **kwargs):
        # TODO: figure out how/if we want to configure channel-specific sampling rates
        sampling_rates = {ch: 500e6 for ch in self.active_channels()}
        # TODO: figure out how we want to configure error_threshold
        error_threshold = 1e-6
        # TODO: think about how to configure queue behaviour (cyclic/one shot for example)

        # flush the onboard RAM and reset waveform counter
        self.instrument.flush_waveform()
        waveform_counter = 1

        # for each pulse:
        #   - implement
        # for each channel:
        #   - load waveforms
        #   - queue waveforms
        #   - start awg channel

        waveforms = dict()

        for pulse in self.pulse_sequence():
            channel_waveforms = pulse.implement(instrument=self.instrument,
                                                sampling_rates=sampling_rates,
                                                threshold=error_threshold)

            for ch in channel_waveforms:
                if ch in waveforms:
                    waveforms[ch] += channel_waveforms[ch]
                else:
                    waveforms[ch] = channel_waveforms[ch]

        # Sort the list of waveforms for each channel and calculate delays or throw error on overlapping waveforms.
        for ch in waveforms:
            waveforms[ch] = sorted(waveforms[ch], key=lambda k: k['t_start'])

            for i, wf in enumerate(waveforms[ch]):
                if i == 0:
                    wf['delay'] = 0.0
                else:
                    delay = wf['t_start'] - waveforms[ch][i-1]['t_stop']
                    if delay < 0:
                        raise Exception('Overlapping pulses are not allowed for {}. Adjust t_start and t_stop values '
                                        'or consider using the CombinationPulse'.format(self))
                    else:
                        wf['delay'] = int(round(float(delay*1e9)/10))

        self.instrument.off()
        self.instrument.flush_waveform()
        for ch in waveforms:
            self.instrument.awg_flush(self._channels[ch].id)
            self.instrument.set_channel_wave_shape(wave_shape=6, channel_number=self._channels[ch].id)
            self.instrument.set_channel_amplitude(amplitude=1.0, channel_number=self._channels[ch].id)
            waveform_array = waveforms[ch]
            ch_wf_counter = 1
            for waveform in waveform_array:
                print('loading waveform-object {} in M3201A with waveform id {}'.format(id(waveform['waveform']),
                                                                                        waveform_counter))
                self.instrument.load_waveform(waveform['waveform'], waveform_counter)
                if ch_wf_counter == 1:
                    trigger_mode = 1  # software trigger for first wf
                else:
                    trigger_mode = 0  # auto trigger for every wf that follows
                print('queueing waveform with id {} to awg channel {} for {} cycles with delay {} and trigger {}'
                      .format(waveform_counter, self._channels[ch].id, int(waveform['cycles']), int(waveform['delay']),
                              trigger_mode))
                self.instrument.awg_queue_waveform(self._channels[ch].id, waveform_counter, trigger_mode,
                                                   int(waveform['delay']), int(waveform['cycles']), prescaler=0)
                waveform_counter += 1
                ch_wf_counter += 1
            print('starting awg channel {}'.format(self._channels[ch].id))
            self.instrument.awg_start(self._channels[ch].id)
        pass

    def start(self):
        mask = 0
        for c in self._get_active_channel_ids():
            mask |= 1 << c
        self.instrument.awg_start_multiple(mask)

    def get_final_additional_pulses(self, **kwargs):
        return []

    def write_raw(self, cmd):
        pass

    def ask_raw(self, cmd):
        pass

    def software_trigger(self):
        for c in self._get_active_channel_ids():
            self.instrument.awg_trigger(c)


class SinePulseImplementation(PulseImplementation, SinePulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SinePulse, **kwargs)

    def target_pulse(self, pulse, interface, **kwargs):
        print('targeting SinePulse for M3201A interface {}'.format(interface))
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
        if targeted_pulse.t_start == 0:
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
        # print('implementing SinePulse for the M3201A interface')
        # use t_start, t_stop, sampling_rate, ... to make a waveform object that can be queued in interface.setup()
        # basically, each implement in all PulseImplementations will be a waveform factory

        if isinstance(self.connection, SingleConnection):
            channels = [self.connection.output['channel'].name]
        else:
            raise Exception('No implementation for connection {}'.format(
                self.connection))

        assert self.frequency < min(sampling_rates[ch] for ch in channels) / 2, \
            'Sine frequency is higher than the Nyquist frequency ' \
            'for channels {}'.format(channels)

        waveforms = {}

        # channel independent parameters
        duration = self.t_stop - self.t_start
        period = 1 / self.frequency
        cycles = duration // period
        # TODO: maybe make n_max an argument? Or even better: make max_samples a parameter?
        wave_form_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5

        for ch in channels:
            # TODO: check if sampling rate is indeed something we want to configure on a channel basis
            period_sample = 1 / sampling_rates[ch]

            n_min = -(-cycles // 2**16)

            n, error, samples = pulse_to_waveform_sequence(duration, self.frequency, sampling_rates[ch], threshold,
                                                           n_min=n_min, n_max=1000,
                                                           sample_points_multiple=wave_form_multiple)

            # the first waveform (waveform_1) is repeated n times
            # the second waveform is for the final part of the total wave so the total wave looks like:
            #   n_cycles * waveform_1 + waveform_2
            waveform_1_period = period_sample * samples
            t_list_1 = np.linspace(self.t_start, self.t_start + waveform_1_period, samples, endpoint=False)

            waveform_1_cycles = cycles // n
            waveform_1_duration = waveform_1_period * waveform_1_cycles

            waveform_2_start = self.t_start + waveform_1_duration
            waveform_2_samples = wave_form_multiple * round(
                ((self.t_stop - waveform_2_start) / period_sample + 1) / wave_form_multiple)
            waveform_2_stop = waveform_2_start + period_sample * (waveform_2_samples - 1)
            t_list_2 = np.linspace(waveform_2_start, waveform_2_stop, waveform_2_samples, endpoint=True)

            waveform_1 = {}
            waveform_2 = {}

            waveform_1_data = self.get_voltage(t_list_1)

            waveform_1['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                         waveform_data_a=waveform_1_data)
            waveform_1['cycles'] = waveform_1_cycles
            waveform_1['t_start'] = self.t_start
            waveform_1['t_stop'] = waveform_2_start

            if len(t_list_2) == 0:
                waveforms[ch] = [waveform_1]
            else:
                waveform_2_data = self.get_voltage(t_list_2)

                waveform_2['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                             waveform_data_a=waveform_2_data)
                waveform_2['cycles'] = 1
                waveform_2['t_start'] = waveform_2_start
                waveform_2['t_stop'] = waveform_2_stop

                waveforms[ch] = [waveform_1, waveform_2]

        return waveforms


class DCPulseImplementation(PulseImplementation, DCPulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=AWGPulse, **kwargs)

    def target_pulse(self, pulse, interface, **kwargs):
        print('targeting DCPulse for {}'.format(interface))
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
        if targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-3,
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

        # channel independent parameters
        duration = self.t_stop - self.t_start
        wave_form_multiple = 5

        for ch in channels:
            period_sample = 1 / sampling_rates[ch]

            period = period_sample * wave_form_multiple
            cycles = duration // period

            n = -(-cycles // 2 ** 16)

            samples = n * wave_form_multiple

            waveform_1_period = period_sample * samples
            t_list_1 = np.linspace(self.t_start, self.t_start + waveform_1_period, samples, endpoint=False)

            waveform_1_cycles = cycles // n
            waveform_1_duration = waveform_1_period * waveform_1_cycles

            waveform_2_start = self.t_start + waveform_1_duration
            waveform_2_samples = wave_form_multiple * round(
                ((self.t_stop - waveform_2_start) / period_sample + 1) / wave_form_multiple)
            waveform_2_stop = waveform_2_start + period_sample * (waveform_2_samples - 1)
            t_list_2 = np.linspace(waveform_2_start, waveform_2_stop, waveform_2_samples, endpoint=True)

            waveform_1 = {}
            waveform_2 = {}

            waveform_1_data = self.get_voltage(t_list_1)

            waveform_1['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                         waveform_data_a=waveform_1_data)
            waveform_1['cycles'] = waveform_1_cycles
            waveform_1['t_start'] = self.t_start
            waveform_1['t_stop'] = waveform_2_start

            if len(t_list_2) == 0:
                waveforms[ch] = [waveform_1]
            else:
                waveform_2_data = self.get_voltage(t_list_2)

                waveform_2['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                             waveform_data_a=waveform_2_data)
                waveform_2['cycles'] = 1
                waveform_2['t_start'] = waveform_2_start
                waveform_2['t_stop'] = waveform_2_stop

                waveforms[ch] = [waveform_1, waveform_2]

        return waveforms


class AWGPulseImplementation(PulseImplementation, AWGPulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=AWGPulse, **kwargs)

    def target_pulse(self, pulse, interface, **kwargs):
        print('targeting AWGPulse for {}'.format(interface))
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
        if targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-3,
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

        wave_form_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5

        for ch in channels:
            # TODO: check if sampling rate is indeed something we want to configure on a channel basis
            period_sample = 1 / sampling_rates[ch]

            waveform_start = self.t_start
            waveform_samples = wave_form_multiple * round(
                ((self.t_stop - waveform_start) / period_sample + 1) / wave_form_multiple)
            waveform_stop = waveform_start + period_sample * (waveform_samples - 1)
            t_list = np.linspace(waveform_start, waveform_stop, waveform_samples, endpoint=True)

            waveform_data = self.get_voltage(t_list)

            waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                        waveform_data_a=waveform_data),
                        'cycles': 1,
                        't_start': self.t_start,
                        't_stop': waveform_stop}

            waveforms[ch] = [waveform]

        return waveforms


class CombinationPulseImplementation(PulseImplementation, CombinationPulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=CombinationPulse, **kwargs)

    def target_pulse(self, pulse, interface, **kwargs):
        print('targeting CombinationPulse for {}'.format(interface))
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
        if targeted_pulse.t_start == 0:
            targeted_pulse.additional_pulses.append(
                TriggerPulse(t_start=pulse.t_start,
                             duration=1e-3,
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

        wave_form_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5

        for ch in channels:
            # TODO: check if sampling rate is indeed something we want to configure on a channel basis
            period_sample = 1 / sampling_rates[ch]

            waveform_start = self.t_start
            waveform_samples = wave_form_multiple * round(
                ((self.t_stop - waveform_start) / period_sample + 1) / wave_form_multiple)
            waveform_stop = waveform_start + period_sample * (waveform_samples - 1)
            t_list = np.linspace(waveform_start, waveform_stop, waveform_samples, endpoint=True)

            waveform_data = self.get_voltage(t_list)

            waveform = {'waveform': instrument.new_waveform_from_double(waveform_type=0,
                                                                        waveform_data_a=waveform_data),
                        'cycles': 1,
                        't_start': self.t_start,
                        't_stop': waveform_stop}

            waveforms[ch] = [waveform]

        return waveforms
