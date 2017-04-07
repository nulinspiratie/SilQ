import numpy as np

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.pulses import SinePulse, PulseImplementation, TriggerPulse
from silq.meta_instruments.layout import SingleConnection


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

    def stop(self):
        # stop all AWG channels and sets FG channels to 'No Signal'
        self.instrument.off()

    def setup(self):
        # TODO: figure out how/if we want to configure channel-specific sampling rates
        sampling_rates = {ch: 500e6 for ch in self.active_channels()}
        error_threshold = 1e-6

        # flush the onboard RAM and reset waveform counter
        self.instrument.flush_waveform()
        waveform_counter = 1

        # for each pulse:
        #   - implement
        # for each channel:
        #   - load waveforms
        #   - queue waveforms
        #   - start awg channel

        for pulse in self.pulse_sequence():
            channel_waveforms = pulse.implement(instrument=self.instrument,
                                                sampling_rates=sampling_rates,
                                                threshold=error_threshold)

            for ch in channel_waveforms:
                waveform_array = channel_waveforms[ch]
                ch_wf_counter = 1
                for waveform in waveform_array:
                    print('loading waveform-object {} in M3201A with waveform id {}'.format(id(waveform['waveform']),
                                                                                            waveform_counter))
                    if ch_wf_counter == 1:
                        trigger_mode = 1  # software trigger for first wf
                    else:
                        trigger_mode = 0  # auto trigger for every wf that follows
                    print('queueing waveform with id {} to awg channel {} for {} cycles with delay {} and trigger {}'
                          .format(waveform_counter, self._channels[ch].id, waveform['cycles'], waveform['delay'],
                                  trigger_mode))
                    waveform_counter += 1
                    ch_wf_counter += 1
                print('starting awg channel {}'.format(self._channels[ch].id))
        pass

    def start(self):
        mask = 0
        for c in self.active_channels():
            mask |= 1 << c
        self.instrument.awg_start_multiple(mask)

    def get_final_additional_pulses(self, **kwargs):
        return []

    def write_raw(self, cmd):
        pass

    def ask_raw(self, cmd):
        pass


class SinePulseImplementation(PulseImplementation, SinePulse):
    def __init__(self, **kwargs):
        PulseImplementation.__init__(self, pulse_class=SinePulse, **kwargs)

    def target_pulse(self, pulse, interface, **kwargs):
        print('targeting SinePulse for M3201A interface {}'.format(interface))
        # Target the generic pulse to this specific interface
        targeted_pulse = PulseImplementation.target_pulse(
            self, pulse, interface=interface, **kwargs)

        # Add a trigger requirement, which is sent back to the Layout
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
        print('implementing SinePulse for the M3201A interface')
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
        n_max = min(1000, cycles)
        wave_form_multiple = 5  # the M3201A AWG needs the waveform length to be a multiple of 5

        for ch in channels:
            # TODO: check if sampling rate is indeed something we want to configure on a channel basis
            period_sample = 1 / sampling_rates[ch]

            # brute force method to find the minimum number of sample points in a single waveform cycle, such that it
            # satisfies the error threshold
            extra_sample = False
            n = 0

            for n in range(1, n_max + 1):
                error = (n * period) % (period_sample * wave_form_multiple)
                error_extra_sample = (period_sample * wave_form_multiple) - error
                if error_extra_sample < error:
                    extra_sample = True
                    error = error_extra_sample
                else:
                    extra_sample = False
                error = error / n / period
                if error < threshold:
                    break
                else:
                    continue

            samples = (n * period) // (period_sample * wave_form_multiple) * wave_form_multiple
            if extra_sample:
                samples += wave_form_multiple

            # calculate waveform based on the number of samples
            # TODO: what unit does t_start have? Is this consistent across all pulses?
            # TODO: Make this very clear in the documentation! Answer is: we use ms as the standard time unit.

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

            waveform_1_data = [self.get_voltage(t_list_1)]
            waveform_2_data = [self.get_voltage(t_list_2)]

            waveform_1['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                         waveform_data_a=waveform_1_data)
            waveform_1['cycles'] = waveform_1_cycles
            waveform_1['delay'] = 0.0

            waveform_2['waveform'] = instrument.new_waveform_from_double(waveform_type=0,
                                                                         waveform_data_a=waveform_2_data)
            waveform_2['cycles'] = 1
            waveform_2['delay'] = 0.0

            waveforms[ch] = [waveform_1, waveform_2]

        return waveforms
