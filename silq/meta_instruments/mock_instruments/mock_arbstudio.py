from functools import partial

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


def print_function(*args, **kwargs):
    print('args={args}, kwargs={kwargs}'.format(args=args, kwargs=kwargs))

class MockArbStudio(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        channels = [1, 2, 3, 4]
        channel_functions = ['trigger_source', 'trigger_mode',
                             'clear_waveforms']
        # functions = ['load_waveforms', 'load_sequence']
        # Initialize waveforms and sequences
        self._waveforms = [[] for k in channels]
        for ch in channels:
            for fn in channel_functions:
                self.add_parameter(
                    'ch{}_{}'.format(ch, fn),
                    get_cmd=partial(print_function, ch=ch, function=fn,
                                    mode='get'),
                    set_cmd=partial(print_function, ch=ch, function=fn,
                                    mode='set'),
                    vals=vals.Anything())

            self.add_function('ch{}_add_waveform'.format(ch),
                              call_cmd=partial(self._add_waveform, ch),
                              args=[vals.Anything()])

            self.add_parameter('ch{}_sequence'.format(ch),
                               parameter_class=ManualParameter,
                               label='Channel {} Sequence'.format(ch),
                               initial_value=[],
                               vals=vals.Anything())

    def _add_waveform(self, channel, waveform):
        print_function(channel=channel, function='add_waveform',
                       waveform=waveform)
        assert len(waveform)%2 == 0, 'Waveform must have an even number of points'
        assert len(waveform)> 2, 'Waveform must have at least four points'
        self._waveforms[channel - 1].append(waveform)

    def load_waveforms(self, channels=[]):
        print_function(channels=channels, function='load_waveforms')

    def load_sequence(self, channels=[]):
        print_function(channels=channels, function='load_sequence')

    def get_waveforms(self):
        return self._waveforms