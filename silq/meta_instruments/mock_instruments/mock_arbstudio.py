from functools import partial

from qcodes import Instrument


def print_function(*args, **kwargs):
    print('args={args}, kwargs={kwargs}'.format(args=args, kwargs=kwargs))

class MockArbStudio(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        channels = ['ch1', 'ch2', 'ch3', 'ch4']
        functions = ['trigger_source', 'trigger_mode',
                     'add_waveform', 'sequence']
        for ch in channels:
            for fn in functions:
                pass
                self.add_parameter(
                    '{}_{}'.format(ch, fn),
                    set_cmd=partial(print_function, ch=ch, function=fn)
                    )