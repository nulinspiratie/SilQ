from functools import partial

from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


def print_function(*args, **kwargs):
    print('args={args}, kwargs={kwargs}'.format(args=args, kwargs=kwargs))

class MockPulseBlaster(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self._instructions = []

        functions = ['stop', 'start', 'start_programming', 'stop_programming',
                     'detect_boards', 'select_board']
        for function in functions:
            self.add_function(function,
                              call_cmd=partial(print_function, function=function),
                              args=[vals.Anything()])

    def send_instruction(self, flags, instruction, inst_args, length):
        print_function(flags, instruction, inst_args, length,
                       function='send_instruction')
        self._instructions += [(flags, instruction, inst_args, length)]