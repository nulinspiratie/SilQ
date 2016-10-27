from functools import partial

from . import MockInstrument

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class MockPulseBlaster(MockInstrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        functions = ['stop', 'start', 'start_programming', 'stop_programming',
                     'detect_boards']
        for function in functions:
            self.add_function(function,
                              call_cmd=partial(self.print_function,
                                               function=function))
        functions = ['select_board']
        for function in functions:
            self.add_function(function,
                              call_cmd=partial(self.print_function,
                                               function=function),
                              args=[vals.Anything()])

        self.add_parameter('core_clock',
                           parameter_class=ManualParameter,
                           initial_value=500)

        self.add_parameter('instructions',
                           parameter_class = ManualParameter,
                           initial_value=[],
                           vals=vals.Anything())

    def send_instruction(self, flags, instruction, inst_args, length):
        if not self.silent():
            self.print_function(flags, instruction, inst_args, length,
                           function='send_instruction')
        self.instructions(self.instructions() +
                          [(flags, instruction, inst_args, length)])
