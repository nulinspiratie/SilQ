from functools import partial

from . import MockInstrument

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals



class MockATS(MockInstrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        functions = ['config', 'acquire']
        for function in functions:
            self.add_function(function,
                              call_cmd=partial(self.print_function,
                                               function=function),
                              args=[vals.Anything()])