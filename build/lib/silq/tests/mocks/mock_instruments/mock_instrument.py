from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

class MockInstrument(Instrument):
    def __init__(self, name, silent=False, **kwargs):
        super().__init__(name, **kwargs)
        self._silent = False
        self.add_parameter('silent',
                           parameter_class=ManualParameter,
                           initial_value=silent,
                           vals=vals.Bool())

    def print_function(self, *args, function='{f}', **kwargs):
        if not self.silent():
            print('{ins}.{fun}(args={args}, kwargs={kwargs})'.format(
                ins=self.name, fun=function, args=args, kwargs=kwargs))