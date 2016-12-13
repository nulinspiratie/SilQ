from qcodes import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

class TestInnerInstrument(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        pass

class TestOuterInstrument(Instrument):
    shared_kwargs = ['instruments']
    def __init__(self, name, instruments=[], **kwargs):
        super().__init__(name, **kwargs)
        self.instruments = instruments
        self.instrument = TestInnerInstrument('testIns', server_name='inner_server')

    def get_instruments(self):
        return self.instruments


class TestInstrument(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.add_parameter(name='x_val',
                           get_cmd=lambda: self.x,
                           vals=vals.Anything())
        self.add_parameter(name='pulse',
                           parameter_class=ManualParameter,
                           vals=vals.Anything())

    def set_x(self, val):
        self.x = val