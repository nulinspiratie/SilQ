import qcodes as qc
from qcodes.instrument.parameter import Parameter


class TestValParameter(Parameter):
    def __init__(self, name, **kwargs):
        self.val = 0
        super().__init__(name, **kwargs)

    @property
    def valfun(self):
        return self.val

    @valfun.setter
    def valfun(self, val):
        self.val = val

    def get(self):
        return self.val

    def set(self, val):
        self.val = val

class TestLoopParameter(Parameter):
    def __init__(self, name, param, **kwargs):
        self.val = 0
        self.param = param
        super().__init__(name, **kwargs)

    def get(self):
        qc.Loop(self.param[0:5:1], delay=0.5).each(self.param).run(
            name='inner_loop', background=False, data_manager=False)
        return self.param()