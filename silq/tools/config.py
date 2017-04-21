from blinker import signal
from qcodes.config.config import DotDict


class PulseConfig(DotDict):
    def __init__(self, name):
        self.__dict__['name'] = name
        self.__dict__['signal'] = signal('pulse_config:' + self.name)

    def __setitem__(self, key, val):
        super().__setitem__(key, val)
        self.signal.send(self, **{key: val})

    __setattr__ = __setitem__