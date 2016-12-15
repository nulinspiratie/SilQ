import qcodes as qc
from qcodes.instrument.parameter import Parameter
from qcodes import config
from qcodes.data.io import DiskIO

from silq.tools.data_tools import get_latest_data_folder

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
        if 'data_folder' in config['user']:
            print('data folder: {}'.format(config['user']['data_folder']))
        print('latest folder: {}'.format(get_latest_data_folder()))
        return self.param()


class SetParameter(Parameter):
    def __init__(self, name, param, **kwargs):
        super().__init__(name, **kwargs)
        self.param = param

    def get(self):
        return self.param()

    def set(self, val):
        self.param(val)


class ConfigParameter(Parameter):
    def __init__(self, name, key, **kwargs):
        super().__init__(name, **kwargs)

        if type(key) is str:
            self.config = config['user']
            self.key = key
        else:
            self.keys = key
            self.key = key[-1]

    def get_config(self):
        c = config['user']
        if hasattr(self, 'keys'):
            for key in self.keys[:-1]:
                c = c[key]

        return c

    def get(self):
        return self.get_config()[self.key]

    def set(self, val):
        self.get_config()[self.key] = val


class TestMeasureParameter(Parameter):
    def __init__(self, name, target_param, target_val, **kwargs):
        super().__init__(name, **kwargs)
        self.target_param = target_param
        self.target_val = target_val

    def get(self):
        return - abs(self.target_param() - self.target_val)