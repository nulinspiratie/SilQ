import numpy as np

import qcodes as qc
from qcodes.instrument.parameter import Parameter
from qcodes import config
from qcodes.data.io import DiskIO

from silq.tools.data_tools import get_data_folder


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
        print('latest folder: {}'.format(get_data_folder()))
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


class TestStoreParameter(Parameter):
    def __init__(self, data_manager, shape=(100,2), formatter=None, **kwargs):
        self.data_manager = data_manager
        self.formatter=formatter
        super().__init__(shape=shape, **kwargs)

    def get(self):
        result = np.random.randint(1,100,self.shape)

        t0 = time()
        while self.data_manager.ask('get_measuring'):
            print('pausing')
            sleep(2)

        self.data_set = data_tools.create_raw_data_set(
            name=self.name,
            data_manager=self.data_manager,
            shape=self.shape,
            formatter=self.formatter)
        data_tools.store_data(data_manager=self.data_manager,
                              result=result)
        print('time taken: {:.2f}'.format(time() - t0))
        return result


class GaussianParameter(Parameter):
    def __init__(self, name, set_parameter, x0=0, std=1, **kwargs):
        super().__init__(name, **kwargs)

        self.x0 = x0
        self.std = std
        self.set_parameter = set_parameter

    def get(self):
        set_val = self.set_parameter()
        return np.exp(-(set_val - self.x0)**2 / self.std**2)

class Gaussian2DParameter(Parameter):
    def __init__(self, name, set_parameters, x0=(0, 0), std=1, **kwargs):
        super().__init__(name, **kwargs)

        self.x0 = x0
        self.std = std
        self.set_parameters = set_parameters

    def get(self):
        set_vals = [p() for p in self.set_parameters]
        r = sum([(set_val - x0)**2 for set_val, x0 in zip(set_vals, self.x0)])
        return np.exp(-r / self.std**2)
