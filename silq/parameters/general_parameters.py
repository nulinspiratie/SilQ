import os
from time import time, sleep
import numpy as np

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data.data_set import new_data, DataMode
from qcodes.data.data_array import DataArray
from silq.tools import data_tools

class CombinedParameter(Parameter):
    def __init__(self, parameters, ratios=None,
                 name=None, label=None, units=None, **kwargs):
        if name is None:
            name = '_'.join([parameter.name for parameter in parameters])
        if label is None:
            label = ' and '.join([parameter.name for parameter in parameters])
        if units is None:
            units = parameters[0].units
        super().__init__(name, label=label, units=units, **kwargs)
        self.parameters = parameters

        if ratios is None:
            ratios = [1 for parameter in parameters]
        self.ratios = ratios

    def get(self):
        return self.parameters[0]() / self.ratios[0]

    def set(self, val):
        for parameter, ratio in zip(self.parameters, self.ratios):
            parameter(val * ratio)
            sleep(0.005)

class ScaledParameter(Parameter):
    def __init__(self, parameter, ratio=1,
                 name=None, label=None, units=None, **kwargs):
        if name is None:
            name = parameter.name
        if label is None:
            label = parameter.name
        if units is None:
            units = parameter.units
        super().__init__(name, label=label, units=units, **kwargs)
        self.parameter = parameter

        self.ratio = ratio
        self._meta_attrs.extend(['ratio'])

    def get(self):
        return self.parameter() / self.ratio

    def set(self, val):
        self.parameter(val * self.ratio)

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

class StoreParameter(Parameter):
    def __init__(self, shape, data_manager, formatter=None, **kwargs):
        super().__init__(name='random_parameter', shape=shape, **kwargs)
        self.data_manager = data_manager
        self.setup(formatter=formatter)


    def setup(self, formatter=None):
        data_array_set = DataArray(name='set_vals',
                                   shape=(self.shape[0],),
                                   preset_data=np.arange(self.shape[0]),
                                   is_setpoint=True)
        index0 = DataArray(name='index0', shape=self.shape,
                           preset_data=np.full(self.shape,
                                               np.arange(self.shape[-1]),
                                               dtype=np.int))
        data_array_values = DataArray(name='data_vals',
                                      shape=self.shape,
                                      set_arrays=(
                                      data_array_set, index0))

        data_mode = DataMode.PUSH_TO_SERVER

        data_folder = data_tools.get_latest_data_folder()
        loc_provider = qc.data.location.FormatLocation(
            fmt=data_folder+'/traces/#{counter}_trace_{time}')

        self.data_set = new_data(
            location=loc_provider,
            arrays=[data_array_set, index0, data_array_values],
            mode=data_mode,
            data_manager=self.data_manager, name='test_data_parameter',
            formatter=formatter)

    def get(self):
        result = np.random.randint(1, 100, size=self.shape)

        loop_indices = slice(0,self.shape[0],1)
        ids_values = {'data_vals': result}
                      # 'set_vals_set': np.arange(self.shape[0]),
                      # 'index0': None}
        self.data_manager.write('store_data', loop_indices, ids_values)
        self.data_manager.write('finalize_data')
        return result
