import os
from time import time, sleep
import numpy as np

import qcodes as qc
from qcodes import config
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data.data_set import new_data, DataMode
from qcodes.data.data_array import DataArray
from silq.tools import data_tools

properties_config = config['user'].get('properties', {})


class CombinedParameter(Parameter):
    def __init__(self, parameters, name=None, label=None, units=None, **kwargs):
        if name is None:
            name = '_'.join([parameter.name for parameter in parameters])
        if label is None:
            label = ' and '.join([parameter.name for parameter in parameters])
        if units is None:
            units = parameters[0].units
        super().__init__(name, label=label, units=units, **kwargs)
        self.parameters = parameters

    def get(self):
        value = self.parameters[0]()
        self._save_val(value)
        return value

    def set(self, value):
        self._save_val(value)
        for parameter in self.parameters:
            parameter(value)
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
        value = self.parameter() / self.ratio
        self._save_val(value)
        return value

    def set(self, val):
        value = val * self.ratio
        self._save_val(value)
        self.parameter(value)


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


class AttributeParameter(Parameter):
    def __init__(self, object, attribute, **kwargs):
        """
        Creates a parameter that can set/get an attribute from an object
        Args:
            object: object whose attribute to set/get
            attribute: attribute to set/get
            **kwargs: Other parameter kwargs
        """
        name = kwargs.pop('name', attribute)
        super().__init__(name=name, **kwargs)

        self.object = object
        self.attribute = attribute

    def set(self, value):
        setattr(self.object, self.attribute, value)
        self._save_val(value)

    def get(self):
        value =  getattr(self.object, self.attribute)
        self._save_val(value)
        return value


class ConfigParameter(Parameter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattribute__(self, item):
        """
        Used when requesting an attribute. If the attribute is explicitly set to
        None, it will check the config if the item exists.
        Args:
            item: Attribute to be retrieved

        Returns:

        """
        value = object.__getattribute__(self, item)
        if value is not None:
            return value

        value = self._attribute_from_config(item)
        return value

    def _attribute_from_config(self, item):
        """
        Check if attribute exists somewhere in the config
        It first ill check properties config if a key matches the item
        with self.mode appended. This is only checked if the param has a mode.
        Finally, it will check if properties_config contains the item
        """
        # check if {item}_{self.mode} is in properties_config
        # if mode is None, mode_str='', in which case it checks for {item}
        item_mode = '{}{}'.format(item, self.mode_str)
        if item_mode in properties_config:
            return properties_config[item_mode]

        # Check if item is in properties config
        if item in properties_config:
            return properties_config[item]

        return None
