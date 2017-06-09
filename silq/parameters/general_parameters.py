import os
from time import time, sleep
import numpy as np

import qcodes as qc
from qcodes import config
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.data.data_set import new_data
from qcodes.data.data_array import DataArray
from silq.tools import data_tools

properties_config = config['user'].get('properties', {})
pulse_config = config['user'].get('pulses', {})


class CombinedParameter(Parameter):
    """
    Combines multiple parameters into a single parameter.
    Setting this parameter sets all underlying parameters to this value
    Getting this parameter gets the value of the first parameter
    """
    def __init__(self, parameters, name=None, label=None, unit=None, offsets=None, **kwargs):
        if name is None:
            name = '_'.join([parameter.name for parameter in parameters])
        if label is None:
            label = ' and '.join([parameter.label for parameter in parameters])
        if unit is None:
            unit = parameters[0].unit
        super().__init__(name, label=label, unit=unit, **kwargs)

        self.parameters = parameters
        self.offsets = offsets

    def get(self):
        value = self.parameters[0]()
        if self.offsets is not None:
            value -= self.offsets[0]
        self._save_val(value)
        return value

    def set(self, value):
        self._save_val(value)
        for k, parameter in enumerate(self.parameters):
            if self.offsets is not None:
                offset = self.offsets[k]
            else:
                offset = 0
            parameter(value + offset)
            sleep(0.005)


class ScaledParameter(Parameter):
    """
    Creates a new parameter that scales a previous parameter by a ratio
    Setting this parameter sets the underlying parameter multiplied by the ratio
    Getting this parameter gets the underlying parameter divided by the ratio
    """
    def __init__(self, parameter, scale=1,
                 name=None, label=None, unit=None, **kwargs):
        if name is None:
            name = parameter.name
        if label is None:
            label = parameter.label
        if unit is None:
            unit = parameter.unit
        super().__init__(name, label=label, unit=unit, **kwargs)
        self.parameter = parameter

        self.scale = scale
        self._meta_attrs.extend(['scale'])

    def get(self):
        value = self.parameter() / self.scale
        self._save_val(value)
        return value

    def set(self, val):
        value = val * self.scale
        self._save_val(val)
        self.parameter(value)


class StoreParameter(Parameter):
    """
    Stores data in a separate file (not working anymore, needs upgrading)
    """
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

        data_folder = data_tools.get_latest_data_folder()
        loc_provider = qc.data.location.FormatLocation(
            fmt=data_folder+'/traces/#{counter}_trace_{time}')

        self.data_set = new_data(
            location=loc_provider,
            arrays=[data_array_set, index0, data_array_values],
            name='test_data_parameter',
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
    def __init__(self, object, attribute, scale=None, **kwargs):
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
        self.scale = scale

    def set(self, value):
        if self.scale is not None:
            value = tuple(value / scale for scale in self.scale)
        setattr(self.object, self.attribute, value)
        self._save_val(value)

    def get(self):
        value =  getattr(self.object, self.attribute)
        if self.scale is not None:
            value = value[0] * self.scale[0]
        self._save_val(value)
        return value

class ConfigPulseAttribute(Parameter):
    def __init__(self, pulse_name, attribute, **kwargs):
        """
        Creates a parameter that can set/get an attribute from an object
        Args:
            pulse_name: name of pulse in config['pulses'] keys
            attribute: attribute to set/get
            **kwargs: Other parameter kwargs
        """
        name = kwargs.pop('name', '{}_{}'.format(pulse_name, attribute))
        super().__init__(name=name, **kwargs)

        self.pulse_name = pulse_name
        self.attribute = attribute

    def set(self, value):
        pulse_config[self.pulse_name][self.attribute] = value
        self._save_val(value)

    def get(self):
        value = pulse_config[self.pulse_name][self.attribute]
        self._save_val(value)
        return value

