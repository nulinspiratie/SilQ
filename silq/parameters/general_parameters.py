from time import sleep
import numpy as np

import qcodes as qc
from qcodes.instrument.parameter import Parameter
from qcodes.data.data_set import new_data
from qcodes.data.data_array import DataArray

from silq import config
from silq.tools import data_tools

properties_config = config.get('properties', {})
pulse_config = config.get('pulses', {})
__all__ = ['CombinedParameter', 'ScaledParameter', 'StoreParameter',
           'AttributeParameter', 'ConfigPulseAttribute']


class CombinedParameter(Parameter):
    """
    Combines multiple parameters into a single parameter.
    Setting this parameter sets all underlying parameters to this value
    Getting this parameter gets the value of the first parameter
    """
    def __init__(self, parameters, name=None, label='', unit=None, offsets=None,
                 scales=None, **kwargs):
        if name is None:
            name = '_'.join([parameter.name for parameter in parameters])

        self.label = None
        if unit is None:
            unit = parameters[0].unit

        self.parameters = parameters
        self.offsets = offsets
        self.scales = scales

        super().__init__(name, label=label, unit=unit, **kwargs)

    @property
    def label(self):
        if self._label:
            return self._label

        if self.scales is None and self.offsets is None:
            return ' and '.join([parameter.label for parameter in self.parameters])
        else:
            labels = []
            for k, parameter in enumerate(self.parameters):
                if self.scales is not None and self.scales[k] != 1:
                    label = f'{self.scales[k]:.3g} * {parameter.name}'
                else:
                    label = parameter.name

                if self.offsets is not None:
                    label += f' + {self.offsets[k]:.4g}'

                labels.append(label)

            return f'({", ".join(labels)})'

    @label.setter
    def label(self, label):
        self._label = label

    def zero_offset(self):
        self.offsets = [param() for param in self.parameters]

    def calculate_individual_values(self, value):
        """
        Calulate values of parameters from a combined value
        Args:
            value: combined value

        Returns:
            list of values for each parameter
        """
        vals = []
        for k, parameter in enumerate(self.parameters):
            val = value
            if self.scales is not None:
                val *= self.scales[k]
            if self.offsets is not None:
                val += self.offsets[k]
            vals.append(val)

        return vals

    def get_raw(self):
        value = self.parameters[0]()
        if self.offsets is not None:
            value -= self.offsets[0]
        if self.scales is not None:
            value /= self.scales[0]
        return value

    def set_raw(self, value):
        individual_values = self.calculate_individual_values(value)
        for parameter, val in zip(self.parameters, individual_values):
            parameter(val)
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

    def get_raw(self):
        value = self.parameter() / self.scale
        self._save_val(value)
        return value

    def set_raw(self, val):
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

        data_folder = data_tools.get_data_folder()
        loc_provider = qc.data.location.FormatLocation(
            fmt=data_folder+'/traces/#{counter}_trace_{time}')

        self.data_set = new_data(
            location=loc_provider,
            arrays=[data_array_set, index0, data_array_values],
            name='test_data_parameter',
            formatter=formatter)

    def get_raw(self):
        result = np.random.randint(1, 100, size=self.shape)

        loop_indices = slice(0,self.shape[0],1)
        ids_values = {'data_vals': result}
                      # 'set_vals_set': np.arange(self.shape[0]),
                      # 'index0': None}
        self.data_manager.write('store_data', loop_indices, ids_values)
        self.data_manager.write('finalize_data')
        return result


class AttributeParameter(Parameter):
    def __init__(self, object, attribute, name=None, scale=None, is_key=None,
                 **kwargs):
        """
        Creates a parameter that can set/get an attribute from an object
        Args:
            object: object whose attribute to set/get
            attribute: attribute to set/get
            is_key: whether the attribute is a key in a dictionary
            **kwargs: Other parameter kwargs
        """
        name = name if name is not None else attribute
        super().__init__(name=name, **kwargs)

        self.object = object
        self.attribute = attribute
        self.scale = scale
        self.is_key = isinstance(object, dict) if is_key is None else is_key

    def set_raw(self, value):
        if self.scale is not None:
            value = tuple(value / scale for scale in self.scale)
        if not self.is_key:
            setattr(self.object, self.attribute, value)
        else:
            self.object[self.attribute] = value
        self._save_val(value)

    def get_raw(self):
        if not self.is_key:
            value =  getattr(self.object, self.attribute)
        else:
            value = self.object[self.attribute]
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

    def set_raw(self, value):
        pulse_config[self.pulse_name][self.attribute] = value
        self._save_val(value)

    def get_raw(self):
        value = pulse_config[self.pulse_name][self.attribute]
        self._save_val(value)
        return value
