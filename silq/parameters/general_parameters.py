from typing import List
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
__all__ = ['CombinedParameter', 'AttributeParameter']


class CombinedParameter(Parameter):
    """Combines multiple parameters into a single parameter.

    Setting this parameter sets all underlying parameters to this value, after
    applying possible scale and offset in that order.
    Getting this parameter gets the value of the first parameter, and applies
    offset and scale in that order.

    Args:
        parameters: Parameters to be combined.
        name: Name of ``CombinedParameter``, by default equal to the names of
            the composed parameters separated by underscores.
        label: Label of ``CombinedParameter``, by default equal to the labels of
            the composed parameters separated by ``and``. Also includes any
            scale and offset.
        unit: Parameter unit.
        offsets: Optional offset for parameters. If set, must have equal number
            of elements as parameters
        scales: Optional scale for parameters. If set, must have equal number
            of elements as parameters.
        **kwargs: Additional kwargs passed to ``Parameter``.

    Note:
        * All args are also attributes.
        * While QCoDeS already has a ``CombinedParameter``, it has some
          shortcomings which are addressed here. Maybe in the future this will
          be PR'ed to the main QCoDeS repository.
    """
    def __init__(self,
                 parameters: List[Parameter],
                 name: str = None,
                 label: str = '',
                 unit: str = None,
                 offsets: List[float] = None,
                 scales: List[float] = None,
                 **kwargs):
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

    def zero_offset(self, offset=0):
        """Use current values of parameters as offsets."""
        if self.scales is not None:
            self.offsets = [param() - offset * scale for param, scale in
                       zip(self.parameters, self.scales)]
        else:
            self.offsets = [param() for param in self.parameters]
        return self.offsets

    def calculate_individual_values(self, value):
        """Calulate values of parameters from a combined value

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


class AttributeParameter(Parameter):
    """Creates a parameter that can set/get an attribute from an object.

    Args:
        object: Object whose attribute to set/get.
        attribute: Attribute to set/get
        is_key: whether the attribute is a key in a dictionary. If not
            specified, it will check if ``AttributeParameter.object`` is a dict.
        **kwargs: Additional kwargs passed to ``Parameter``.
    """
    def __init__(self,
                 object: object,
                 attribute: str,
                 name: str = None,
                 is_key: bool = None,
                 **kwargs):
        name = name if name is not None else attribute
        super().__init__(name=name, **kwargs)

        self.object = object
        self.attribute = attribute
        self.is_key = isinstance(object, dict) if is_key is None else is_key

    def set_raw(self, value):
        if not self.is_key:
            setattr(self.object, self.attribute, value)
        else:
            self.object[self.attribute] = value

    def get_raw(self):
        if not self.is_key:
            value =  getattr(self.object, self.attribute)
        else:
            value = self.object[self.attribute]
        return value