from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes import validators as vals
import numpy as np

class VirtualVoltage(Parameter):
    def __init__(self, name, ratios, **kwargs):
        super().__init__(**kwargs)

class VoltageControl(Instrument):
    shared_kwargs = ['parameters']
    def __init__(self, name, parameters, offset=None, **kwargs):
        super().__init__(name, **kwargs)

        self.real_parameters = parameters

        if offset is None:
            offset = [0 for parameter in parameters]
        assert len(offset) == len(parameters), \
            "Length of offset does not equal length of parameters"
        self.add_parameter(name='offset',
                           initial_value=offset,
                           parameter_class=ManualParameter,
                           vals=List(elements=vals.Numbers(),
                                     length=len(parameters)))

        self.add_parameter(name='parameter1',
                           get_cmd=self.real_parameters[0])

    def add_virtual_voltage(self, name, ratios):
        pass

class List(vals.Validator):
    valid_types = (list, np.array, set)

    def __init__(self, elements=None, length=None):
        self.elements = elements
        self.length = length

    def validate(self, value, context=''):
        if not isinstance(value, self.valid_types):
            raise TypeError(
                '{} is not list-like; {}'.format(repr(value), context))

        if self.length is not None and len(value) is not self.length:
            raise ValueError(
                '{} does not have length {}'.format(value, self.length))

        if self.elements is not None:
            for element in value:
                self.elements.validate(element)

    def __repr__(self):
        return '<List({},len({}))>'.format(self.elements, self.length)