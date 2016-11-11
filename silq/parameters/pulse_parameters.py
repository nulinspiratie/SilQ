from time import sleep

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter

class PulsePropertyParameter(Parameter):
    def __init__(self, pulse, property, **kwargs):
        super().__init__(**kwargs)

        self.pulse = pulse
        self.property = property

    def set(self, value):
        setattr(self.pulse, self.property, value)

    def get(self):
        return getattr(self.pulse, self.property)

class ParameterPulsePropertyParameter(Parameter):
    def __init__(self, parameter, pulse_name, property, **kwargs):
        super().__init__(**kwargs)

        self.parameter = parameter
        self.pulse_name = pulse_name
        self.property = property

    def set(self, value):
        setattr(self.parameter.pulse_sequence[self.pulse_name], self.property,
                value)

    def get(self):
        print(self.parameter.pulse_sequence[self.pulse_name])
        print(getattr(self.parameter.pulse_sequence[self.pulse_name], self.property))
        return getattr(self.parameter.pulse_sequence[self.pulse_name], self.property)