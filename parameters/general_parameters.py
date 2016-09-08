from time import sleep

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter


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
