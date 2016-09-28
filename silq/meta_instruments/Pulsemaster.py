import numpy as np

from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils import validators as vals


class PulseMaster(Instrument):
    shared_kwargs = ['setup']

    def __init__(self, name, setup, **kwargs):
        super().__init__(name, **kwargs)

        self.setup = setup

        self.instruments = {instrument.name: instrument
                            for instrument in self.setup.instruments}

        self.add_parameter('trigger_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self.instruments.keys()))

        self.add_parameter('acquisition_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self.instruments.keys()))
