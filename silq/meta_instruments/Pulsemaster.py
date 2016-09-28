import numpy as np

from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils import validators as vals


class PulseMaster(Instrument):
    shared_kwargs = ['layout']

    def __init__(self, name, layout, **kwargs):
        super().__init__(name, **kwargs)

        self.layout = layout

        self.instruments = {instrument.name: instrument
                            for instrument in self.layout.instruments}

        self.add_parameter('pulse_sequence',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Anything())


