from qcodes import Instrument
from qcodes.utils import validators as vals


class Chip(Instrument):
    """Meta instrument representing chip in the fridge
    
    This instrument is only used for as a`Connection` endpoint"""
    def __init__(self, name, channels, **kwargs):
        super().__init__(name, **kwargs)
        self.add_parameter(name='channels',
                           set_cmd=None,
                           initial_value=channels,
                           vals=vals.Anything())
