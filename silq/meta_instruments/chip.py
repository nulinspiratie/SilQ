from qcodes import Instrument

class Chip(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        