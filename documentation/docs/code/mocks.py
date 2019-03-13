import numpy as np
from qcodes import Parameter, Instrument
from time import sleep


class FakeSignalParameter(Parameter):
    def __init__(self, name,
                 microwave_instrument_name='microwave_source',
                 f0 = 42.1e9,
                 gamma_factor=1e6,
                 unit='V',
                 delay=0.2,
                 **kwargs):
        super().__init__(name, unit=unit, **kwargs)
        self._instrument = Instrument._all_instruments[microwave_instrument_name]()
        self.f0 = f0
        self.gamma_factor = gamma_factor
        self.delay = delay

    def get_raw(self):
        """Return signal according to Rabi's formula"""
        sleep(self.delay)  # Initial delay
        A = 10**(self._instrument.power() / 10)
        f = self._instrument.frequency()

        gamma = A * self.gamma_factor
        Omega = np.sqrt(gamma**2 + (f - self.f0)**2)
        return gamma**2 / Omega**2 * np.sin(Omega / gamma * np.pi / 2)**2
