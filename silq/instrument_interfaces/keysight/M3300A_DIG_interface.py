from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection, CombinedConnection

from qcodes.utils import validators as vals

class M3300A_DIG_Interface(InstrumentInterface):
    def __init__(self, instrument_name, **kwargs):
        super().__init__(instrument_name, **kwargs)

        # Initialize channels
        self._input_channels = {
            'ch{}'.format(k): Channel(instrument_name=self.instrument_name(),
                            name='ch{}'.format(k), input=True)
             for k in range(8)
        }

        self._channels = {
            **self._input_channels,
            'trig_in': Channel(instrument_name=self.instrument_name(),
                               name='trig_in', input=True),
        }



    def start(self):
        pass

    def stop(self):
        pass

