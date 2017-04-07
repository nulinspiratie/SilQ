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

    def get_final_additional_pulses(self, **kwargs):
        return []

    def setup(self, **kwargs):
        self.instrument.RF_output('off')

        self.instrument.phase_modulation('off')

        frequencies = [pulse.implement()['frequency']
                       for pulse in self._pulse_sequence]

        powers = [pulse.power for pulse in self._pulse_sequence]
        assert len(set(frequencies)) == 1, "Cannot handle multiple frequencies"
        assert len(set(powers)) == 1, "Cannot handle multiple pulse powers"

        if any('deviation' in pulse.implement()
               for pulse in self._pulse_sequence):
            deviations = [pulse.implement()['deviation']
                    for pulse in self._pulse_sequence]
            assert len(set(deviations)) == 1, "Cannot handle multiple " \
                                              "deviations"
            deviation = deviations[0]
            self.instrument.frequency_deviation(deviation)
            self.instrument.frequency_modulation('on')
            self.instrument.frequency_modulation_source(
                self.modulation_channel())
        else:
            self.instrument.frequency_deviation(0)
            self.instrument.frequency_modulation('off')

        frequency = frequencies[0]
        power = powers[0]
        self.instrument.frequency(frequency)
        self.instrument.power(power)

        self.instrument.pulse_modulation('on')
        self.instrument.pulse_modulation_source('ext')
        self.instrument.output_modulation('on')

    def start(self):
        pass

    def stop(self):
        pass

