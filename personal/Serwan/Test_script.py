from functools import partial
from importlib import reload
import unittest

from silq.pulses import PulseSequence, DCPulse, TriggerPulse, SinePulse
from silq.instrument_interfaces import get_instrument_interface
from silq.meta_instruments.chip import Chip
from silq.meta_instruments.layout import Layout
from silq.meta_instruments.mock_instruments import MockArbStudio

from qcodes import Instrument
from functools import partial
from importlib import reload
import unittest

from silq.pulses import PulseSequence, DCPulse, TriggerPulse, SinePulse
from silq.instrument_interfaces import get_instrument_interface
from silq.meta_instruments.chip import Chip
from silq.meta_instruments.layout import Layout
from silq.meta_instruments.mock_instruments import MockArbStudio, MockPulseBlaster



if __name__ == "__main__":

    class TestLayout(unittest.TestCase):
        def setUp(self):
            self.arbstudio = MockArbStudio(name='arbstudio')
            self.pulseblaster = MockPulseBlaster(name='pulseblaster')
            self.chip = Chip(name='chip')

            self.instruments = [self.arbstudio, self.pulseblaster, self.chip]
            self.instrument_interfaces = {
            instrument.name: get_instrument_interface(instrument)
            for instrument in self.instruments}
            self.layout = Layout(name='Layout',
                                 instrument_interfaces=list(
                                     self.instrument_interfaces.values()),
                                 server_name='layout')
            self.instruments += [self.layout]

            self.arbstudio.silent(True)
            self.pulseblaster.silent(True)

            # print('Layout instruments: {}'.format(self.layout.instruments()))

            # instruments that need to be closed during tear down

        def tearDown(self):
            for instrument in self.instruments:
                instrument.close()
            for instrument_interface in self.instrument_interfaces.values():
                instrument_interface.close()
        def test_arbstudio_pulseblaster(self):
            self.layout.primary_instrument('pulseblaster')
            self.layout.add_connection(output_arg='arbstudio.ch1',
                                       input_arg='chip.TGAC')
            self.layout.add_connection(output_arg='arbstudio.ch2',
                                       input_arg='chip.DF', default=True)

            self.layout.add_connection(output_arg='pulseblaster.ch1',
                                       input_arg='arbstudio.trig_in',
                                       trigger=True)

            trigger_connections = self.layout.get_connections(
                input_instrument='arbstudio', trigger=True)
            trigger_connection = trigger_connections[0]

            pulse_sequence = PulseSequence()
            empty_pulse = DCPulse(name='empty', t_start=0, duration=10,
                                  amplitude=1.5)
            #         load_pulse = DCPulse(name='load', t_start=10, duration=10, amplitude=-1.5)
            #         read_pulse = DCPulse(name='read', t_start=20, duration=10, amplitude=0)
            #         pulses = [empty_pulse, load_pulse, read_pulse]
            pulses = [empty_pulse]
            for pulse in pulses:
                pulse_sequence.add(pulse)

            self.layout.target_pulse_sequence(pulse_sequence)
            print(self.instrument_interfaces['arbstudio'].pulse_sequence())
            print(self.instrument_interfaces['pulseblaster'].pulse_sequence())

    test = TestLayout()
    test.setUp()
    test.test_arbstudio_pulseblaster()
    test.tearDown()