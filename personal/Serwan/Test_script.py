# Imports
from functools import partial
from importlib import reload
import unittest

import qcodes as qc
from qcodes import Instrument
qc.config['core']['legacy_mp'] = True
qc.loops.USE_MP = True

import silq
from silq.pulses import PulseSequence, DCPulse, TriggerPulse, SinePulse, MeasurementPulse
from silq.instrument_interfaces import get_instrument_interface, Channel
from silq.meta_instruments.chip import Chip
from silq.meta_instruments.layout import Layout, Connection, SingleConnection
from silq.meta_instruments.mock_instruments import MockArbStudio, MockPulseBlaster, MockATS, MockAcquisitionController


if __name__ == "__main__":
    # import silq
    # silq.initialize("E")WJN


    class TestPulseSequence(unittest.TestCase):
        def setUp(self):
            self.pulse_sequence = PulseSequence()

        def test_pulse_equality(self):
            pulse1 = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
            self.assertTrue(pulse1)
            pulse2 = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
            self.assertEqual(pulse1, pulse2)
            pulse3 = DCPulse(name='dc', amplitude=2.5, duration=10, t_start=0)
            self.assertNotEqual(pulse1, pulse3)

        def test_add_remove_pulse(self):
            if self.pulse_sequence:
                isempty = False
            else:
                isempty = True
            self.assertTrue(isempty)

            pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
            self.pulse_sequence.add(pulse)
            self.assertIn(pulse, self.pulse_sequence)

            if self.pulse_sequence:
                isempty = False
            else:
                isempty = True
            self.assertFalse(isempty)

            # Remove pulses
            self.pulse_sequence.clear()
            self.assertEqual(len(self.pulse_sequence.pulses), 0)

            if self.pulse_sequence:
                isempty = False
            else:
                isempty = True
            self.assertTrue(isempty)

        def test_sort(self):
            pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
            pulse2 = DCPulse(name='dc2', amplitude=1.5, duration=10, t_start=0)
            self.pulse_sequence.add([pulse1, pulse2])
            self.assertEqual(pulse2, self.pulse_sequence[0])

        def test_get_pulses(self):
            self.assertListEqual(self.pulse_sequence.get_pulses(), [])
            pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
            pulse2 = DCPulse(name='dc2', amplitude=2.5, duration=10, t_start=1)
            pulse3 = TriggerPulse(name='trig', duration=12, t_start=1)
            self.pulse_sequence.add([pulse1, pulse2, pulse3])

            subset_pulses = self.pulse_sequence.get_pulses()
            self.assertListEqual(subset_pulses, [pulse1, pulse2, pulse3])
            subset_pulses = self.pulse_sequence.get_pulses(t_start=1)
            self.assertListEqual(subset_pulses, [pulse1, pulse2, pulse3])
            subset_pulses = self.pulse_sequence.get_pulses(duration=10)
            self.assertListEqual(subset_pulses, [pulse1, pulse2])
            subset_pulses = self.pulse_sequence.get_pulses(amplitude=1.5)
            self.assertListEqual(subset_pulses, [pulse1])
            subset_pulses = self.pulse_sequence.get_pulses(amplitude=('>', 1.5))
            self.assertListEqual(subset_pulses, [pulse2])
            subset_pulses = self.pulse_sequence.get_pulses(
                amplitude=('>=', 1.5))
            self.assertListEqual(subset_pulses, [pulse1, pulse2])

            pulse = self.pulse_sequence.get_pulse(amplitude=1.5)
            self.assertEqual(pulse, pulse1)
            print(self.pulse_sequence)
            pulse = self.pulse_sequence.get_pulse(duration=12)
            self.assertEqual(pulse, pulse3)
            with self.assertRaises(RuntimeError):
                pulse = self.pulse_sequence.get_pulse(duration=10)

        def test_transition_voltages(self):
            # To test transitions, pulses must be on the same connection
            channel_out = Channel('arbstudio', 'ch1', id=1, output=True)
            channel_in = Channel('device', 'input', id=1, output=True)
            c1 = SingleConnection(output_instrument='arbstudio',
                                  output_channel=channel_out,
                                  input_instrument='device',
                                  input_channel=channel_in)
            pulses = [DCPulse(name='dc1', amplitude=0, duration=5, t_start=0,
                              connection=c1),
                      DCPulse(name='dc2', amplitude=1, duration=10, t_start=5,
                              connection=c1),
                      DCPulse(name='dc3', amplitude=2, duration=8, t_start=15,
                              connection=c1),
                      DCPulse(name='dc4', amplitude=3, duration=7, t_start=12,
                              connection=c1)]

            self.pulse_sequence.add(pulses)

            transition_voltage = self.pulse_sequence.get_transition_voltages(
                pulse=pulses[1])
            self.assertTupleEqual(transition_voltage, (0, 1))

            transition_voltage = self.pulse_sequence.get_transition_voltages(
                connection=c1, t=5)
            self.assertTupleEqual(transition_voltage, (0, 1))

            transition_voltage = self.pulse_sequence.get_transition_voltages(
                connection=c1, t=15)
            self.assertTupleEqual(transition_voltage, (1, 2))


    test = TestPulseSequence()
    test.setUp()
    test.test_transition_voltages()