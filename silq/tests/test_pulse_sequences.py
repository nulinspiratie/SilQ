import unittest
import tempfile
from copy import copy, deepcopy

from silq.pulses import PulseSequence, DCPulse, TriggerPulse, Pulse
from silq.instrument_interfaces import Channel
from silq.meta_instruments.layout import SingleConnection
from silq.tools.config import *
import silq

import qcodes as qc


class TestPulseSequence(unittest.TestCase):
    def test_pulse_sequence_bool(self):
        pulse_sequence = PulseSequence()
        self.assertFalse(pulse_sequence)

        pulse_sequence.add(Pulse(duration=5))
        self.assertTrue(pulse_sequence)

        pulse_sequence.clear()
        self.assertFalse(pulse_sequence)

    def test_add_remove_pulse(self):
        pulse_sequence = PulseSequence()
        self.assertFalse(pulse_sequence)

        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        self.assertIn(pulse, pulse_sequence)
        self.assertTrue(pulse_sequence)

        # Remove pulses using clear
        pulse_sequence.clear()
        self.assertEqual(len(pulse_sequence.pulses), 0)
        self.assertFalse(pulse_sequence)

        # Remove pulses using .remove
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        pulse_sequence.remove(pulse)
        self.assertEqual(len(pulse_sequence.pulses), 0)
        self.assertFalse(pulse_sequence)

    def test_sort(self):
        pulse_sequence = PulseSequence()
        pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
        pulse2 = DCPulse(name='dc2', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse1, pulse2)

        self.assertEqual(pulse_sequence[0], pulse2)

    def test_pulse_sequence_id(self):
        pulse_sequence = PulseSequence()
        pulse_sequence.add(Pulse(name='read', duration=1))
        p1_read = pulse_sequence['read']
        self.assertIsNone(p1_read.id)

        pulse_sequence.add(Pulse(name='load', duration=1))
        self.assertIsNone(p1_read.id)

        pulse_sequence.add(Pulse(name='read', duration=1))
        self.assertEqual(p1_read.id, 0)
        self.assertEqual(pulse_sequence.get_pulse(name='read', id=0),
                         p1_read)
        self.assertEqual(pulse_sequence.get_pulse(name='read[0]'),
                         p1_read)
        p2_read = pulse_sequence['read[1]']
        self.assertNotEqual(p2_read, p1_read)

        pulse_sequence.add(Pulse(name='read', duration=1))
        p3_read = pulse_sequence['read[2]']
        self.assertNotEqual(p3_read, p1_read)
        self.assertNotEqual(p3_read, p2_read)


class TestPulseSequenceOld(unittest.TestCase):
    def setUp(self):
        self.silq_environment = silq.environment
        self.silq_config = silq.config

        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2'],
            'properties': {},
            'env1': {'properties': {'x': 1, 'y': 2}}}
        self.config = DictConfig('cfg', config=self.d)
        qc.config.user.silq_config = silq.config = self.config

    def tearDown(self):
        silq.environment = self.silq_environment
        qc.config.user.silq_config = silq.config = self.silq_config

    def test_sort(self):
        pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
        pulse2 = DCPulse(name='dc2', amplitude=1.5, duration=10, t_start=0)
        self.pulse_sequence.add(pulse1, pulse2)

        self.assertEqual(pulse2, self.pulse_sequence[0])

    def test_get_pulses(self):
        self.assertListEqual(self.pulse_sequence.get_pulses(), [])
        pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
        pulse2 = DCPulse(name='dc2', amplitude=2.5, duration=10, t_start=1)
        pulse3 = TriggerPulse(name='trig', duration=12, t_start=1)
        self.pulse_sequence.add(pulse1, pulse2, pulse3)

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
        subset_pulses = self.pulse_sequence.get_pulses(amplitude=('>=', 1.5))
        self.assertListEqual(subset_pulses, [pulse1, pulse2])

        pulse = self.pulse_sequence.get_pulse(amplitude=1.5)
        self.assertEqual(pulse, pulse1)
        pulse = self.pulse_sequence.get_pulse(duration=12)
        self.assertEqual(pulse, pulse3)
        with self.assertRaises(RuntimeError):
            self.pulse_sequence.get_pulse(duration=10)

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

        self.pulse_sequence.add(*pulses)

        self.assertRaises(TypeError, self.pulse_sequence.get_transition_voltages)
        self.assertRaises(TypeError, self.pulse_sequence.get_transition_voltages,
                          connection=c1)
        self.assertRaises(TypeError, self.pulse_sequence.get_transition_voltages,
                          t=5)

        transition_voltage = self.pulse_sequence.get_transition_voltages(
            pulse=pulses[1])
        self.assertTupleEqual(transition_voltage, (0, 1))

        transition_voltage = self.pulse_sequence.get_transition_voltages(
            connection=c1, t=5)
        self.assertTupleEqual(transition_voltage, (0, 1))

        transition_voltage = self.pulse_sequence.get_transition_voltages(
            connection=c1, t=15)
        self.assertTupleEqual(transition_voltage, (1, 2))


if __name__ == '__main__':
    unittest.main()