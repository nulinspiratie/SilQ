import unittest
import tempfile
from copy import copy, deepcopy

from silq.pulses import PulseSequence, DCPulse, TriggerPulse, Pulse
from silq.instrument_interfaces import Channel
from silq.meta_instruments.layout import SingleConnection
from silq.tools.config import *
import silq

import qcodes as qc

class TestPulse(unittest.TestCase):
    def test_pulse_equality(self):
        pulse1 = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        self.assertTrue(pulse1)
        pulse2 = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        self.assertEqual(pulse1, pulse2)
        pulse3 = DCPulse(name='dc', amplitude=2.5, duration=10, t_start=0)
        self.assertNotEqual(pulse1, pulse3)

    def test_pulse_id(self):
        p = Pulse(name='read')
        self.assertEqual(p.id, None)

        self.assertEqual(p.full_name, 'read')
        p.id = 0
        self.assertEqual(p.name, 'read')
        self.assertEqual(p.full_name, 'read[0]')
        self.assertTrue(p.satisfies_conditions(name='read', id=0))
        self.assertTrue(p.satisfies_conditions(name='read[0]'))

    def test_pulse_duration_t_stop(self):
        p = Pulse(t_start=1, t_stop=3)
        self.assertEqual(p.duration, 2)

        p.duration = 4
        self.assertEqual(p.t_stop, 5)


class TestPulseSignals(unittest.TestCase):
    def test_signal_emit(self):
        p = Pulse(t_start=1, t_stop=2)

        p2 = Pulse(duration=3)
        p['t_start'].connect(p2['t_start'], offset=5)
        self.assertEqual(p2.t_start, 6)
        self.assertEqual(p2.duration, 3)
        self.assertEqual(p2.t_stop, 9)

        p.t_start = 3
        self.assertEqual(p.t_stop, 4)
        self.assertEqual(p2.t_start, 8)
        self.assertEqual(p2.t_stop, 11)

        p2.t_start = 5
        self.assertEqual(p2.t_start, 5)
        self.assertEqual(p2.duration, 3)
        self.assertEqual(p2.t_stop, 8)

        # The signal connection remains even after changing its value
        p.t_start = 10
        self.assertEqual(p2.t_start, 15)

    def test_signal_copy(self):
        p = Pulse(t_start=1)

        p2 = Pulse()
        p['t_start'].connect(p2['t_start'])
        self.assertEqual(p2.t_start, 1)

        p3 = copy(p2)
        self.assertEqual(p3.t_start, 1)

        p4 = deepcopy(p2)
        self.assertEqual(p4.t_start, 1)

        p.t_start = 2
        self.assertEqual(p2.t_start, 2)
        self.assertEqual(p3.t_start, 1)
        self.assertEqual(p4.t_start, 1)


class TestPulseConfig(unittest.TestCase):
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

    def test_set_item(self):
        silq.environment = None

        p = Pulse('read')
        self.assertEqual(p.t_start, 0)
        self.assertEqual(p.t_stop, 10)

        self.config.pulses.read.t_start = 5

        self.assertEqual(p.t_start, 5)
        self.assertEqual(p.t_stop, 15)

    def test_set_item_environment(self):
        silq.environment = 'env1'

        p = Pulse('read')
        self.assertEqual(p.t_start, None)
        self.assertEqual(p.t_stop, None)

        self.config.pulses.read.t_start = 5
        self.assertEqual(p.t_start, None)
        self.assertEqual(p.t_stop, None)

        self.config.env1 = {'pulses': {'read': {'t_start': 5}}}
        self.assertEqual(p.t_start, 5)
        self.assertEqual(p.t_stop, None)

        self.config.env1.pulses.read.t_stop = 10
        self.assertEqual(p.t_start, 5)
        self.assertEqual(p.t_stop, 10)

#
# class TestPulseSequence(unittest.TestCase):
#     def setUp(self):
#
#         config.clear()
#         config.properties = {}
#
#         self.pulse_sequence = PulseSequence()
#
#     def test_add_remove_pulse(self):
#         if self.pulse_sequence:
#             isempty = False
#         else:
#             isempty = True
#         self.assertTrue(isempty)
#
#         pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
#         self.pulse_sequence.add(pulse)
#         self.assertIn(pulse, self.pulse_sequence)
#
#         if self.pulse_sequence:
#             isempty = False
#         else:
#             isempty = True
#         self.assertFalse(isempty)
#
#         # Remove pulses
#         self.pulse_sequence.clear()
#         self.assertEqual(len(self.pulse_sequence.pulses), 0)
#
#         if self.pulse_sequence:
#             isempty = False
#         else:
#             isempty = True
#         self.assertTrue(isempty)
#
#     def test_sort(self):
#         pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
#         pulse2 = DCPulse(name='dc2', amplitude=1.5, duration=10, t_start=0)
#         self.pulse_sequence.add(pulse1, pulse2)
#
#         self.assertEqual(pulse2, self.pulse_sequence[0])
#
#     def test_get_pulses(self):
#         self.assertListEqual(self.pulse_sequence.get_pulses(), [])
#         pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
#         pulse2 = DCPulse(name='dc2', amplitude=2.5, duration=10, t_start=1)
#         pulse3 = TriggerPulse(name='trig', duration=12, t_start=1)
#         self.pulse_sequence.add(pulse1, pulse2, pulse3)
#
#         subset_pulses = self.pulse_sequence.get_pulses()
#         self.assertListEqual(subset_pulses, [pulse1, pulse2, pulse3])
#         subset_pulses = self.pulse_sequence.get_pulses(t_start=1)
#         self.assertListEqual(subset_pulses, [pulse1, pulse2, pulse3])
#         subset_pulses = self.pulse_sequence.get_pulses(duration=10)
#         self.assertListEqual(subset_pulses, [pulse1, pulse2])
#         subset_pulses = self.pulse_sequence.get_pulses(amplitude=1.5)
#         self.assertListEqual(subset_pulses, [pulse1])
#         subset_pulses = self.pulse_sequence.get_pulses(amplitude=('>', 1.5))
#         self.assertListEqual(subset_pulses, [pulse2])
#         subset_pulses = self.pulse_sequence.get_pulses(amplitude=('>=', 1.5))
#         self.assertListEqual(subset_pulses, [pulse1, pulse2])
#
#         pulse = self.pulse_sequence.get_pulse(amplitude=1.5)
#         self.assertEqual(pulse, pulse1)
#         pulse = self.pulse_sequence.get_pulse(duration=12)
#         self.assertEqual(pulse, pulse3)
#         with self.assertRaises(RuntimeError):
#             self.pulse_sequence.get_pulse(duration=10)
#
#     def test_transition_voltages(self):
#         # To test transitions, pulses must be on the same connection
#         channel_out = Channel('arbstudio', 'ch1', id=1, output=True)
#         channel_in = Channel('device', 'input', id=1, output=True)
#         c1 = SingleConnection(output_instrument='arbstudio',
#                               output_channel=channel_out,
#                               input_instrument='device',
#                               input_channel=channel_in)
#         pulses = [DCPulse(name='dc1', amplitude=0, duration=5, t_start=0,
#                           connection=c1),
#                   DCPulse(name='dc2', amplitude=1, duration=10, t_start=5,
#                           connection=c1),
#                   DCPulse(name='dc3', amplitude=2, duration=8, t_start=15,
#                           connection=c1),
#                   DCPulse(name='dc4', amplitude=3, duration=7, t_start=12,
#                           connection=c1)]
#
#         self.pulse_sequence.add(*pulses)
#
#         self.assertRaises(TypeError, self.pulse_sequence.get_transition_voltages)
#         self.assertRaises(TypeError, self.pulse_sequence.get_transition_voltages,
#                           connection=c1)
#         self.assertRaises(TypeError, self.pulse_sequence.get_transition_voltages,
#                           t=5)
#
#         transition_voltage = self.pulse_sequence.get_transition_voltages(
#             pulse=pulses[1])
#         self.assertTupleEqual(transition_voltage, (0, 1))
#
#         transition_voltage = self.pulse_sequence.get_transition_voltages(
#             connection=c1, t=5)
#         self.assertTupleEqual(transition_voltage, (0, 1))
#
#         transition_voltage = self.pulse_sequence.get_transition_voltages(
#             connection=c1, t=15)
#         self.assertTupleEqual(transition_voltage, (1, 2))
#
#     def test_pulse_sequence_id(self):
#         self.pulse_sequence.add(Pulse(name='read', duration=1))
#         p1_read = self.pulse_sequence['read']
#         self.assertIsNone(p1_read.id)
#
#         self.pulse_sequence.add(Pulse(name='load', duration=1))
#         self.assertIsNone(p1_read.id)
#
#         self.pulse_sequence.add(Pulse(name='read', duration=1))
#         self.assertEqual(p1_read.id, 0)
#         self.assertEqual(self.pulse_sequence.get_pulse(name='read', id=0),
#                          p1_read)
#         self.assertEqual(self.pulse_sequence.get_pulse(name='read[0]'),
#                          p1_read)
#         p2_read = self.pulse_sequence['read[1]']
#         self.assertNotEqual(p2_read, p1_read)
#
#         self.pulse_sequence.add(Pulse(name='read', duration=1))
#         p3_read = self.pulse_sequence['read[2]']
#         self.assertNotEqual(p3_read, p1_read)
#         self.assertNotEqual(p3_read, p2_read)


if __name__ == '__main__':
    unittest.main()