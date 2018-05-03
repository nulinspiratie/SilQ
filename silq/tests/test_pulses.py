import unittest
from copy import copy, deepcopy

from silq.pulses import DCPulse, Pulse
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

    def test_pulse_no_id(self):
        p = Pulse('name')
        p.id = None


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


if __name__ == '__main__':
    unittest.main()