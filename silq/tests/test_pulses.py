import unittest
import tempfile
from copy import deepcopy


from silq.pulses import PulseSequence, DCPulse, TriggerPulse, Pulse
from silq.instrument_interfaces import Channel
from silq.meta_instruments.layout import SingleConnection
from silq.tools.config import *
import silq


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


class TestPulseSignals(unittest.TestCase):
    def setUp(self):
        self.is_set = False
        self.dict = {}

    def set_val(self, _, **kwargs):
        self.is_set = True

    def set_dict(self, _, **kwargs):
        self.dict.update(**kwargs)

    def test_set_val(self):
        self.assertFalse(self.is_set)
        self.set_val(self)
        self.assertTrue(self.is_set)

    def test_signal_emit(self):
        p = Pulse()
        p.signal.connect(self.set_dict)
        self.assertFalse('t_start' in self.dict)
        p.t_start = 1
        self.assertEqual(1, self.dict['t_start'])

    def test_subsequent_pulses(self):
        p1 = Pulse(t_start=0, t_stop=10)
        self.assertEqual(p1.t_start, 0)
        self.assertEqual(p1.duration, 10)
        self.assertEqual(p1.t_stop, 10)

        p2 = Pulse(t_start=PulseMatch(p1, 't_stop', delay=1), duration=4)
        self.assertEqual(p2.t_start, 11)
        self.assertEqual(p2.duration, 4)
        self.assertEqual(p2.t_stop, 15)

        p1.t_stop = 14
        self.assertEqual(p1.t_start, 0)
        self.assertEqual(p1.duration, 14)
        self.assertEqual(p1.t_stop, 14)

        self.assertEqual(p2.t_start, 15)
        self.assertEqual(p2.duration, 4)
        self.assertEqual(p2.t_stop, 19)

        p1.t_stop = 16
        self.assertEqual(p1.t_start, 0)
        self.assertEqual(p1.duration, 16)
        self.assertEqual(p1.t_stop, 16)

        self.assertEqual(p2.t_start, 17)
        self.assertEqual(p2.duration, 4)
        self.assertEqual(p2.t_stop, 21)

        p2.t_start = 0
        self.assertEqual(p2.t_start, 0)
        self.assertEqual(p2.duration, 4)
        self.assertEqual(p2.t_stop, 4)

        p1.t_stop = 20
        self.assertEqual(p2.t_start, 0)
        self.assertEqual(p2.duration, 4)
        self.assertEqual(p2.t_stop, 4)


class TestPulseConfig(unittest.TestCase):
    def setUp(self):
        self.signal = signal('config:env.pulses.read')

        config.clear()
        config.properties = {'default_environment': 'env'}

        self.dict = {}

        self.pulses_config = DictConfig(name='pulses',
                                        folder=None,
                                        config={'read': {}})
        config.env = {'pulses': self.pulses_config,
                      'properties': {}}
        self.pulse_config = self.pulses_config.read

    def tearDown(self):
        for key in self.pulses_config:
            signal('config:pulses.' + key).receivers = {}

    def set_dict(self, sender, **kwargs):
        self.dict.update(**kwargs)

    def test_set_item(self):
        with self.assertRaises(KeyError):
            _ = self.pulse_config['duration']
        self.pulse_config['duration'] = 1
        self.assertEqual(self.pulse_config['duration'], 1)
        self.assertEqual(self.pulse_config.duration, 1)

    def test_signal(self):
        self.signal.connect(self.set_dict)
        self.pulse_config.duration = 1
        self.assertIn('duration', self.pulse_config)
        self.assertEqual(self.pulse_config.duration, 1)

        self.pulses_config.read2 = {'t_start':
                                        'config:env.pulses.read.duration'}
        self.assertIsInstance(self.pulses_config.read2, DictConfig)

        self.assertEqual(self.pulses_config.read2.t_start, 1)

        signal('config:env.pulses.read').connect(self.set_dict)
        signal('config:env.pulses.read2').connect(self.set_dict)

        self.pulse_config.duration = 3
        self.assertEqual(self.dict['duration'], 3)
        self.assertEqual(self.dict['t_start'], 3)

        self.pulses_config.read2.t_start = 20
        self.dict = {}
        self.pulse_config.duration = 5
        self.assertEqual(self.dict['duration'], 5)
        self.assertNotIn('t_start', self.dict)

    def test_config(self):
        self.assertIsInstance(self.pulse_config, DictConfig)
        self.pulse_config.t_start = 4
        self.assertEqual(self.pulse_config.t_start, 4)
        self.assertEqual(self.pulses_config.read.t_start, 4)

        # Create new dict
        d = {'read2': {'t_start': 1, 'subdict': {'t_start': 2}}}
        dict_config = DictConfig(name='iterdict', config=d)
        self.assertIsInstance(dict_config.read2, DictConfig)
        self.assertIsInstance(dict_config.read2.subdict, DictConfig)
        self.assertEqual(dict_config.read2.t_start, 1)
        self.assertEqual(dict_config.read2.subdict.t_start, 2)

    def test_pulse_from_config(self):
        self.pulse_config.t_start = 10
        p = Pulse(name='read', duration = 10)
        pseq = PulseSequence([p])
        self.assertEqual(p.t_start, 10)
        self.assertEqual(pseq['read'].t_start, 10)

        p.t_start = 20
        self.assertEqual(p.t_start, 20)

        self.pulse_config.t_start = 0
        self.assertEqual(p.t_start, 0)
        self.assertEqual(pseq['read'].t_start, 0)

        self.pulse_config.t_start = 'config:env.pulses.read.t_stop'
        with self.assertRaises(AttributeError):
            self.pulse_config.t_start

        self.pulse_config.t_stop = 50
        self.assertEqual(self.pulse_config.t_start, 50)
        self.assertEqual(p.t_start, 50)
        self.assertEqual(pseq['read'].t_start, 50)

        self.pulse_config.t_start = 40
        self.assertEqual(self.pulse_config.t_start, 40)
        self.assertEqual(p.t_start, 40)
        self.assertEqual(pseq['read'].t_start, 40)

        self.pulse_config.t_stop = 60
        self.assertEqual(self.pulse_config.t_start, 40)
        self.assertEqual(p.t_start, 40)
        self.assertEqual(pseq['read'].t_start, 40)

    def test_pulse_from_properties_config(self):
        read_pulse = Pulse(name='read', duration=10)
        pseq = PulseSequence([read_pulse])

        self.assertEqual(read_pulse.t_skip, None)
        self.assertEqual(pseq['read'].t_skip, None)

        config.env.properties = {'t_skip': 1}
        self.assertEqual(read_pulse.t_skip, 1)
        self.assertEqual(pseq['read'].t_skip, 1)

        config.env.properties = {'t_skip': 2}
        self.assertEqual(read_pulse.t_skip, 2)
        self.assertEqual(pseq['read'].t_skip, 2)

        read_pulse = Pulse(name='read', duration=10)
        pseq = PulseSequence([read_pulse])
        self.assertEqual(read_pulse.t_skip, 2)
        self.assertEqual(pseq['read'].t_skip, 2)

        config.env.properties = {'t_skip': 1}
        self.assertEqual(read_pulse.t_skip, 1)
        self.assertEqual(pseq['read'].t_skip, 1)

    def test_pulse_attr_after_load(self):
        self.pulse_config.duration = 10
        pulse = Pulse('read')
        self.assertEqual(pulse.duration, 10)

        config.env.pulses.read.duration = 20

        # Save config to temporary folder
        with tempfile.TemporaryDirectory() as folderpath:
            config.save(folder=folderpath)
            config.env.pulses.read.duration = 10

            # Pulse duration has not yet been updated
            self.assertEqual(pulse.duration, 10)

            config.load(folderpath)
            # Pulse duration has not yet been updated
            self.assertEqual(pulse.duration, 20)


class TestPulseSequence(unittest.TestCase):
    def setUp(self):

        config.clear()
        config.properties = {}

        self.pulse_sequence = PulseSequence()

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

    def test_pulse_sequence_id(self):
        self.pulse_sequence.add(Pulse(name='read', duration=1))
        p1_read = self.pulse_sequence['read']
        self.assertIsNone(p1_read.id)

        self.pulse_sequence.add(Pulse(name='load', duration=1))
        self.assertIsNone(p1_read.id)

        self.pulse_sequence.add(Pulse(name='read', duration=1))
        self.assertEqual(p1_read.id, 0)
        self.assertEqual(self.pulse_sequence.get_pulse(name='read', id=0),
                         p1_read)
        self.assertEqual(self.pulse_sequence.get_pulse(name='read[0]'),
                         p1_read)
        p2_read = self.pulse_sequence['read[1]']
        self.assertNotEqual(p2_read, p1_read)

        self.pulse_sequence.add(Pulse(name='read', duration=1))
        p3_read = self.pulse_sequence['read[2]']
        self.assertNotEqual(p3_read, p1_read)
        self.assertNotEqual(p3_read, p2_read)


if __name__ == '__main__':
    unittest.main()