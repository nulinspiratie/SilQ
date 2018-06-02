import unittest
import tempfile
from copy import copy, deepcopy

from silq.pulses import PulseSequence, DCPulse, TriggerPulse, Pulse
from silq.instrument_interfaces import Channel
from silq.meta_instruments.layout import SingleConnection


class Registrar:
    def __init__(self):
        self.values = []

    def __call__(self, value):
        self.values.append(value)


class TestPulseSequence(unittest.TestCase):
    def test_pulse_sequence_bool(self):
        pulse_sequence = PulseSequence()
        self.assertFalse(pulse_sequence)

        pulse_sequence.add(Pulse(duration=5))
        self.assertTrue(pulse_sequence)

        pulse_sequence.clear()
        self.assertFalse(pulse_sequence)

    def test_pulse_full_name(self):
        p = Pulse('pulse1')
        self.assertEqual(p.full_name, 'pulse1')
        p.id = 2
        self.assertEqual(p.full_name, 'pulse1[2]')

        p = DCPulse('pulse2')
        self.assertEqual(p.full_name, 'pulse2')
        p.id = 2
        self.assertEqual(p.full_name, 'pulse2[2]')

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

    def test_pulse_no_duration_error(self):
        p = Pulse()
        with self.assertRaises(AssertionError):
            PulseSequence([p])

    def test_get_pulses(self):
        pulse_sequence = PulseSequence()
        self.assertListEqual(pulse_sequence.get_pulses(), [])
        pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
        pulse2 = DCPulse(name='dc2', amplitude=2.5, duration=10, t_start=1)
        pulse3 = TriggerPulse(name='trig', duration=12, t_start=1)
        pulse_sequence.add(pulse1, pulse2, pulse3)

        subset_pulses = pulse_sequence.get_pulses()
        self.assertListEqual(subset_pulses, [pulse1, pulse2, pulse3])
        subset_pulses = pulse_sequence.get_pulses(t_start=1)
        self.assertListEqual(subset_pulses, [pulse1, pulse2, pulse3])
        subset_pulses = pulse_sequence.get_pulses(duration=10)
        self.assertListEqual(subset_pulses, [pulse1, pulse2])
        subset_pulses = pulse_sequence.get_pulses(amplitude=1.5)
        self.assertListEqual(subset_pulses, [pulse1])
        subset_pulses = pulse_sequence.get_pulses(amplitude=('>', 1.5))
        self.assertListEqual(subset_pulses, [pulse2])
        subset_pulses = pulse_sequence.get_pulses(amplitude=('>=', 1.5))
        self.assertListEqual(subset_pulses, [pulse1, pulse2])

        pulse = pulse_sequence.get_pulse(amplitude=1.5)
        self.assertEqual(pulse, pulse1)
        pulse = pulse_sequence.get_pulse(duration=12)
        self.assertEqual(pulse, pulse3)
        with self.assertRaises(RuntimeError):
            pulse_sequence.get_pulse(duration=10)

    def test_get_pulse(self):
        pulse_sequence = PulseSequence()
        p = Pulse('p1', duration=1)
        p1_added, = pulse_sequence.add(p)

        self.assertIs(pulse_sequence.get_pulse(name='p1'), p1_added)

        p2_added, = pulse_sequence.add(p)
        with self.assertRaises(RuntimeError):
            pulse_sequence.get_pulse(name='p1')
        self.assertIs(pulse_sequence.get_pulse(name='p1[0]'), p1_added)
        self.assertIs(pulse_sequence.get_pulse(name='p1[1]'), p2_added)

    def test_get_pulses_connection_label(self):
        pulse_sequence = PulseSequence()
        pulse1, pulse2 = pulse_sequence.add(
            Pulse('pulse1', duration=1, connection_label='connection'),
            Pulse('pulse1', duration=2)
        )

        retrieved_pulse = pulse_sequence.get_pulse(
            connection_label='connection')
        self.assertEqual(retrieved_pulse, pulse1)

        retrieved_pulse = pulse_sequence.get_pulse(name='pulse1',
                                                   connection_label='connection')
        self.assertEqual(retrieved_pulse, pulse1)

    def test_get_pulses_connection_label_from_connection(self):
        connection = SingleConnection(output_instrument='ins1',
                                      output_channel=Channel('ins1', 'ch1'),
                                      input_instrument='ins2',
                                      input_channel=Channel('ins1', 'ch1'),
                                      label='connection')
        pulse_sequence = PulseSequence()
        pulse1, pulse2 = pulse_sequence.add(
            Pulse('pulse1', duration=1, connection=connection),
            Pulse('pulse1', duration=2)
        )

        retrieved_pulse = pulse_sequence.get_pulse(
            connection_label='connection')
        self.assertEqual(retrieved_pulse, pulse1)

        retrieved_pulse = pulse_sequence.get_pulse(name='pulse1',
                                                   connection_label='connection')
        self.assertEqual(retrieved_pulse, pulse1)

        pulse2.connection_label = 'connection'
        self.assertEqual(
            len(pulse_sequence.get_pulses(connection_label='connection')), 2)

    def test_get_pulses_connection(self):
        connection = SingleConnection(output_instrument='ins1',
                                      output_channel=Channel('ins1', 'ch1'),
                                      input_instrument='ins2',
                                      input_channel=Channel('ins1', 'ch1'),
                                      label='connection')
        pulse_sequence = PulseSequence()
        pulse1, pulse2 = pulse_sequence.add(
            Pulse('pulse1', duration=1, connection=connection),
            Pulse('pulse1', duration=2)
        )

        retrieved_pulse = pulse_sequence.get_pulse(
            connection=connection)
        self.assertEqual(retrieved_pulse, pulse1)

        retrieved_pulse = pulse_sequence.get_pulse(name='pulse1',
                                                   connection_label='connection')
        self.assertEqual(retrieved_pulse, pulse1)

        pulse2.connection_label = 'connection'
        self.assertEqual(
            len(pulse_sequence.get_pulses(connection=connection)), 2)

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

        pulse_sequence = PulseSequence(pulses)

        self.assertRaises(TypeError, pulse_sequence.get_transition_voltages)
        self.assertRaises(TypeError, pulse_sequence.get_transition_voltages,
                          connection=c1)
        self.assertRaises(TypeError, pulse_sequence.get_transition_voltages,
                          t=5)

        transition_voltage = pulse_sequence.get_transition_voltages(
            pulse=pulses[1])
        self.assertTupleEqual(transition_voltage, (0, 1))

        transition_voltage = pulse_sequence.get_transition_voltages(
            connection=c1, t=5)
        self.assertTupleEqual(transition_voltage, (0, 1))

        transition_voltage = pulse_sequence.get_transition_voltages(
            connection=c1, t=15)
        self.assertTupleEqual(transition_voltage, (1, 2))

    def test_pulse_sequence_duration(self):
        pulse_sequence = PulseSequence()
        pulse_sequence.duration
        self.assertEqual(pulse_sequence.duration, 0)

        pulse_sequence.duration = None
        self.assertEqual(pulse_sequence.duration, 0)

        pulse_sequence.duration = 1
        self.assertEqual(pulse_sequence.duration, 1)

        pulse_sequence.add(DCPulse(duration=5))
        self.assertEqual(pulse_sequence.duration, 5)

        pulse_sequence.clear()
        self.assertEqual(pulse_sequence.duration, 0)

    def test_t_list(self):
        pulse_sequence = PulseSequence()
        self.assertEqual(pulse_sequence.t_list, [0])
        self.assertEqual(pulse_sequence.t_start_list, [])
        self.assertEqual(pulse_sequence.t_stop_list, [])

        pulse_sequence.add(Pulse(t_start=1, duration=5))
        self.assertEqual(pulse_sequence.t_list, [1, 6])
        self.assertEqual(pulse_sequence.t_start_list, [1])
        self.assertEqual(pulse_sequence.t_stop_list, [6])

        pulse_sequence.clear()
        self.assertEqual(pulse_sequence.t_list, [0])
        self.assertEqual(pulse_sequence.t_start_list, [])
        self.assertEqual(pulse_sequence.t_stop_list, [])

    def test_getitem(self):
        pulse_sequence = PulseSequence()
        pulse1, = pulse_sequence.add(DCPulse('DC', duration=1))

        self.assertIs(pulse_sequence['DC'], pulse1)
        self.assertIs(pulse_sequence['duration'],
                      pulse_sequence.parameters['duration'])

        pulse1.id=0
        self.assertIs(pulse_sequence['DC'], pulse1)
        self.assertIs(pulse_sequence['DC[0]'], pulse1)

        pulse2, = pulse_sequence.add(DCPulse('DC', duration=1))
        self.assertIs(pulse_sequence['DC[0]'], pulse1)
        self.assertIs(pulse_sequence['DC[1]'], pulse2)
        with self.assertRaises(KeyError):
            pulse_sequence['DC']

    def test_snapshot(self):
        pulse_sequence = PulseSequence()
        snapshot = pulse_sequence.snapshot()
        for parameter_name, parameter in pulse_sequence.parameters.items():
            if parameter.unit:
                parameter_name += f' ({parameter.unit})'
            self.assertEqual(snapshot.pop(parameter_name), parameter())

        self.assertEqual(len(snapshot), 1)

        pulse_sequence.add(Pulse(duration=5))

        snapshot = pulse_sequence.snapshot()
        for parameter_name, parameter in pulse_sequence.parameters.items():
            if parameter_name == 'pulses':
                continue
            if parameter.unit:
                parameter_name += f' ({parameter.unit})'
            self.assertEqual(snapshot.pop(parameter_name), parameter())

        for k, pulse_snapshot in enumerate(snapshot['pulses']):
            self.assertEqual(pulse_snapshot, pulse_sequence.pulses[k].snapshot())


class TestCopyPulseSequence(unittest.TestCase):
    def test_copy_empty_pulse_sequence(self):
        pulse_sequence = PulseSequence()
        pulse_sequence_copy = copy(pulse_sequence)

        pulse_sequence.duration = 10
        self.assertNotEqual(pulse_sequence_copy.duration, 10)


class TestPulseSequenceEquality(unittest.TestCase):
    def test_empty_pulse_sequence_equality(self):
        pulse_sequence = PulseSequence()
        self.assertEqual(pulse_sequence, pulse_sequence)

        pulse_sequence2 = PulseSequence()
        self.assertEqual(pulse_sequence, pulse_sequence2)

    def test_non_empty_pulse_sequence_equality(self):
        pulse_sequence = PulseSequence([DCPulse('read', duration=1, amplitude=2)])
        self.assertEqual(pulse_sequence, pulse_sequence)

        pulse_sequence2 = PulseSequence([DCPulse('read', duration=1, amplitude=2)])
        self.assertEqual(pulse_sequence, pulse_sequence2)

        pulse_sequence2['read'].duration = 2
        self.assertNotEqual(pulse_sequence, pulse_sequence2)

        pulse_sequence2['read'].duration = 1
        pulse_sequence2.allow_pulse_overlap = False
        self.assertNotEqual(pulse_sequence, pulse_sequence2)

        pulse_sequence2.allow_pulse_overlap = True
        self.assertEqual(pulse_sequence, pulse_sequence)
        pulse_sequence2.duration = 5
        self.assertNotEqual(pulse_sequence, pulse_sequence2)

    def test_copy_pulse_sequence_equality(self):
        pulse_sequence = PulseSequence()
        pulse_sequence_copy = copy(pulse_sequence)
        self.assertEqual(pulse_sequence, pulse_sequence_copy)

        pulse = DCPulse('read', duration=1, amplitude=2)
        pulse_sequence.add(pulse)
        self.assertNotEqual(pulse_sequence, pulse_sequence_copy)
        pulse_sequence_copy_2 = copy(pulse_sequence)
        self.assertEqual(pulse_sequence, pulse_sequence_copy_2)

        self.assertEqual(pulse_sequence_copy_2, PulseSequence([pulse]))

    def test_pulse_signalling_after_copy(self):
        pulse_sequence = PulseSequence()
        pulse, = pulse_sequence.add(DCPulse('read', duration=1, amplitude=2))
        self.assertEqual(pulse_sequence.enabled_pulses, [pulse])
        self.assertEqual(pulse_sequence.disabled_pulses, [])

        pulse.enabled = False
        self.assertEqual(pulse_sequence.enabled_pulses, [])
        self.assertEqual(pulse_sequence.disabled_pulses, [pulse])

        pulse_sequence_copy = copy(pulse_sequence)
        self.assertEqual(pulse_sequence.enabled_pulses, [])
        self.assertEqual(pulse_sequence.disabled_pulses, [pulse])

        pulse.enabled = True
        self.assertEqual(pulse_sequence.enabled_pulses, [pulse])
        self.assertEqual(pulse_sequence.disabled_pulses, [])


class TestPulseSequenceAddRemove(unittest.TestCase):
    def test_add_remove_pulse(self):
        pulse_sequence = PulseSequence()
        self.assertFalse(pulse_sequence)

        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        self.assertIn(pulse, pulse_sequence)
        self.assertTrue(pulse_sequence)

    def test_initialize_with_pulses(self):
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse2 = DCPulse(name='dc', amplitude=1.5, duration=10)
        pulse_sequence = PulseSequence(pulses=[pulse, pulse2])
        self.assertEqual(len(pulse_sequence), 2)
        self.assertEqual(pulse_sequence.pulses[0], pulse)
        self.assertNotEqual(pulse_sequence.pulses[1], pulse2) #t_start differs
        pulse2.t_start = 10
        self.assertEqual(pulse_sequence.pulses[1], pulse2)

    def test_add_multiple_pulses(self):
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse2 = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=10)
        pulse_sequence = PulseSequence()
        pulse_sequence.add(pulse, pulse2)
        self.assertEqual(len(pulse_sequence), 2)
        self.assertEqual(pulse_sequence.pulses[0], pulse)
        self.assertEqual(pulse_sequence.pulses[1], pulse2)

        pulse3 = DCPulse(name='dc', amplitude=1.5, duration=10)
        pulse3_added, = pulse_sequence.add(pulse3)
        # This one shouldn't be equal since t_stop was not set
        self.assertNotEqual(pulse_sequence.pulses[2], pulse3)
        pulse3.t_start = pulse3_added.t_start
        self.assertEqual(pulse_sequence.pulses[2], pulse3)

    def test_remove_pulse_clear(self):
        # Remove pulses using clear
        pulse_sequence = PulseSequence()
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        pulse_sequence.clear()
        self.assertEqual(len(pulse_sequence.pulses), 0)

    def test_remove_pulse_remove(self):
        # Remove pulses using .remove
        pulse_sequence = PulseSequence()
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        pulse_sequence.remove(pulse)
        self.assertEqual(len(pulse_sequence.pulses), 0)

    def test_remove_reinstantiated_pulse(self):
        # Remove other pulse using .remove
        pulse_sequence = PulseSequence()
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        pulse2 = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.remove(pulse2) # Should work since all attributes match
        self.assertEqual(len(pulse_sequence.pulses), 0)

    def test_remove_wrong_pulse_remove(self):
        # Remove other pulse using .remove
        pulse_sequence = PulseSequence()
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        pulse2 = DCPulse(name='dc', amplitude=2, duration=10, t_start=0)
        with self.assertRaises(AssertionError):
            pulse_sequence.remove(pulse2) # Should not work since different amplitude
        self.assertEqual(len(pulse_sequence.pulses), 1)

    def test_remove_pulse_by_name(self):
        # Remove pulses using .remove
        pulse_sequence = PulseSequence()
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        self.assertEqual(pulse.name, 'dc')
        self.assertEqual(pulse.full_name, 'dc')
        added_pulse, = pulse_sequence.add(pulse)
        self.assertEqual(added_pulse.full_name, 'dc')

        pulse_sequence.remove('dc')
        self.assertEqual(len(pulse_sequence.pulses), 0)

    def test_add_same_name_pulses_sequentially(self):
        pulse_sequence = PulseSequence()
        p = Pulse('DC', duration=5)
        added_pulse, = pulse_sequence.add(p)
        self.assertEqual(added_pulse.id, None)
        self.assertEqual(added_pulse.full_name, 'DC')

        added_pulse2, = pulse_sequence.add(p)
        self.assertEqual(added_pulse.id, 0)
        self.assertEqual(added_pulse.full_name, 'DC[0]')
        self.assertEqual(added_pulse2.id, 1)
        self.assertEqual(added_pulse2.full_name, 'DC[1]')

    def test_remove_wrong_pulse_by_name(self):
        # Remove pulses using .remove
        pulse_sequence = PulseSequence()
        pulse = DCPulse(name='dc', amplitude=1.5, duration=10, t_start=0)
        pulse_sequence.add(pulse)
        with self.assertRaises(AssertionError):
            pulse_sequence.remove('dc2')
        self.assertEqual(len(pulse_sequence.pulses), 1)

    def test_add_disabled_pulse(self):
        pulse_sequence = PulseSequence()
        pulses = []
        for pulse in [Pulse(name='p1', duration=1),
                      Pulse(name='p2', duration=1, enabled=False),
                      Pulse(name='p3', duration=1)]:
            pulses += [pulse_sequence.add(pulse)[0]]
        self.assertEqual(pulse_sequence.enabled_pulses, [pulses[0],
                                                         pulses[2]])
        self.assertEqual(pulse_sequence.disabled_pulses, [pulses[1]])

    def test_final_delay(self):
        original_final_delay = PulseSequence.default_final_delay

        pulse_sequence = PulseSequence()
        self.assertEqual(pulse_sequence.final_delay, original_final_delay)

        PulseSequence.default_final_delay = 1
        pulse_sequence = PulseSequence()
        self.assertEqual(pulse_sequence.final_delay, 1)

        pulse_sequence = PulseSequence(final_delay=2)
        self.assertEqual(pulse_sequence.final_delay, 2)

        PulseSequence.default_final_delay = original_final_delay


class TestPulseSequenceIndirectTstart(unittest.TestCase):
    def test_first_pulse_no_tstart(self):
        p = Pulse(duration=1)
        pulse_sequence = PulseSequence(pulses=[p])
        self.assertEqual(pulse_sequence[0].t_start, 0)

    def test_second_pulse_no_tstart(self):
        p = Pulse(duration=1)
        pulse_sequence = PulseSequence(pulses=[p, p])
        self.assertEqual(pulse_sequence[0].t_start, 0)
        self.assertEqual(pulse_sequence[1].t_start, 1)

    def test_second_pulse_no_tstart_sequential(self):
        p = Pulse(duration=1)
        pulse_sequence = PulseSequence(pulses=[p])
        pulse_sequence.add(p)
        self.assertEqual(pulse_sequence[0].t_start, 0)
        self.assertEqual(pulse_sequence[1].t_start, 1)

    def test_add_pulse_separate_connection(self):
        pulse_sequence = PulseSequence()
        pulse1, = pulse_sequence.add(Pulse(duration=1))
        self.assertEqual(pulse1.t_start, 0)

        pulse2, = pulse_sequence.add(Pulse(duration=2, connection_label='output'))
        self.assertEqual(pulse2.t_start, 0)

        pulse3, = pulse_sequence.add(Pulse(duration=2, connection_label='output'))
        self.assertEqual(pulse3.t_start, 2)

        connection = SingleConnection(output_instrument='ins1',
                                      output_channel=Channel('ins1', 'ch1'),
                                      input_instrument='ins2',
                                      input_channel=Channel('ins2', 'ch1'))
        pulse4, = pulse_sequence.add(Pulse(duration=5, connection=connection))
        self.assertEqual(pulse4.t_start, 0)
        pulse5, = pulse_sequence.add(Pulse(duration=5, connection=connection))
        self.assertEqual(pulse5.t_start, 5)

        output_connection = SingleConnection(output_instrument='ins1',
                                             output_channel=Channel('ins1', 'ch1'),
                                             input_instrument='ins2',
                                             input_channel=Channel('ins2', 'ch1'),
                                             label='output')
        pulse6, = pulse_sequence.add(Pulse(duration=5, connection=output_connection))
        self.assertEqual(pulse6.t_start, 4)


class TestPulseSequenceSignalling(unittest.TestCase):
    def test_change_first_t_stop(self):
        p = Pulse(duration=1)
        pulse_sequence = PulseSequence(pulses=[p, p])
        self.assertEqual(pulse_sequence[1].t_start, 1)

        pulse_sequence[0].duration = 2
        self.assertEqual(pulse_sequence[0].t_stop, 2)
        self.assertEqual(pulse_sequence[1].t_start, 2)

        pulse_sequence[0].t_start = 2
        self.assertEqual(pulse_sequence[0].t_stop, 4)
        self.assertEqual(pulse_sequence[1].t_start, 4)

    def test_change_first_t_stop_three_pulses(self):
        p = Pulse(duration=1)
        pulse_sequence = PulseSequence(pulses=[p, p, p])
        self.assertEqual(pulse_sequence[1].t_start, 1)
        self.assertEqual(pulse_sequence[2].t_start, 2)

        pulse_sequence[0].duration = 2
        self.assertEqual(pulse_sequence[0].t_stop, 2)
        self.assertEqual(pulse_sequence[1].t_start, 2)
        self.assertEqual(pulse_sequence[2].t_start, 3)

        pulse_sequence[0].t_start = 2
        self.assertEqual(pulse_sequence[0].t_stop, 4)
        self.assertEqual(pulse_sequence[1].t_start, 4)
        self.assertEqual(pulse_sequence[2].t_start, 5)

    def test_connected_pulses_t_startoffset(self):
        pulse_sequence = PulseSequence()

        p = Pulse(duration=1)
        pulse1, pulse2 = pulse_sequence.add(p, p)

        # also connect to t_start to measure how often it's called
        registrar = Registrar()
        pulse2['t_start'].connect(registrar)
        self.assertEqual(registrar.values, [1])

        pulse1.t_stop = 2
        self.assertEqual(registrar.values, [1, 2])
        self.assertEqual(pulse2.t_start, 2)

        pulse1['t_stop'].connect(pulse2['t_start'], offset=1)
        self.assertEqual(pulse2.t_start, 3)
        self.assertEqual(registrar.values, [1, 2, 3])

        pulse1.t_stop = 5
        self.assertEqual(pulse2.t_start, 6)
        self.assertEqual(registrar.values, [1, 2, 3, 6])

    def test_change_enabled_pulses(self):
        pulse_sequence = PulseSequence()
        pulses = []
        for pulse in [Pulse(name='p1', duration=1),
                      Pulse(name='p2', duration=1, enabled=False),
                      Pulse(name='p3', duration=1)]:
            pulses += [pulse_sequence.add(pulse)[0]]

        self.assertEqual(pulse_sequence.enabled_pulses, [pulses[0],
                                                         pulses[2]])
        self.assertEqual(pulse_sequence.disabled_pulses, [pulses[1]])

        pulses[0].enabled = False
        self.assertEqual(pulse_sequence.enabled_pulses, [pulses[2]])
        self.assertEqual(pulse_sequence.disabled_pulses, [pulses[0], pulses[1]])

        pulses[0].enabled = True
        self.assertEqual(pulse_sequence.enabled_pulses, [pulses[0],
                                                         pulses[2]])
        self.assertEqual(pulse_sequence.disabled_pulses, [pulses[1]])


if __name__ == '__main__':
    unittest.main()