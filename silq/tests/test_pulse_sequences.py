import unittest
from copy import copy, deepcopy
import pickle
import numpy as np
import random

import silq
from silq import DictConfig
from silq.pulses import PulseSequence, DCPulse, TriggerPulse, Pulse
from silq.instrument_interfaces import Channel
from silq.meta_instruments.layout import SingleConnection
import qcodes as qc

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
        with self.assertRaises(SyntaxError):
            PulseSequence([p])

    def test_get_pulses(self):
        pulse_sequence = PulseSequence()
        self.assertListEqual(pulse_sequence.get_pulses(), [])
        pulse1 = DCPulse(name='dc1', amplitude=1.5, duration=10, t_start=1)
        pulse2 = DCPulse(name='dc2', amplitude=2.5, duration=10, t_start=1)
        pulse3 = TriggerPulse(name='trig', duration=12, t_start=1)
        pulse_sequence.add(pulse1, pulse2, pulse3)

        subset_pulses = pulse_sequence.get_pulses()
        self.assertEqual(subset_pulses[0], pulse1)
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
            if parameter_name in ['enabled_pulses', 'pulses']:
                continue
            self.assertEqual(snapshot.pop(parameter_name), parameter(), msg=parameter_name)

        self.assertEqual(len(snapshot), 2)

        pulse_sequence.add(Pulse(duration=5))

        snapshot = pulse_sequence.snapshot()
        for parameter_name, parameter in pulse_sequence.parameters.items():
            if parameter_name in ['pulses', 'enabled_pulses']:
                continue
            if parameter.unit:
                parameter_name += f' ({parameter.unit})'
            self.assertEqual(snapshot.pop(parameter_name), parameter(), msg=parameter_name)

        for k, pulse_snapshot in enumerate(snapshot['pulses']):
            self.assertEqual(pulse_snapshot, pulse_sequence.pulses[k].snapshot(), msg=repr(pulse_sequence.pulses[k]))

    def test_pulse_overlap(self):
        pulse_sequence = PulseSequence(allow_pulse_overlap=False)
        pulse1 = DCPulse(t_start=0, duration=10e-3)
        pulse2 = DCPulse(t_start=5e-3, duration=10e-3)
        with self.assertRaises(AssertionError):
            pulse_sequence.add(pulse1, pulse2)

    def test_pulse_overlap_no_t_start(self):
        pulse_sequence = PulseSequence(allow_pulse_overlap=False)
        pulse1 = DCPulse(duration=10e-3)
        pulse_sequence.add(pulse1, pulse1)

    def test_pulse_overlap_no_t_start_sequential(self):
        pulse_sequence = PulseSequence(allow_pulse_overlap=False)
        pulse1 = DCPulse(duration=10e-3)
        pulse_sequence.add(pulse1)
        pulse_sequence.add(pulse1)

    def test_pulse_sequence_times(self):
        pulse_sequence = PulseSequence()
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 0)
        self.assertEqual(pulse_sequence.t_stop, 0)

        DC_pulse, = pulse_sequence.add(DCPulse('DC', duration=1))
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 1)
        self.assertEqual(pulse_sequence.t_stop, 1)

        pulse_sequence.t_start = 2
        self.assertEqual(pulse_sequence.t_start, 2)
        self.assertEqual(pulse_sequence.duration, 1)
        self.assertEqual(pulse_sequence.t_stop, 3)
        self.assertEqual(DC_pulse.t_start, 2)
        self.assertEqual(DC_pulse.duration, 1)
        self.assertEqual(DC_pulse.t_stop, 3)

    def test_last_pulse(self):
        pulse_sequence = PulseSequence()
        read_pulse, plunge_pulse = pulse_sequence.add(
            DCPulse('read', duration=1),
            DCPulse('plunge', duration=1)
        )
        self.assertEqual(pulse_sequence._last_pulse, plunge_pulse)


class TestPulseSequenceQuickAdd(unittest.TestCase):
    def test_quick_add_pulses(self):
        pulses = [DCPulse(duration=10),
                  DCPulse(duration=10)]
        pulse_sequence = PulseSequence()
        added_pulses = pulse_sequence.quick_add(*pulses)

        self.assertNotEqual(pulses, added_pulses)
        self.assertEqual(pulses[0].t_start, None)
        self.assertEqual(added_pulses[0].t_start, 0)
        self.assertEqual(added_pulses[1].t_start, 10)

        self.assertEqual(pulse_sequence.duration, 20)

    def test_quick_add_pulse_id(self):
        pulses = [DCPulse('DC', duration=10),
                  DCPulse('DC', duration=10),
                  DCPulse('DC2', duration=10)]
        pulse_sequence = PulseSequence()
        added_pulses = pulse_sequence.quick_add(*pulses)
        pulse_sequence.finish_quick_add()

        self.assertNotEqual(pulses, added_pulses)
        self.assertEqual(pulses[0].t_start, None)
        self.assertEqual(added_pulses[0].t_start, 0)
        self.assertEqual(added_pulses[1].t_start, 10)
        self.assertEqual(added_pulses[2].t_start, 20)

        self.assertEqual(pulses[0].full_name, 'DC')
        self.assertEqual(pulses[1].full_name, 'DC')
        self.assertEqual(pulses[2].full_name, 'DC2')

        self.assertEqual(added_pulses[0].full_name, 'DC[0]')
        self.assertEqual(added_pulses[1].full_name, 'DC[1]')
        self.assertEqual(added_pulses[2].full_name, 'DC2')

        self.assertEqual(pulse_sequence.duration, 30)

    def test_quick_add_unsorted_pulses(self):
        pulses = []

        t = 0
        for k in range(30):
            duration = np.round(np.random.rand(), 11)
            pulses.append(DCPulse('DC', t_start=t, duration=duration))
            t += duration
        random.shuffle(pulses)

        pulse_sequence = PulseSequence()
        added_pulses = pulse_sequence.quick_add(*pulses)

        for pulse in added_pulses:
            self.assertEqual(pulse.id, None)

        pulse_sequence.finish_quick_add()

        t = 0
        for k, pulse in enumerate(pulse_sequence.pulses):
            self.assertEqual(pulse.id, k)
            self.assertAlmostEqual(pulse.t_start, t)
            t += pulse.duration

        self.assertAlmostEqual(pulse_sequence.duration, t)

    def test_overlapping_pulses(self):
        pulses = [DCPulse(t_start=0, duration=10),
                  DCPulse(t_start=5, duration=10)]
        pulse_sequence = PulseSequence(allow_pulse_overlap=False)
        pulse_sequence.quick_add(*pulses)

        with self.assertRaises(AssertionError):
            pulse_sequence.finish_quick_add()

    def test_overlapping_pulses_different_connection_label(self):
        pulses = [DCPulse(t_start=0, duration=10, connection_label='con1'),
                  DCPulse(t_start=5, duration=10, connection_label='con2')]
        pulse_sequence = PulseSequence(allow_pulse_overlap=False)
        pulse_sequence.quick_add(*pulses)
        pulse_sequence.finish_quick_add()

    def test_overlapping_random_pulses(self):
        for connection_label in ['connection1', 'connection2', 'connection3', None]:
            pulses = []
            t = 0
            for k in range(30):
                duration = np.round(np.random.rand(), 11)
                pulses.append(DCPulse('DC', t_start=t, duration=duration,
                                      connection_label='connection1'))
                t += duration
            random.shuffle(pulses)

            pulse_sequence = PulseSequence(allow_pulse_overlap=False)
            pulse_sequence.quick_add(*pulses)
            pulse_sequence.finish_quick_add()  # No overlap

            # Add pulses with connection label
            second_pulses = []
            t = 0
            for k in range(30):
                duration = np.round(np.random.rand(), 11)
                second_pulses.append(DCPulse('DC', t_start=t, duration=duration,
                                             connection_label='connection2'))
                t += duration
            random.shuffle(second_pulses)

            pulse_sequence.quick_add(*second_pulses)
            pulse_sequence.finish_quick_add()  # No overlap

            # Add pulse that potentially overlaps based on connection_label
            overlapping_pulse = DCPulse(t_start=pulse_sequence.duration / 2, duration=1e-5,
                                        connection_label=connection_label)
            overlapping_pulse_copy, = pulse_sequence.quick_add(overlapping_pulse)

            if connection_label in ['connection1', 'connection2', None]:
                with self.assertRaises(AssertionError):
                    pulse_sequence.finish_quick_add()
            else:
                pulse_sequence.finish_quick_add()
                pulse_sequence.remove(overlapping_pulse_copy)
                pulse_sequence.finish_quick_add()

    def test_pulse_sequence_times(self):
        pulses = [DCPulse(t_start=0, duration=10, connection_label='con1'),
                  DCPulse(t_start=5, duration=10, connection_label='con2')]
        pulse_sequence = PulseSequence()
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 0)
        self.assertEqual(pulse_sequence.t_stop, 0)

        pulse_sequence.quick_add(*pulses)
        pulse_sequence.finish_quick_add()
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 15)
        self.assertEqual(pulse_sequence.t_stop, 15)

        pulse_sequence.t_start = 2
        self.assertEqual(pulse_sequence.t_start, 2)
        self.assertEqual(pulse_sequence.duration, 15)
        self.assertEqual(pulse_sequence.t_stop, 17)
        self.assertEqual(pulse_sequence.pulses[0].t_start, 2)
        self.assertEqual(pulse_sequence.pulses[0].duration, 10)
        self.assertEqual(pulse_sequence.pulses[0].t_stop, 12)
        self.assertEqual(pulse_sequence.pulses[1].t_start, 7)
        self.assertEqual(pulse_sequence.pulses[1].duration, 10)
        self.assertEqual(pulse_sequence.pulses[1].t_stop, 17)


class TestCopyPulseSequence(unittest.TestCase):
    def test_copy_empty_pulse_sequence(self):
        pulse_sequence = PulseSequence()
        pulse_sequence_copy = copy(pulse_sequence)

        pulse_sequence.duration = 10
        self.assertNotEqual(pulse_sequence_copy.duration, 10)

    def test_copy_filled_pulse_sequence(self):
        pulse_sequence = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ])

        pulse_sequence_copy = copy(pulse_sequence)
        self.assertEqual(pulse_sequence_copy.duration, 3)

        self.assertTupleEqual(pulse_sequence_copy.pulses, pulse_sequence_copy.enabled_pulses)


class TestCopyCountPulseSequence(unittest.TestCase):
    def reset_executions(self):
        self.executions = {
            'Pulse': {'__copy__': 0, '__deepcopy__': 0},
            'PulseSequence': {'__copy__': 0, '__deepcopy__': 0}
        }

    def setUp(self) -> None:
        self.reset_executions()
        # wrap pulse.copy
        self.pulse_copy = Pulse.__copy__
        def copy_fun_wrapped(*args, **kwargs):
            self.executions[Pulse.__name__]['__copy__'] += 1
            return self.pulse_copy(*args, **kwargs)
        Pulse.__copy__ = copy_fun_wrapped

        # wrap pulse.copy
        self.pulse_sequence_copy = PulseSequence.__copy__
        def copy_fun_wrapped(*args, **kwargs):
            self.executions[PulseSequence.__name__]['__copy__'] += 1
            return self.pulse_sequence_copy(*args, **kwargs)
        PulseSequence.__copy__ = copy_fun_wrapped

        # wrap __deepcopy
        from qcodes.instrument import parameter_node
        self.deepcopy = __deepcopy__ = parameter_node.__deepcopy__

        def __deepcopy_wrapped(other_self, *args, **kwargs):
            if isinstance(other_self, (Pulse, PulseSequence)):
                class_name = 'Pulse' if isinstance(other_self, Pulse) else 'PulseSequence'
                self.executions[class_name]['__deepcopy__'] += 1
            return __deepcopy__(other_self, *args, **kwargs)
        parameter_node.__deepcopy__ = __deepcopy_wrapped

    def tearDown(self):
        Pulse.__copy__ = self.pulse_copy
        PulseSequence.__copy__ = self.pulse_sequence_copy

        from qcodes.instrument import parameter_node
        parameter_node.__deepcopy__ = self.deepcopy


    def test_copy_empty_pulse_sequence(self):
        pulse_sequence = PulseSequence()
        copy(pulse_sequence)
        self.assertEqual(self.executions['Pulse']['__copy__'], 0)
        self.assertEqual(self.executions['Pulse']['__deepcopy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__copy__'], 1)
        self.assertEqual(self.executions['PulseSequence']['__deepcopy__'], 0)

    def test_deepcopy_empty_pulse_sequence(self):
        pulse_sequence = PulseSequence()
        deepcopy(pulse_sequence)
        self.assertEqual(self.executions['Pulse']['__copy__'], 0)
        self.assertEqual(self.executions['Pulse']['__deepcopy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__copy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__deepcopy__'], 1)

    def test_copy_pulse_sequence_one_pulse(self):
        pulse_sequence = PulseSequence()
        pulse_sequence.add(DCPulse(duration=2, amplitude=1), copy=False)

        copy(pulse_sequence)
        self.assertEqual(self.executions['Pulse']['__copy__'], 1)
        self.assertEqual(self.executions['Pulse']['__deepcopy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__copy__'], 1)
        self.assertEqual(self.executions['PulseSequence']['__deepcopy__'], 0)

    def test_deepcopy_pulse_sequence_one_pulse(self):
        pulse_sequence = PulseSequence()
        pulse_sequence.add(DCPulse(duration=2, amplitude=1), copy=False)

        deepcopy(pulse_sequence)
        self.assertEqual(self.executions['Pulse']['__copy__'], 0)
        # One extra for _last_pulse. This really shouldn't happen though, but no big deal
        self.assertEqual(self.executions['Pulse']['__deepcopy__'], 2)
        self.assertEqual(self.executions['PulseSequence']['__copy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__deepcopy__'], 1)

    def test_copy_pulse_sequence_two_pulses(self):
        pulse_sequence = PulseSequence()
        pulse_sequence.add(DCPulse(duration=2, amplitude=1), copy=False)
        pulse_sequence.add(DCPulse(duration=2, amplitude=1), copy=False)

        copy(pulse_sequence)
        self.assertEqual(self.executions['Pulse']['__copy__'], 2)
        self.assertEqual(self.executions['Pulse']['__deepcopy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__copy__'], 1)
        self.assertEqual(self.executions['PulseSequence']['__deepcopy__'], 0)

    def test_deepcopy_pulse_sequence_two_pulses(self):
        pulse_sequence = PulseSequence()
        pulse_sequence.add(DCPulse(duration=2, amplitude=1), copy=False)
        pulse_sequence.add(DCPulse(duration=2, amplitude=1), copy=False)

        deepcopy(pulse_sequence)
        self.assertEqual(self.executions['Pulse']['__copy__'], 0)
        # One extra for _last_pulse. This really shouldn't happen though, but no big deal
        self.assertEqual(self.executions['Pulse']['__deepcopy__'], 3)
        self.assertEqual(self.executions['PulseSequence']['__copy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__deepcopy__'], 1)

    def test_copy_nested_pulse_sequence(self):
        pulse_sequence = PulseSequence()
        nested_pulse_sequence = PulseSequence()
        nested_pulse_sequence.add(DCPulse(duration=2, amplitude=1), copy=False)
        pulse_sequence.add_pulse_sequences(nested_pulse_sequence)

        copy(pulse_sequence)
        self.assertEqual(self.executions['Pulse']['__copy__'], 1)
        self.assertEqual(self.executions['Pulse']['__deepcopy__'], 0)
        self.assertEqual(self.executions['PulseSequence']['__copy__'], 2)
        self.assertEqual(self.executions['PulseSequence']['__deepcopy__'], 0)


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
        self.assertEqual(pulse_sequence.enabled_pulses, (pulse, ))
        self.assertEqual(pulse_sequence.disabled_pulses, ())

        pulse.enabled = False
        self.assertEqual(pulse_sequence.enabled_pulses, ())
        self.assertEqual(pulse_sequence.disabled_pulses, (pulse, ))

        pulse_sequence_copy = copy(pulse_sequence)
        self.assertEqual(pulse_sequence.enabled_pulses, ())
        self.assertEqual(pulse_sequence.disabled_pulses, (pulse, ))

        pulse.enabled = True
        self.assertEqual(pulse_sequence.enabled_pulses, (pulse, ))
        self.assertEqual(pulse_sequence.disabled_pulses, ())


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
        self.assertEqual(pulse_sequence.enabled_pulses,
                         (pulses[0], pulses[2]))
        self.assertEqual(pulse_sequence.disabled_pulses, (pulses[1], ))

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

        self.assertTupleEqual(pulse_sequence.enabled_pulses,
                              (pulses[0], pulses[2]))
        self.assertEqual(pulse_sequence.disabled_pulses, (pulses[1],))

        pulses[0].enabled = False
        self.assertEqual(pulse_sequence.enabled_pulses, (pulses[2], ))
        self.assertEqual(pulse_sequence.disabled_pulses, (pulses[0], pulses[1]))

        pulses[0].enabled = True
        self.assertEqual(pulse_sequence.enabled_pulses,
                         (pulses[0], pulses[2]))
        self.assertEqual(pulse_sequence.disabled_pulses, (pulses[1],))


class TestPulseSequencePickling(unittest.TestCase):
    def test_pickle_empty_pulse_sequence(self):
        pulse_sequence = PulseSequence()
        pickle_dump = pickle.dumps(pulse_sequence)
        pickled_pulse_sequence = pickle.loads(pickle_dump)

    def test_pickle_pulse_sequence_single_pulse(self):
        p = DCPulse('pulse', duration=2, amplitude=3)
        pulse_sequence = PulseSequence(pulses=[p])
        self.assertEqual(pulse_sequence.pulses[0].name, 'pulse')
        self.assertEqual(pulse_sequence.pulses[0].t_start, 0)
        self.assertEqual(pulse_sequence.pulses[0].duration, 2)
        self.assertEqual(pulse_sequence.pulses[0].amplitude, 3)
        self.assertEqual(pulse_sequence.duration, 2)

        pickle_dump = pickle.dumps(pulse_sequence)
        pickled_pulse_sequence = pickle.loads(pickle_dump)
        self.assertEqual(pickled_pulse_sequence.pulses[0].name, 'pulse')
        self.assertEqual(pickled_pulse_sequence.pulses[0].t_start, 0)
        self.assertEqual(pickled_pulse_sequence.pulses[0].duration, 2)
        self.assertEqual(pickled_pulse_sequence.pulses[0].amplitude, 3)
        self.assertEqual(pickled_pulse_sequence.duration, 2)

    def test_pickle_pulse_sequence_two_pulses(self):
        p = DCPulse('pulse', duration=2, amplitude=3)
        pulse_sequence = PulseSequence(pulses=[p, p])
        self.assertEqual(pulse_sequence.pulses[0].full_name, 'pulse[0]')
        self.assertEqual(pulse_sequence.pulses[1].full_name, 'pulse[1]')
        self.assertEqual(pulse_sequence.pulses[0].t_start, 0)
        self.assertEqual(pulse_sequence.pulses[0].duration, 2)
        self.assertEqual(pulse_sequence.pulses[0].amplitude, 3)
        self.assertEqual(pulse_sequence.duration, 4)

        pickle_dump = pickle.dumps(pulse_sequence)
        pickled_pulse_sequence = pickle.loads(pickle_dump)
        self.assertEqual(pickled_pulse_sequence.pulses[0].full_name, 'pulse[0]')
        self.assertEqual(pickled_pulse_sequence.pulses[0].t_start, 0)
        self.assertEqual(pickled_pulse_sequence.pulses[0].duration, 2)
        self.assertEqual(pickled_pulse_sequence.pulses[0].amplitude, 3)

        self.assertEqual(pickled_pulse_sequence.pulses[1].full_name, 'pulse[1]')
        self.assertEqual(pickled_pulse_sequence.pulses[1].t_start, 2)
        self.assertEqual(pickled_pulse_sequence.pulses[1].duration, 2)
        self.assertEqual(pickled_pulse_sequence.pulses[1].amplitude, 3)

        self.assertEqual(pickled_pulse_sequence.duration, 4)


class TestCompositePulseSequences(unittest.TestCase):
    def test_basic_composite_pulse_sequence(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ])
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ])

        pulse_sequence = PulseSequence(pulse_sequences=[pulse_sequence1, pulse_sequence2])
        self.assertEqual(pulse_sequence1.t_start, 0)
        self.assertEqual(pulse_sequence2.t_start, 3)

        self.assertEqual(pulse_sequence1[0].t_start, 0)
        self.assertEqual(pulse_sequence1[1].t_start, 1)
        self.assertEqual(pulse_sequence2[0].t_start, 3)
        self.assertEqual(pulse_sequence2[1].t_start, 4)

        self.assertTupleEqual(
            pulse_sequence.pulses,
            (*pulse_sequence1.pulses, *pulse_sequence2.pulses)
        )
        self.assertListEqual(list(pulse_sequence), [*pulse_sequence1, *pulse_sequence2])

    def test_named_composite_pulse_sequence(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ], name='ESR1')
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ], name='ESR2')

        pulse_sequence = PulseSequence(pulse_sequences=[pulse_sequence1, pulse_sequence2],
                                       name='ESR3')
        self.assertEqual(pulse_sequence1.t_start, 0)
        self.assertEqual(pulse_sequence2.t_start, 3)

        self.assertEqual(pulse_sequence1[0].t_start, 0)
        self.assertEqual(pulse_sequence1[1].t_start, 1)
        self.assertEqual(pulse_sequence2[0].t_start, 3)
        self.assertEqual(pulse_sequence2[1].t_start, 4)

        self.assertTupleEqual(
            pulse_sequence.pulses,
            (*pulse_sequence1.pulses, *pulse_sequence2.pulses)
        )
        self.assertListEqual(list(pulse_sequence), [*pulse_sequence1, *pulse_sequence2])

        self.assertEqual(pulse_sequence[0].full_name, 'ESR1.read')
        self.assertEqual(pulse_sequence[1].full_name, 'ESR1.read2')
        self.assertEqual(pulse_sequence[2].full_name, 'ESR2.read3')
        self.assertEqual(pulse_sequence[3].full_name, 'ESR2.read4')

        self.assertEqual(pulse_sequence[0], pulse_sequence['ESR1.read'])
        self.assertEqual(pulse_sequence[1], pulse_sequence['ESR1.read2'])
        self.assertEqual(pulse_sequence[2], pulse_sequence['ESR2.read3'])
        self.assertEqual(pulse_sequence[3], pulse_sequence['ESR2.read4'])

    def test_composite_pulse_sequence_differing_duration(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ])
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ])

        pulse_sequence = PulseSequence(
            pulse_sequences=[pulse_sequence1, pulse_sequence2])
        self.assertEqual(pulse_sequence1.t_start, 0)
        self.assertEqual(pulse_sequence2.t_start, 3)

        pulse_sequence['read'].duration = 3
        self.assertEqual(pulse_sequence1.t_start, 0)
        self.assertEqual(pulse_sequence2.t_start, 5)
        self.assertEqual(pulse_sequence1[0].t_start, 0)
        self.assertEqual(pulse_sequence1[1].t_start, 3)
        self.assertEqual(pulse_sequence2[0].t_start, 5)
        self.assertEqual(pulse_sequence2[1].t_start, 6)

    def test_copy_basic_composite_pulse_sequence(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ])
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ])

        pulse_sequence = PulseSequence(pulse_sequences=[pulse_sequence1, pulse_sequence2])

        pulse_sequence = copy(pulse_sequence)
        pulse_sequence1, pulse_sequence2 = pulse_sequence.pulse_sequences

        self.assertEqual(pulse_sequence1.t_start, 0)
        self.assertEqual(pulse_sequence2.t_start, 3)

        self.assertEqual(pulse_sequence1[0].t_start, 0)
        self.assertEqual(pulse_sequence1[1].t_start, 1)
        self.assertEqual(pulse_sequence2[0].t_start, 3)
        self.assertEqual(pulse_sequence2[1].t_start, 4)

        self.assertTupleEqual(
            pulse_sequence.pulses,
            (*pulse_sequence1.pulses, *pulse_sequence2.pulses)
        )
        self.assertListEqual(list(pulse_sequence), [*pulse_sequence1, *pulse_sequence2])

    def test_copy_pulse_in_composite_sequence(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ])
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ])

        pulse_sequence = PulseSequence(pulse_sequences=[pulse_sequence1, pulse_sequence2])

        pulse_sequence = copy(pulse_sequence)

        pulse = copy(pulse_sequence[2])
        self.assertEqual(pulse.t_start, 3)

    def test_add_copied_pulse_in_composite_sequence(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ])
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ])

        pulse_sequence = PulseSequence(pulse_sequences=[pulse_sequence1, pulse_sequence2])

        pulse_sequence = copy(pulse_sequence)

        pulse = copy(pulse_sequence[2])
        self.assertEqual(pulse.t_start, 3)

        new_pulse_sequence = PulseSequence()
        new_pulse, = new_pulse_sequence.add(pulse)
        self.assertEqual(new_pulse.t_start, 0)

    def test_add_nested_pulse_to_skeleton(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ], name='nested1')
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ], name='nested2')

        pulse_sequence = PulseSequence(
            name='main', pulse_sequences=[pulse_sequence1, pulse_sequence2]
        )
        pulse = pulse_sequence[2]

        skeleton_pulse_sequence = PulseSequence()
        skeleton_pulse_sequence.clone_skeleton(pulse_sequence)
        self.assertEqual(skeleton_pulse_sequence.name, 'main')
        skeleton_pulse_sequence.add(pulse)
        self.assertEqual(skeleton_pulse_sequence[0].name, pulse.name)
        skeleton_pulse_sequence.add(pulse, nest=True)
        self.assertEqual(skeleton_pulse_sequence.pulse_sequences[1][0].name, pulse.name)

        pulse_sequence = copy(pulse_sequence)
        pulse = copy(pulse)

        skeleton_pulse_sequence = PulseSequence()
        skeleton_pulse_sequence.clone_skeleton(pulse_sequence)
        self.assertEqual(skeleton_pulse_sequence.name, 'main')
        skeleton_pulse_sequence.add(pulse)
        self.assertEqual(skeleton_pulse_sequence[0].name, pulse.name)
        skeleton_pulse_sequence.add(pulse, nest=True)
        self.assertEqual(skeleton_pulse_sequence.pulse_sequences[1][0].name, pulse.name)

    def test_disabling_first_pulse_sequence(self):
        pulse_sequence1 = PulseSequence([
            DCPulse('read', duration=1),
            DCPulse('read2', duration=2)
        ], name='pulse_sequence1')
        pulse_sequence2 = PulseSequence([
            DCPulse('read3', duration=1),
            DCPulse('read4', duration=2)
        ], name='pulse_sequence2')

        pulse_sequence = PulseSequence(pulse_sequences=[pulse_sequence1, pulse_sequence2])

        self.assertEqual(pulse_sequence2.t_start, 3)

        pulse_sequence1.enabled = False

        self.assertEqual(pulse_sequence2.t_start, 0)


class TestPulseSequenceGenerators(unittest.TestCase):
    def setUp(self):
        self.silq_environment = silq.environment
        self.silq_config = silq.config

        self.d = {
            'pulses': {
                'empty': {'duration': 1, 'amplitude': -1},
                'plunge': {'duration': 0.5, 'amplitude': 1},
                'read_long': {'duration': 5, 'amplitude': 0},
                'read_initialize': {'duration': 3, 'amplitude': 0},
                'ESR': {'duration': 0.1, 'power': 0},
            },
            'properties': {},
        }
        self.config = DictConfig('cfg', config=self.d)
        qc.config.user.silq_config = silq.config = self.config

    def tearDown(self):
        silq.environment = self.silq_environment
        qc.config.user.silq_config = silq.config = self.silq_config

    def test_ESR_pulse_sequence(self):
        from silq.pulses.pulse_sequences import ESRPulseSequence
        from silq.pulses import SinePulse

        pulse_sequence = ESRPulseSequence()
        pulse_sequence.ESR['ESR_pulse'] = SinePulse('ESR', duration=1, frequency=38e9)
        pulse_sequence.ESR['stage_pulse'].duration = 2
        pulse_sequence.ESR['read_pulse'].duration = 2
        pulse_sequence.EPR['enabled'] = False
        pulse_sequence.post_pulses = []

        pulse_sequence.generate()

        # Copy the pulse sequence
        copy(pulse_sequence)

    def test_ESR_pulse_sequence_composite(self):
        from silq.pulses.pulse_sequences import ESRPulseSequenceComposite
        pulse_sequence = ESRPulseSequenceComposite()

        pulse_sequence.ESR.pulse_settings['pre_delay'] = 0.2
        pulse_sequence.ESR.pulse_settings['post_delay'] = 0.1

        pulse_sequence.generate()

        self.assertEqual(pulse_sequence.ESR.t_start, 0)
        self.assertEqual(pulse_sequence.ESR.duration, 3.4)
        self.assertEqual(pulse_sequence.ESR.t_stop, 3.4)
        self.assertEqual(pulse_sequence['ESR.plunge'].t_start, 0)
        self.assertEqual(pulse_sequence['ESR.plunge'].duration, 0.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].t_start, 0.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].duration, 3)
        self.assertEqual(pulse_sequence.EPR.t_start, 3.4)
        self.assertEqual(pulse_sequence.EPR.duration, 6.5)
        self.assertEqual(pulse_sequence.EPR.t_stop, 9.9)
        self.assertEqual(pulse_sequence['EPR.empty'].t_start, 3.4)
        self.assertEqual(pulse_sequence['EPR.empty'].duration, 1)
        self.assertEqual(pulse_sequence['EPR.plunge'].t_start, 4.4)
        self.assertEqual(pulse_sequence['EPR.plunge'].duration, 0.5)
        self.assertEqual(pulse_sequence['EPR.read_long'].t_start, 4.9)
        self.assertEqual(pulse_sequence['EPR.read_long'].duration, 5)
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 9.9)
        self.assertEqual(pulse_sequence.t_stop, 9.9)

    def test_ESR_pulse_sequence_composite_modified(self):
        from silq.pulses.pulse_sequences import ESRPulseSequenceComposite
        pulse_sequence = ESRPulseSequenceComposite()

        pulse_sequence.ESR.pulse_settings['pre_delay'] = 0.2
        pulse_sequence.ESR.pulse_settings['post_delay'] = 0.1

        pulse_sequence.generate()

        pulse_sequence['ESR.plunge'].duration = 1.4

        self.assertEqual(pulse_sequence.ESR.t_start, 0)
        self.assertEqual(pulse_sequence.ESR.duration, 4.4)
        self.assertEqual(pulse_sequence.ESR.t_stop, 4.4)
        self.assertEqual(pulse_sequence['ESR.plunge'].t_start, 0)
        self.assertEqual(pulse_sequence['ESR.plunge'].duration, 1.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].t_start, 1.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].duration, 3)
        self.assertEqual(pulse_sequence.EPR.t_start, 4.4)
        self.assertEqual(pulse_sequence.EPR.duration, 6.5)
        self.assertEqual(pulse_sequence.EPR.t_stop, 10.9)
        self.assertEqual(pulse_sequence['EPR.empty'].t_start, 4.4)
        self.assertEqual(pulse_sequence['EPR.empty'].duration, 1)
        self.assertEqual(pulse_sequence['EPR.plunge'].t_start, 5.4)
        self.assertEqual(pulse_sequence['EPR.plunge'].duration, 0.5)
        self.assertEqual(pulse_sequence['EPR.read_long'].t_start, 5.9)
        self.assertEqual(pulse_sequence['EPR.read_long'].duration, 5)
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 10.9)
        self.assertEqual(pulse_sequence.t_stop, 10.9)

    def test_ESR_pulse_sequence_composite_copied(self):
        from silq.pulses.pulse_sequences import ESRPulseSequenceComposite
        pulse_sequence = ESRPulseSequenceComposite()

        pulse_sequence.ESR.pulse_settings['pre_delay'] = 0.2
        pulse_sequence.ESR.pulse_settings['post_delay'] = 0.1

        pulse_sequence.generate()

        pulse_sequence = copy(pulse_sequence)

        self.assertTrue(hasattr(pulse_sequence, 'EPR'))
        self.assertTrue(hasattr(pulse_sequence, 'ESR'))

        self.assertEqual(pulse_sequence.ESR.t_start, 0)
        self.assertEqual(pulse_sequence.ESR.duration, 3.4)
        self.assertEqual(pulse_sequence.ESR.t_stop, 3.4)
        self.assertEqual(pulse_sequence['ESR.plunge'].t_start, 0)
        self.assertEqual(pulse_sequence['ESR.plunge'].duration, 0.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].t_start, 0.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].duration, 3)
        self.assertEqual(pulse_sequence.EPR.t_start, 3.4)
        self.assertEqual(pulse_sequence.EPR.duration, 6.5)
        self.assertEqual(pulse_sequence.EPR.t_stop, 9.9)
        self.assertEqual(pulse_sequence['EPR.empty'].t_start, 3.4)
        self.assertEqual(pulse_sequence['EPR.empty'].duration, 1)
        self.assertEqual(pulse_sequence['EPR.plunge'].t_start, 4.4)
        self.assertEqual(pulse_sequence['EPR.plunge'].duration, 0.5)
        self.assertEqual(pulse_sequence['EPR.read_long'].t_start, 4.9)
        self.assertEqual(pulse_sequence['EPR.read_long'].duration, 5)
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 9.9)
        self.assertEqual(pulse_sequence.t_stop, 9.9)

    def test_ESR_pulse_sequence_composite_modified_copied(self):
        from silq.pulses.pulse_sequences import ESRPulseSequenceComposite
        pulse_sequence = ESRPulseSequenceComposite()

        pulse_sequence.ESR.pulse_settings['pre_delay'] = 0.2
        pulse_sequence.ESR.pulse_settings['post_delay'] = 0.1

        pulse_sequence.generate()

        pulse_sequence['ESR.plunge'].duration = 1.4

        pulse_sequence = copy(pulse_sequence)

        self.assertEqual(pulse_sequence.ESR.t_start, 0)
        self.assertEqual(pulse_sequence.ESR.duration, 4.4)
        self.assertEqual(pulse_sequence.ESR.t_stop, 4.4)
        self.assertEqual(pulse_sequence['ESR.plunge'].t_start, 0)
        self.assertEqual(pulse_sequence['ESR.plunge'].duration, 1.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].t_start, 1.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].duration, 3)
        self.assertEqual(pulse_sequence.EPR.t_start, 4.4)
        self.assertEqual(pulse_sequence.EPR.duration, 6.5)
        self.assertEqual(pulse_sequence.EPR.t_stop, 10.9)
        self.assertEqual(pulse_sequence['EPR.empty'].t_start, 4.4)
        self.assertEqual(pulse_sequence['EPR.empty'].duration, 1)
        self.assertEqual(pulse_sequence['EPR.plunge'].t_start, 5.4)
        self.assertEqual(pulse_sequence['EPR.plunge'].duration, 0.5)
        self.assertEqual(pulse_sequence['EPR.read_long'].t_start, 5.9)
        self.assertEqual(pulse_sequence['EPR.read_long'].duration, 5)
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 10.9)
        self.assertEqual(pulse_sequence.t_stop, 10.9)

    def test_ESR_pulse_sequence_composite_disable_EPR(self):
        from silq.pulses.pulse_sequences import ESRPulseSequenceComposite
        pulse_sequence = ESRPulseSequenceComposite()

        pulse_sequence.ESR.pulse_settings['pre_delay'] = 0.2
        pulse_sequence.ESR.pulse_settings['post_delay'] = 0.1

        pulse_sequence.EPR.enabled = False

        pulse_sequence.generate()

        self.assertEqual(pulse_sequence.ESR.t_start, 0)
        self.assertEqual(pulse_sequence.ESR.duration, 3.4)
        self.assertEqual(pulse_sequence.ESR.t_stop, 3.4)

        self.assertEqual(pulse_sequence['ESR.plunge'].t_start, 0)
        self.assertEqual(pulse_sequence['ESR.plunge'].duration, 0.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].t_start, 0.4)
        self.assertEqual(pulse_sequence['ESR.read_initialize'].duration, 3)

        self.assertEqual(pulse_sequence.EPR.enabled, False)

        self.assertListEqual([p for p in pulse_sequence],
                             [p for p in pulse_sequence.ESR])
        self.assertEqual(pulse_sequence.t_start, 0)
        self.assertEqual(pulse_sequence.duration, 3.4)
        self.assertEqual(pulse_sequence.t_stop, 3.4)

if __name__ == '__main__':
    unittest.main()
