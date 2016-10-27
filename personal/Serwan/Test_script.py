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
    class TestLayout(unittest.TestCase):
        def setUp(self):
            silent = True
            self.arbstudio = MockArbStudio(name='arbstudio', silent=silent,
                                           server_name='')
            self.pulseblaster = MockPulseBlaster(name='pulseblaster',
                                                 silent=silent, server_name='')
            self.chip = Chip(name='chip', server_name='')

            self.ATS = MockATS(name='ATS', server_name='ATS_server',
                               silent=silent)
            self.ATS.config(sample_rate=100000)
            self.acquisition_controller = MockAcquisitionController(
                name='basic_acquisition_controller',
                alazar_name='ATS',
                server_name='ATS_server',
                silent=silent)

            self.instruments = [self.arbstudio, self.pulseblaster, self.chip,
                                self.ATS]
            self.interfaces = {
            instrument.name: get_instrument_interface(instrument)
            for instrument in self.instruments}

            self.interfaces['ATS'].add_acquisition_controller(
                'basic_acquisition_controller',
                cls_name='Basic_AcquisitionController')
            self.layout = Layout(name='layout',
                                 instrument_interfaces=list(
                                     self.interfaces.values()),
                                 server_name='layout_server')

            self.instruments += [self.acquisition_controller, self.layout]

        def tearDown(self):
            for instrument in self.instruments:
                instrument.close()
            for instrument_interface in self.interfaces.values():
                instrument_interface.close()

        def test_multipulse(self):
            self.layout.acquisition_outputs([])
            self.layout.primary_instrument('pulseblaster')
            self.layout.acquisition_instrument('ATS')

            print(self.layout.acquisition_instrument())

            self.layout.add_connection(output_arg='pulseblaster.ch1',
                                       input_arg='arbstudio.trig_in',
                                       trigger=True)
            self.layout.add_connection(output_arg='pulseblaster.ch2',
                                       input_arg='ATS.trig_in',
                                       trigger=True)

            c1 = self.layout.add_connection(output_arg='arbstudio.ch1',
                                            input_arg='chip.TGAC')
            c2 = self.layout.add_connection(output_arg='arbstudio.ch2',
                                            input_arg='chip.DF')
            c3 = self.layout.add_connection(output_arg='arbstudio.ch3',
                                            input_arg='ATS.chC')

            self.layout.combine_connections(c1, c2, default=True)

            self.layout.add_connection(output_arg='chip.output',
                                       input_arg='ATS.chA')
            self.layout.acquisition_outputs([('chip.output', 'output'),
                                             ('arbstudio.ch3', 'pulses')])

            trigger_connections = self.layout.get_connections(
                input_instrument='arbstudio', trigger=True)
            self.assertEqual(len(trigger_connections), 1)
            trigger_connection = trigger_connections[0]
            self.assertEqual(trigger_connection.output['instrument'],
                             'pulseblaster')

            pulse_sequence = PulseSequence()
            empty_pulse = DCPulse(name='empty', t_start=0, duration=10,
                                  amplitude=1.5)
            load_pulse = DCPulse(name='load', t_start=10, duration=10,
                                 amplitude=-1.5)
            read_pulse = DCPulse(name='read', t_start=20, duration=10,
                                 amplitude=0, acquire=True)
            pulses = [empty_pulse, load_pulse, read_pulse]
            for pulse in pulses:
                pulse_sequence.add(pulse)

            self.layout.target_pulse_sequence(pulse_sequence)
            self.layout.setup()

            print(self.pulseblaster.instructions())
            print(self.interfaces['pulseblaster'].pulse_sequence())
            self.assertEqual(len(self.pulseblaster.instructions()), 7)
    test = TestLayout()
    test.setUp()
    test.test_multipulse()
    test.tearDown()