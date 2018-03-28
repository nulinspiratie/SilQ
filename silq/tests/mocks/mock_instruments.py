import unittest

from qcodes import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

from silq.pulses import PulseSequence, DCPulse, SinePulse, MeasurementPulse
from silq.instrument_interfaces import get_instrument_interface
from silq.meta_instruments.chip import Chip
from silq.meta_instruments.layout import Layout
from silq.meta_instruments.mock_instruments import MockArbStudio, \
    MockPulseBlaster, MockATS, MockAcquisitionController

class TestInnerInstrument(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        pass

class TestOuterInstrument(Instrument):
    shared_kwargs = ['instruments']
    def __init__(self, name, instruments=[], **kwargs):
        super().__init__(name, **kwargs)
        self.instruments = instruments
        self.instrument = TestInnerInstrument('testIns', server_name='inner_server')

    def get_instruments(self):
        return self.instruments


class TestInstrument(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.add_parameter(name='x_val',
                           get_cmd=lambda: self.x,
                           vals=vals.Anything())
        self.add_parameter(name='pulse',
                           parameter_class=ManualParameter,
                           vals=vals.Anything())

        self.add_parameter(name='pulse_sequence',
                           parameter_class=ManualParameter,
                           vals=vals.Anything())

    def set_x(self, val):
        self.x = val

    def print(self):
        print('printing from test instrument')


class TestArbStudio(unittest.TestCase):
    def setUp(self):
        self.pulse_sequence = PulseSequence()
        self.arbstudio = MockArbStudio(name='mock_arbstudio', server_name='')
        self.arbstudio_interface = get_instrument_interface(self.arbstudio)

    def tearDown(self):
        self.arbstudio.close()
        self.arbstudio_interface.close()

    def test_pulse_implementation(self):
        sine_pulse = SinePulse(t_start=0, duration=10, frequency=1e6,
                               amplitude=1)
        self.assertIsNone(
            self.arbstudio_interface.get_pulse_implementation(sine_pulse))

        DC_pulse = DCPulse(t_start=0, duration=10, amplitude=1)
        self.assertIsNotNone(
            self.arbstudio_interface.get_pulse_implementation(DC_pulse))
        DC_pulse.amplitude = 3
        self.assertIsNone(
            self.arbstudio_interface.get_pulse_implementation(DC_pulse))

    def test_ELR_programming(self):
        empty_pulse = DCPulse(name='empty', t_start=0, duration=10,
                              amplitude=1.5)
        load_pulse = DCPulse(name='load', t_start=10, duration=10,
                             amplitude=-1.5)
        read_pulse = DCPulse(name='read', t_start=20, duration=10, amplitude=0)
        pulses = [empty_pulse, load_pulse, read_pulse]
        for pulse in pulses:
            targeted_pulse = self.arbstudio_interface.get_pulse_implementation(
                pulse)
            self.arbstudio_interface.pulse_sequence.add(targeted_pulse)
        self.assertEqual(len(pulses),
                         len(self.arbstudio_interface.pulse_sequence.pulses))


class TestATS(unittest.TestCase):
    def setUp(self):
        silent = True
        self.ATS = MockATS(name='ATS', server_name='ATS_server', silent=silent)
        self.ATS.config(sample_rate=100000)
        self.acquisition_controller = MockAcquisitionController(
            name='basic_acquisition_controller',
            alazar_name='ATS',
            server_name='ATS_server',
            silent=silent)

    def tearDown(self):
        self.ATS.close()
        self.acquisition_controller.close()
        self.ATS_interface.close()

    def test_initialize_with_acquisition_controller(self):
        from silq.instrument_interfaces.AlazarTech.ATS_interface import \
            ATSInterface
        self.ATS_interface = ATSInterface(instrument_name='ATS',
                                          acquisition_controller_names=[
                                              'basic_acquisition_controller'],
                                          server_name='ATS_server')

    def test_add_acquisition_controller(self):
        self.ATS_interface = get_instrument_interface(self.ATS)
        self.ATS_interface.add_acquisition_controller(
            'basic_acquisition_controller')

    def test_interface_settings(self):
        self.ATS_interface = get_instrument_interface(self.ATS)
        self.ATS_interface.add_acquisition_controller(
            'basic_acquisition_controller')

        self.ATS_interface.set_configuration_settings(trigger_slope1='positive')
        self.assertEqual(self.ATS_interface.setting('trigger_slope1'),
                         'positive')

        self.ATS_interface.update_settings(trigger_slope2='positive')
        self.assertEqual(self.ATS_interface.setting('trigger_slope1'),
                         'positive')
        self.ATS_interface.update_settings(trigger_slope1='negative')
        self.assertEqual(self.ATS_interface.setting('trigger_slope1'),
                         'negative')
        self.assertEqual(self.ATS_interface.setting('trigger_slope2'),
                         'positive')

    def test_setup_ATS(self):
        self.ATS_interface = get_instrument_interface(self.ATS)
        self.ATS_interface.add_acquisition_controller(
            'basic_acquisition_controller')

        self.ATS_interface.update_settings(trigger_engine1='J',
                                           trigger_source1='positive',
                                           records_per_buffer=True)
        self.assertNotIn('trigger_engine1', self.ATS.configuration_settings())
        self.assertNotIn('records_per_buffer',
                         self.ATS.configuration_settings())
        self.ATS_interface.setup_ATS()
        self.assertIn('trigger_engine1', self.ATS.configuration_settings())
        self.assertNotIn('records_per_buffer',
                         self.ATS.configuration_settings())

    def test_setup_acquisition_controller(self):
        self.ATS_interface = get_instrument_interface(self.ATS)
        self.ATS_interface.add_acquisition_controller(
            'basic_acquisition_controller',
            cls_name='Basic_AcquisitionController')
        pulse = MeasurementPulse(t_start=10, t_stop=20)
        self.ATS_interface.pulse_sequence.add(pulse)
        self.ATS_interface.average_mode('point')
        self.ATS_interface.setup()

        self.ATS_interface.update_settings(trigger_engine1='J',
                                           trigger_source1='positive',
                                           records_per_buffer=True)
        self.assertNotIn('trigger_engine1', self.ATS.configuration_settings())
        self.assertNotIn('records_per_buffer',
                         self.ATS.configuration_settings())
        self.assertEqual(self.acquisition_controller.average_mode(), 'point')

        self.ATS_interface.setup_acquisition_controller()
        self.assertNotIn('trigger_engine1',
                         self.acquisition_controller.acquisition_settings())
        self.assertIn('records_per_buffer',
                      self.acquisition_controller.acquisition_settings())


class TestLayout(unittest.TestCase):
    def setUp(self):
        silent = True
        self.arbstudio = MockArbStudio(name='arbstudio', silent=silent)
        self.pulseblaster = MockPulseBlaster(name='pulseblaster', silent=silent)
        self.chip = Chip(name='chip')

        self.ATS = MockATS(name='ATS', server_name='ATS_server', silent=silent)
        self.ATS.config(sample_rate=100000)
        self.acquisition_controller = MockAcquisitionController(
            name='basic_acquisition_controller',
            alazar_name='ATS',
            silent=silent)

        self.instruments = [self.arbstudio, self.pulseblaster, self.chip,
                            self.ATS]
        self.interfaces = {instrument.name: get_instrument_interface(instrument)
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

    def test_connections(self):
        self.layout.add_connection(output_arg='arbstudio.ch1',
                                   input_arg='chip.TGAC', default=True)
        self.layout.add_connection(output_arg='arbstudio.ch2',
                                   input_arg='chip.DF')
        self.layout.primary_instrument('arbstudio')
        connections = self.layout.get_connections()
        self.assertEqual(len(connections), 2)

        connections = self.layout.get_connections(input_channel='DF')
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0].input['channel'].name, 'DF')

        DC_pulse = DCPulse(t_start=0, duration=10, amplitude=1)
        connection = self.layout.get_pulse_connection(DC_pulse)
        self.assertEqual(connection.output['channel'].name, 'ch1')

    def test_pulse_implementation(self):
        self.layout.add_connection(output_arg='arbstudio.ch1',
                                   input_arg='chip.TGAC', default=True)
        self.layout.add_connection(output_arg='arbstudio.ch2',
                                   input_arg='chip.DF')
        self.layout.primary_instrument('arbstudio')

        sine_pulse = SinePulse(t_start=0, duration=10, frequency=1e6,
                               amplitude=1)
        with self.assertRaises(Exception):
            pulse_instrument = self.layout.get_pulse_instrument(sine_pulse)

        DC_pulse = DCPulse(t_start=0, duration=10, amplitude=1)
        pulse_instrument = self.layout.get_pulse_instrument(DC_pulse)
        self.assertEqual(pulse_instrument, 'arbstudio')

        DC_pulse2 = DCPulse(t_start=0, duration=10, amplitude=4)
        with self.assertRaises(Exception):
            pulse_instrument = self.layout.get_pulse_instrument(DC_pulse2)

    def test_setup_pulses(self):
        self.layout.primary_instrument('arbstudio')
        self.layout.add_connection(output_arg='arbstudio.ch1',
                                   input_arg='chip.TGAC')
        self.layout.add_connection(output_arg='arbstudio.ch2',
                                   input_arg='chip.DF', default=True)

        pulse_sequence = PulseSequence()
        empty_pulse = DCPulse(name='empty', t_start=0, duration=10,
                              amplitude=1.5)
        load_pulse = DCPulse(name='load', t_start=10, duration=10,
                             amplitude=-1.5)
        read_pulse = DCPulse(name='read', t_start=20, duration=10, amplitude=0)
        pulses = [empty_pulse, load_pulse, read_pulse]
        for pulse in pulses:
            pulse_sequence.add(pulse)

        self.layout.pulse_sequence = pulse_sequence
        self.layout.setup()

        waveforms = self.arbstudio.get_waveforms()
        for channel in [0, 2, 3]:
            self.assertEqual(len(waveforms[channel]), 0)
        self.assertEqual(len(waveforms[1]), 3)
        sequence = [self.arbstudio.ch1_sequence(),
                    self.arbstudio.ch2_sequence(),
                    self.arbstudio.ch3_sequence(),
                    self.arbstudio.ch4_sequence()]
        for channel in [0, 2, 3]:
            self.assertEqual(len(sequence[channel]), 0)
        self.assertEqual(len(sequence[1]), 3)

    def test_arbstudio_pulseblaster(self):
        self.layout.acquisition_channels([])
        self.layout.primary_instrument('pulseblaster')
        self.layout.acquisition_instrument('ATS')
        self.layout.add_connection(output_arg='arbstudio.ch1',
                                   input_arg='chip.TGAC')
        self.layout.add_connection(output_arg='arbstudio.ch2',
                                   input_arg='chip.DF', default=True)

        self.layout.add_connection(output_arg='pulseblaster.ch1',
                                   input_arg='arbstudio.trig_in',
                                   trigger=True)

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
        read_pulse = DCPulse(name='read', t_start=20, duration=10, amplitude=0)
        pulses = [empty_pulse, load_pulse, read_pulse]
        for pulse in pulses:
            pulse_sequence.add(pulse)

        self.layout.pulse_sequence = pulse_sequence
        self.layout.setup()

        self.assertEqual(len(self.pulseblaster.instructions()), 7)
        self.assertEqual([ins[0] for ins in self.pulseblaster.instructions()],
                         [1, 0, 1, 0, 1, 0, 0])
        self.assertEqual(self.pulseblaster.instructions()[-1][2], 1)

        self.pulseblaster.instructions([])
        self.interfaces['pulseblaster'].ignore_first_trigger(True)
        self.layout.pulse_sequence = pulse_sequence
        self.layout.setup()

        self.assertEqual(len(self.pulseblaster.instructions()), 7)
        self.assertEqual([ins[0] for ins in self.pulseblaster.instructions()],
                         [0, 0, 1, 0, 1, 0, 1])

    def test_full_setup(self):
        # Setup connectivity to match the Berdina setup
        self.layout.primary_instrument('pulseblaster')
        self.layout.acquisition_instrument('ATS')
        self.layout.add_connection(output_arg='pulseblaster.ch1',
                                   input_arg='arbstudio.trig_in',
                                   trigger=True)
        self.layout.add_connection(output_arg='pulseblaster.ch2',
                                   input_arg='ATS.trig_in',
                                   trigger=True)

        self.layout.add_connection(output_arg='arbstudio.ch1',
                                   input_arg='chip.TGAC')
        self.layout.add_connection(output_arg='arbstudio.ch2',
                                   input_arg='chip.DF', default=True)
        self.layout.add_connection(output_arg='arbstudio.ch3',
                                   input_arg='ATS.chC')

        self.layout.add_connection(output_arg='chip.output',
                                   input_arg='ATS.chA')
        self.layout.acquisition_channels([('chip.output', 'output'),
                                         ('arbstudio.ch3', 'pulses')])

        # Setup pulses
        pulse_sequence = PulseSequence()
        empty_pulse = DCPulse(name='empty', t_start=0, duration=10,
                              amplitude=1.5)
        load_pulse = DCPulse(name='load', t_start=10, duration=10,
                             amplitude=-1.5)
        read_pulse = DCPulse(name='read', t_start=20, duration=10, amplitude=0,
                             acquire=True)
        pulses = [empty_pulse, load_pulse, read_pulse]
        for pulse in pulses:
            pulse_sequence.add(pulse)

        self.layout.pulse_sequence = pulse_sequence
        self.layout.setup()

        # Test Pulseblaster
        self.assertEqual(len(self.pulseblaster.instructions()), 7)
        self.assertEqual([ins[0] for ins in self.pulseblaster.instructions()],
                         [1, 0, 1, 0, 3, 0, 0])
        self.assertEqual(self.pulseblaster.instructions()[-1][2], 1)

        # Test ATS
        ATS_pulse_sequence = self.interfaces['ATS'].pulse_sequence
        self.assertEqual(self.interfaces['ATS'].active_acquisition_controller(),
                         'basic_acquisition_controller')
        self.assertEqual(self.interfaces['ATS'].trigger_slope(), 'positive')
        self.assertTrue(0 < self.interfaces['ATS'].trigger_threshold() < 3.3)

        self.assertEqual(len(ATS_pulse_sequence), 1)
        configuration_settings = self.ATS.configuration_settings()
        acquisition_settings = self.acquisition_controller.acquisition_settings()
        self.assertEqual(configuration_settings['trigger_engine1'], 'J')
        self.assertEqual(acquisition_settings['channel_selection'], 'AC')
        self.assertEqual(configuration_settings['sample_rate'], 100000)

        # Test acquisition
        self.layout.start()
        signal = self.layout.acquisition()
        self.assertEqual(len(signal), 2)
        for ch_label in ['output', 'pulses']:
            idx = self.layout.acquisition.names.index('signal_output')
            signal_channel = signal[idx]
            self.assertEqual(signal_channel[0], idx)
            self.assertEqual(signal_channel.size, (
            self.interfaces['ATS'].setting('samples_per_record')))

    def test_multipulse(self):
        self.layout.acquisition_channels([])
        self.layout.primary_instrument('pulseblaster')
        self.layout.acquisition_instrument('ATS')

        self.layout.add_connection(output_arg='pulseblaster.ch1',
                                   input_arg='arbstudio.trig_in',
                                   trigger=True)
        self.layout.add_connection(output_arg='pulseblaster.ch2',
                                   input_arg='ATS.trig_in',
                                   trigger=True)

        c1 = self.layout.add_connection(output_arg='arbstudio.ch1',
                                        input_arg='chip.TGAC',
                                        pulse_modifiers={'amplitude_scale': 1})
        c2 = self.layout.add_connection(output_arg='arbstudio.ch2',
                                        input_arg='chip.DF',
                                        pulse_modifiers={
                                            'amplitude_scale': -1.5})
        c3 = self.layout.add_connection(output_arg='arbstudio.ch3',
                                        input_arg='ATS.chC')

        self.layout.combine_connections(c1, c2, default=True)

        self.layout.add_connection(output_arg='chip.output',
                                   input_arg='ATS.chA')
        self.layout.acquisition_channels([('chip.output', 'output'),
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
        read_pulse = DCPulse(name='read', t_start=20, duration=10, amplitude=0,
                             acquire=True)
        pulses = [empty_pulse, load_pulse, read_pulse]
        for pulse in pulses:
            pulse_sequence.add(pulse)

        self.layout.pulse_sequence = pulse_sequence
        self.layout.setup(samples=100)

        # Test pulseblaster
        self.assertEqual(len(self.interfaces['pulseblaster'].pulse_sequence),
                         4)
        self.assertEqual(len(self.pulseblaster.instructions()), 7)
        self.assertEqual([ins[0] for ins in self.pulseblaster.instructions()],
                         [1, 0, 1, 0, 3, 0, 0])
        self.assertEqual(self.pulseblaster.instructions()[-1][2], 1)

        # Test arbstudio
        pulse_sequence = self.interfaces['arbstudio'].pulse_sequence
        self.assertEqual(len(pulse_sequence), 6)
        pulses_ch1 = pulse_sequence.get_pulses(output_arg='arbstudio.ch1')
        self.assertEqual([p.t_start for p in pulses_ch1], [0, 10, 20])
        self.assertEqual([p.amplitude for p in pulses_ch1], [1.5, -1.5, 0])
        pulses_ch2 = pulse_sequence.get_pulses(output_arg='arbstudio.ch2')
        self.assertEqual([p.t_start for p in pulses_ch2], [0, 10, 20])
        self.assertEqual([p.amplitude for p in pulses_ch2], [-2.25, 2.25, 0])

        # Test ATS
        pulse_sequence = self.interfaces['ATS'].pulse_sequence
        input_pulse_sequence = self.interfaces['ATS'].input_pulse_sequence
        self.assertEqual(len(pulse_sequence), 2)
        self.assertEqual(len(input_pulse_sequence), 1)
        configuration_settings = self.interfaces['ATS'].configuration_settings()
        acquisition_settings = self.interfaces['ATS'].acquisition_settings()
        self.assertEqual(acquisition_settings['buffers_per_acquisition'], 100)
        self.assertTrue(
            abs(acquisition_settings['samples_per_record'] - 1000) < 16)
        self.assertEqual(acquisition_settings['channel_selection'], 'AC')