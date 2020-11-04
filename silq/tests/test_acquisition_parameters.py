import unittest

import silq
from silq import config
from silq.tools.config import DictConfig
from silq.parameters import AcquisitionParameter

import qcodes as qc

# Need this to ensure that no error is raised when creating AcquisitionParameter
AcquisitionParameter.get = lambda self: 42


class TestAcquisitionParameterConfig(unittest.TestCase):
    def setUp(self):
        config.clear()
        config.properties = {'default_environment': 'test_env'}

    def tearDown(self):
        config.clear()
        config.properties = {}

    def test_no_env(self):
        p = AcquisitionParameter(name='test_param',
                                 names=[],
                                 properties_attrs=['attr1'])
        self.assertIsNone(p.properties_config)
        self.assertIsNone(p.parameter_config)

        config.test_env = {}
        p = AcquisitionParameter(name='test_param',
                                 names=[],
                                 properties_attrs=['attr1'])
        self.assertIsNone(p.properties_config)
        self.assertIsNone(p.parameter_config)

        config.test_env.parameters = {}
        p = AcquisitionParameter(name='test_param',
                                 names=[],
                                 properties_attrs=['attr1'])
        self.assertIsNone(p.properties_config)
        self.assertIsNone(p.parameter_config)

        config.test_env.parameters.test_param = {}
        config.test_env.properties = {}
        p = AcquisitionParameter(name='test_param',
                                 names=[],
                                 properties_attrs=['attr1'])
        self.assertIsInstance(p.properties_config, DictConfig)
        self.assertIsInstance(p.parameter_config, DictConfig)


    def test_properties_config(self):
        config.test_env = {'properties': {}}
        p = AcquisitionParameter(name='test_param',
                                 names=[],
                                 properties_attrs=['attr1'])
        self.assertIsInstance(p.properties_config, DictConfig)
        self.assertFalse(hasattr(p, 'attr1'))

        config.test_env.properties.attr1 = 42
        self.assertIn('attr1', p.properties_config)
        self.assertEqual(p.attr1, 42)

        p.attr1 = 43
        self.assertEqual(p.attr1, 43)

        config.test_env.properties.attr1 = 42
        self.assertIn('attr1', p.properties_config)
        self.assertEqual(p.attr1, 42)

    def test_parameters_config(self):
        config.test_env = {'parameters': {}}
        p = AcquisitionParameter(name='test_param', names=[])
        self.assertIsNone(p.parameter_config)

        config.test_env.parameters.test_param = {}
        p = AcquisitionParameter(name='test_param', names=[])
        self.assertIsInstance(p.parameter_config, DictConfig)

        config.test_env.parameters.test_param.samples = 100
        self.assertEqual(p.samples, 100)

        p = AcquisitionParameter(name='test_param', names=[])
        self.assertEqual(p.samples, 100)


class TestESRParameterComposite(unittest.TestCase):
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

    def test_create_ESR_parameter(self):
        from silq.parameters.acquisition_parameters_composite import ESRParameterComposite

        ESR_parameter = ESRParameterComposite()

        names = [
            'EPR.up_proportion',
            'EPR.contrast',
            'EPR.dark_counts',
            'EPR.voltage_difference_read',
            'ESR.up_proportion',
            'ESR.contrast',
            'ESR.num_traces',
            'ESR.voltage_difference',
            'ESR.threshold_voltage'
        ]
        self.assertEqual(len(names), len(ESR_parameter.names))
        for name in names:
            self.assertIn(name, ESR_parameter.names)

    def test_ESR_parameter_disable_EPR(self):
        from silq.parameters.acquisition_parameters_composite import ESRParameterComposite
        ESR_parameter = ESRParameterComposite()

        ESR_parameter.EPR.enabled = False

        names = [
            'ESR.up_proportion',
            'ESR.num_traces',
            'ESR.voltage_difference',
            'ESR.threshold_voltage'
        ]
        self.assertEqual(len(names), len(ESR_parameter.names))
        for name in names:
            self.assertIn(name, ESR_parameter.names)

    def test_ESR_parameter_multiple_frequencies(self):
        from silq.parameters.acquisition_parameters_composite import ESRParameterComposite
        ESR_parameter = ESRParameterComposite()

        ESR_parameter.frequencies = [12e9, 13e9]
        self.assertEqual(ESR_parameter.analyses.ESR.settings.num_frequencies, 2)

        default_names = [
            'EPR.up_proportion',
            'EPR.contrast',
            'EPR.dark_counts',
            'EPR.voltage_difference_read',
            'ESR.voltage_difference',
            'ESR.threshold_voltage'
        ]
        variable_names = [
            'ESR.up_proportion0',
            'ESR.up_proportion1',
            'ESR.contrast0',
            'ESR.contrast1',
            'ESR.num_traces0',
            'ESR.num_traces1'
        ]
        names = [*default_names, *variable_names]
        self.assertEqual(len(names), len(ESR_parameter.names))
        for name in names:
            self.assertIn(name, ESR_parameter.names)

        ESR_parameter.analyses.ESR.settings.labels = ['A', 'B']
        variable_names = [
            'ESR.up_proportion_A',
            'ESR.up_proportion_B',
            'ESR.contrast_A',
            'ESR.contrast_B',
            'ESR.num_traces_A',
            'ESR.num_traces_B'
        ]
        names = [*default_names, *variable_names]
        self.assertEqual(len(names), len(ESR_parameter.names))
        for name in names:
            self.assertIn(name, ESR_parameter.names)


if __name__ == '__main__':
    unittest.main()
