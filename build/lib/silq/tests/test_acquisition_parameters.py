import unittest

from silq import config
from silq.tools.config import DictConfig
from silq.parameters import AcquisitionParameter

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

if __name__ == '__main__':
    unittest.main()