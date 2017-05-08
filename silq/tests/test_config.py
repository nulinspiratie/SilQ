import unittest
import tempfile

import silq
from silq.tools.config import *
from silq import config
from silq.pulses import Pulse


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2']}
        self.config = DictConfig('env1', folder=self.folder.name,
                                 config=self.d)

    def tearDown(self):
        self.folder.cleanup()

    def test_dicts_equal(self):
        self.assertTrue(self.dicts_equal(self.d, self.d))
        self.assertTrue(self.dicts_equal(self.d, self.config))
        self.assertTrue(self.dicts_equal(self.config, self.config))
        self.config.x = 2
        self.assertFalse(self.dicts_equal(self.d, self.config))

    def test_save_load_dir(self):
        self.config.save(folder=self.folder.name, save_as_dir=True)

        filepath = os.path.join(self.folder.name,
                                '{}.json'.format(self.config.name))
        folderpath = os.path.join(self.folder.name, self.config.name)
        self.assertFalse(os.path.exists(filepath))
        self.assertTrue(os.path.isdir(folderpath))
        for filename in self.config.keys():
            filepath = os.path.join(folderpath, '{}.json'.format(filename))
            self.assertTrue(os.path.exists(filepath))

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertTrue(config_loaded.save_as_dir)
        self.assertTrue(self.dicts_equal(self.config, config_loaded))

        new_folder = tempfile.TemporaryDirectory()
        config_loaded.pulses.save_as_dir = True
        config_loaded.save(folder=new_folder.name)

        os.path.isdir(os.path.join(new_folder.name, 'pulses'))
        os.path.isfile(os.path.join(new_folder.name, 'pulses', 'read.json'))
        os.path.isfile(os.path.join(new_folder.name, 'connections.json'))

        config_loaded2 = DictConfig('env1', folder=new_folder.name)
        self.assertTrue(self.dicts_equal(self.config, config_loaded2))
        self.assertIsInstance(config_loaded2.connections, ListConfig)

    def test_save_load_file(self):
        self.config.save(folder=self.folder.name)

        filepath = os.path.join(self.folder.name,
                                '{}.json'.format(self.config.name))
        folderpath = os.path.join(self.folder.name, self.config.name)
        self.assertTrue(os.path.exists(filepath))
        self.assertFalse(os.path.exists(folderpath))

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertFalse(config_loaded.save_as_dir)
        self.assertTrue(self.dicts_equal(self.config, config_loaded))

    def dicts_equal(self, d1, d2):
        d1_keys = list(d1.keys())
        for d2_key, d2_val in d2.items():
            if d2_key not in d1_keys:
                return False
            else:
                d1_keys.remove(d2_key)
                d1_val = d1[d2_key]
                if isinstance(d1_val, dict):
                    if not isinstance(d2_val, dict):
                        return False
                    if not self.dicts_equal(d1_val, d2_val):
                        return False
                elif not d1_val == d2_val:
                    return False
        return True


class TestPulseEnvironment(unittest.TestCase):
    def setUp(self):
        config
        config.properties = DictConfig('properties')
        config.env1 = {
            'pulses': {
                'read': {}}}
        config.properties.default_environment = 'env1'
        self.p = Pulse(name='read')

        self.pulse_config = silq.config.env1.pulses.read

    def test_attribute_from_config(self):
        self.assertEqual(self.p.environment, 'env1')
        self.assertIsNone(self.p.t_start)

        self.pulse_config.t_start = 0
        self.assertEqual(self.p.t_start, 0)
        self.p.t_start = 5
        self.assertEqual(self.p.t_start, 5)
        self.pulse_config.t_start = 0
        self.assertEqual(self.p.t_start, 0)

    def test_change_environment(self):
        config.env2 = {
            'pulses': {
                'read': {}}}
        self.pulse_config.t_start = 0
        self.assertEqual(self.p.t_start, 0)

        self.p.environment = 'env2'
        self.assertEqual(self.p.t_start, 0)
        config.env2.pulses.read.t_start = 1
        self.assertEqual(self.p.t_start, 1)

        config.env1.pulses.read.t_start = 2
        self.assertEqual(self.p.t_start, 1)

        self.p.environment = 'env1'
        self.assertEqual(self.p.t_start, 2)