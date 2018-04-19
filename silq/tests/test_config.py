import unittest
import tempfile
import os

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

        filepath = os.path.join(self.folder.name, f'{self.config.name}.json')
        folderpath = os.path.join(self.folder.name, self.config.name)
        self.assertFalse(os.path.exists(filepath))
        self.assertTrue(os.path.isdir(folderpath))
        for filename in self.config.keys():
            filepath = os.path.join(folderpath, '{}.json'.format(filename))
            self.assertTrue(os.path.exists(filepath))

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertFalse(config_loaded.save_as_dir)
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

    def test_load_no_update(self):
        self.config.save(folder=self.folder.name, save_as_dir=True)
        self.config.pulses.pop('read')
        old_config = self.config.load(update=False)
        self.assertIn('read', old_config['pulses'])
        self.assertNotIn('read', self.config.pulses)
        self.assertEqual(old_config, self.d)
        self.assertNotEqual(self.config, self.d)

    def test_save_load_file(self):
        self.config.save(folder=self.folder.name)

        filepath = os.path.join(self.folder.name, f'{self.config.name}.json')
        folderpath = os.path.join(self.folder.name, self.config.name)
        self.assertTrue(os.path.exists(filepath))
        self.assertFalse(os.path.exists(folderpath))

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertFalse(config_loaded.save_as_dir)
        self.assertTrue(self.dicts_equal(self.config, config_loaded))

    def test_save_load_dependent(self):
        self.config.pulses.read2 = {'t_start': 1,
                                    't_stop': 'config:pulses.read.t_stop'}
        self.assertEqual(self.config.pulses.read2.t_stop,
                         self.config.pulses.read.t_stop)

        self.assertEqual(dict.__getitem__(self.config.pulses.read2, 't_stop'),
                         'config:pulses.read.t_stop')

        self.config.save(folder=self.folder.name)

        config_loaded = DictConfig('env1', folder=self.folder.name)
        self.assertEqual(config_loaded.pulses.read2.t_stop,
                         self.config.pulses.read.t_stop)
        self.assertEqual(dict.__getitem__(config_loaded.pulses.read2, 't_stop'),
                         'config:pulses.read.t_stop')

    def test_refresh_add_list_config(self):
        self.d['x'] = [1,2,3]
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config)
        self.assertEqual(self.config.x, self.d['x'])
        self.assertEqual(self.config, self.d)

    def test_refresh_add_list_subconfig(self):
        self.d['pulses']['x'] = [1,2,3]
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config.pulses)
        self.assertEqual(self.config.pulses.x, self.d['pulses']['x'])
        self.assertEqual(self.config, self.d)

    def test_refresh_add_dict_config(self):
        self.d['x'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config)
        self.assertEqual(self.config.x, self.d['x'])
        self.assertEqual(self.config, self.d)

    def test_refresh_add_dict_subconfig(self):
        self.d['pulses']['x'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertIn('x', self.config.pulses)
        self.assertEqual(self.config.pulses.x, self.d['pulses']['x'])
        self.assertEqual(self.config, self.d)

    def test_override_refresh(self):
        self.d['pulses']['read'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertIn('read', self.config.pulses)
        self.assertEqual(self.config, self.d)

    def test_override_refresh_remove_DictConfig(self):
        self.d['pulses'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertEqual(self.config, self.d)

    def test_refresh_exclusive_set(self):
        # Override dict setitem to record all items that are set
        dict_set_items = []
        dict_setitem = DictConfig.__setitem__
        def record_dict_setitem(self, key, val):
            dict_set_items.append((key, val))
            dict_setitem(self, key, val)
        DictConfig.__setitem__ = record_dict_setitem

        self.d['x'] = {'test': 'val'}
        self.config.refresh(config=self.d)
        self.assertEqual(dict_set_items, [('x', {'test': 'val'}),
                                          ('test', 'val')])


    def test_add_SubConfig(self):
        subconfig = DictConfig(name='pulses',
                               folder=None,
                               config={'read': {}})
        config.env1 = {'pulses': subconfig}
        self.assertIsInstance(config.env1, DictConfig)
        self.assertEqual(subconfig.config_path, 'config:env1.pulses')

    def test_add_ListConfig(self):
        config.env1 = [1,2,3]
        self.assertIsInstance(config.env1, ListConfig)

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


class TestConfigInheritance(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.d = {
            'pulses': {
                'read': {'t_start': 0,
                         't_stop': 10}},
            'connections': ['connection1', 'connection2']}
        self.config = DictConfig('env1', folder=self.folder.name,
                                 config=self.d)

    def test_inherit_relative(self):
        self.config.pulses.read2 = {'inherit': 'read'}
        self.assertIn('t_start', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_start,
                         self.config.pulses.read.t_start)
        self.assertIn('t_stop', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_stop,
                         self.config.pulses.read.t_stop)

    def test_inherit_absolute(self):
        self.config.pulses2 = {'read':
                                   {'t_start': 1,
                                    't_stop': 21}}
        self.config.pulses.read2 = {'inherit': 'config:pulses2.read'}
        self.assertIn('t_start', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_start,
                         self.config.pulses2.read.t_start)
        self.assertIn('t_stop', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_stop,
                         self.config.pulses2.read.t_stop)

    def test_override_inherit(self):
        self.config.pulses.read2 = {'inherit': 'read', 't_stop': 20}
        self.assertIn('t_start', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_start,
                         self.config.pulses.read.t_start)
        self.assertIn('t_stop', self.config.pulses.read2)
        self.assertEqual(self.config.pulses.read2.t_stop, 20)



class TestPulseEnvironment(unittest.TestCase):
    def setUp(self):
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

if __name__ == '__main__':
    unittest.main()